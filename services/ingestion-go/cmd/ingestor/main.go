package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gotd/td/tg"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"golang.org/x/sync/errgroup"

	"github.com/tg-agg/ingestion-go/internal/api"
	"github.com/tg-agg/ingestion-go/internal/comments"
	"github.com/tg-agg/ingestion-go/internal/config"
	"github.com/tg-agg/ingestion-go/internal/domain"
	"github.com/tg-agg/ingestion-go/internal/media"
	"github.com/tg-agg/ingestion-go/internal/mtproto"
	"github.com/tg-agg/ingestion-go/internal/publisher"
	boltstore "github.com/tg-agg/ingestion-go/internal/storage/bbolt"
	"github.com/tg-agg/ingestion-go/internal/storage/postgres"
	"github.com/tg-agg/ingestion-go/internal/updates"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, logger); err != nil && !errors.Is(err, context.Canceled) {
		logger.Error("ingestion-go exited with error", "err", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger) error {
	cfg, err := config.LoadFromEnv()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	var (
		pgPool *pgxpool.Pool
		repo   *postgres.Repository
	)
	if cfg.DatabaseURL != "" {
		pgPool, err = pgxpool.New(ctx, cfg.DatabaseURL)
		if err != nil {
			return fmt.Errorf("connect postgres: %w", err)
		}
		defer pgPool.Close()
		repo = postgres.NewRepository(pgPool)
	} else {
		logger.Warn("DATABASE_URL is empty; running without PostgreSQL persistence")
	}

	rdb := redis.NewClient(&redis.Options{
		Addr:     cfg.RedisAddr,
		Password: cfg.RedisPassword,
		DB:       cfg.RedisDB,
	})
	defer rdb.Close()
	pub := publisher.NewRedisStreamsPublisher(rdb)

	diskCache, err := media.NewLocalDiskCacheProvider(cfg.MediaCacheDir, cfg.MediaCacheTTL)
	if err != nil {
		return fmt.Errorf("init media disk cache: %w", err)
	}

	scheduler := mtproto.NewStaticSingleTierScheduler(repo, repo, logger)
	sessionKey := cfg.SessionKeys[0]
	bboltPath := filepath.Join(cfg.BBoltDir, sessionKey+".db")
	bStore, err := boltstore.Open(bboltPath)
	if err != nil {
		return fmt.Errorf("open bbolt store: %w", err)
	}
	defer func() {
		if closeErr := bStore.Close(); closeErr != nil {
			logger.Warn("close bbolt failed", "err", closeErr)
		}
	}()

	accountID := 1
	if repo != nil {
		phoneDigest := sha256.Sum256([]byte(sessionKey))
		phoneHash := hex.EncodeToString(phoneDigest[:])
		id, err := repo.UpsertSession(ctx, domain.SessionRecord{
			PhoneHash:     phoneHash,
			FSMState:      domain.SessionActivePolling,
			BBoltFilePath: bboltPath,
		})
		if err != nil {
			return fmt.Errorf("upsert mtproto session row: %w", err)
		}
		accountID = id
		scheduler.RegisterAccount(domain.SessionRecord{
			AccountID:     accountID,
			PhoneHash:     phoneHash,
			FSMState:      domain.SessionActivePolling,
			BBoltFilePath: bboltPath,
		})
	} else {
		scheduler.RegisterAccount(domain.SessionRecord{
			AccountID: accountID,
			FSMState:  domain.SessionActivePolling,
		})
	}

	safeStateStor := updates.NewWatermarkSafeStateStorage(bStore.StateStorage())
	disp := tg.NewUpdateDispatcher()

	var (
		commentSvc *comments.DeltaSyncService
		pipeline   *updates.ChannelPipeline
	)

	onComments := func(cCtx context.Context, channelID int64, postMsgID int, rep domain.RepliesInfo) {
		if commentSvc == nil {
			return
		}
		go func() {
			syncCtx, cancel := context.WithTimeout(cCtx, 30*time.Second)
			defer cancel()
			if _, syncErr := commentSvc.SyncPostComments(syncCtx, channelID, postMsgID, rep); syncErr != nil {
				logger.Warn("async delta comment sync failed",
					"channel_id", channelID,
					"msg_id", postMsgID,
					"err", syncErr,
				)
			}
		}()
	}

	pipeline = updates.NewChannelPipeline(
		accountID,
		cfg.AlbumBufferWindow,
		repo,
		repo,
		pub,
		safeStateStor,
		onComments,
		logger,
	)
	pipeline.RegisterHandlers(disp)

	var (
		managed     *mtproto.ManagedClient
		mediaStream *media.Streamer
		poller      *updates.Poller
	)

	resolver := mtproto.NewUsernameResolver(cfg.ResolveMinInterval, repo, repo)

	if cfg.TGAppID > 0 && cfg.TGAppHash != "" {
		managed, err = mtproto.NewManagedClient(mtproto.ManagedClientParams{
			Config:        cfg,
			AccountID:     accountID,
			SessionKey:    sessionKey,
			Store:         bStore,
			StateStorage:  safeStateStor,
			UpdateHandler: disp,
			Scheduler:     scheduler,
			Watermarks:    repo,
			AccessHashes:  repo,
			GapSink:       pipeline,
			Logger:        logger,
		})
		if err != nil {
			return fmt.Errorf("create managed MTProto client: %w", err)
		}

		apiClient := managed.API()
		commentSvc = comments.NewDeltaSyncService(accountID, apiClient, repo, repo, pub, logger)
		poller = updates.NewPoller(accountID, apiClient, repo, repo, pipeline, cfg.PollInterval, logger)
		refresher := media.NewFileRefRefresher(accountID, apiClient, repo, repo, logger)
		mediaStream = media.NewStreamer(media.GotdUploadClient{Client: managed.Client}, repo, diskCache, refresher, logger)
	} else {
		logger.Warn("TG_APP_ID / TG_APP_HASH unset; running HTTP endpoints in passive mode without live MTProto")
		mediaStream = media.NewStreamer(nil, repo, diskCache, nil, logger)
	}

	registrar := func(regCtx context.Context, username, tier string) (mtproto.ResolvedChannel, error) {
		if managed == nil {
			return mtproto.ResolvedChannel{}, errors.New("MTProto client unavailable (TG_APP_ID / TG_APP_HASH unset)")
		}
		res, err := resolver.ResolveChannel(regCtx, accountID, username, managed.API())
		if err != nil {
			return mtproto.ResolvedChannel{}, err
		}
		if tier != "" && repo != nil {
			u := res.Username
			_ = repo.UpsertChannel(regCtx, domain.ChannelStateRecord{
				ChannelID:       res.ChannelID,
				Username:        &u,
				Title:           res.Title,
				Tier:            tier,
				PollIntervalSec: int(cfg.PollInterval.Seconds()),
				NextPollAt:      time.Now().UTC(),
				IsActive:        true,
			})
		}
		if err := managed.Updates.TrackChannel(regCtx, res.ChannelID, res.AccessHash, 0); err != nil {
			logger.Warn("TrackChannel returned warning", "channel_id", res.ChannelID, "err", err)
		}
		return res, nil
	}

	httpSrv := api.NewHTTPServer(mediaStream, commentSvc, scheduler, registrar, logger)
	srv := &http.Server{
		Addr:              cfg.HTTPListenAddr,
		Handler:           httpSrv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	g, gCtx := errgroup.WithContext(ctx)

	// Periodic cache expiry cleanup.
	g.Go(func() error {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for {
			select {
			case <-gCtx.Done():
				return nil
			case <-ticker.C:
				if cleanErr := diskCache.CleanExpired(gCtx); cleanErr != nil {
					logger.Warn("media cache cleanup warning", "err", cleanErr)
				}
			}
		}
	})

	// Internal HTTP edge-proxy & health server.
	g.Go(func() error {
		logger.Info("starting HTTP server", "addr", cfg.HTTPListenAddr)
		if serveErr := srv.ListenAndServe(); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			return fmt.Errorf("http ListenAndServe: %w", serveErr)
		}
		return nil
	})

	g.Go(func() error {
		<-gCtx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
		if flushErr := pipeline.Assembler().FlushAll(shutdownCtx); flushErr != nil {
			logger.Warn("flush pending album buffers on shutdown failed", "err", flushErr)
		}
		return nil
	})

	if managed != nil {
		g.Go(func() error {
			return managed.Run(gCtx, func(readyCtx context.Context, _ int64) error {
				if poller != nil {
					return poller.Run(readyCtx, 15*time.Second)
				}
				<-readyCtx.Done()
				return nil
			})
		})
	}

	return g.Wait()
}
