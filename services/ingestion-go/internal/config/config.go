package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds Stage 1 (Dev) runtime configuration for services/ingestion-go.
type Config struct {
	DatabaseURL         string
	RedisAddr           string
	RedisPassword       string
	RedisDB             int
	BBoltDir            string
	MediaCacheDir       string
	MediaCacheTTL       time.Duration
	AlbumBufferWindow   time.Duration
	HTTPListenAddr      string
	TGAppID             int
	TGAppHash           string
	Socks5ProxyURL      string
	PollInterval        time.Duration
	ChannelDiffInterval time.Duration
	ResolveMinInterval  time.Duration
	SessionKeys         []string
}

// LoadFromEnv constructs Config from environment variables matching docker-compose.dev.yml.
func LoadFromEnv() (Config, error) {
	cfg := Config{
		DatabaseURL:         os.Getenv("DATABASE_URL"),
		RedisAddr:           getEnvOrDefault("REDIS_ADDR", "localhost:6379"),
		RedisPassword:       os.Getenv("REDIS_PASSWORD"),
		BBoltDir:            getEnvOrDefault("BBOLT_DIR", "/var/lib/ingestor/bbolt"),
		MediaCacheDir:       getEnvOrDefault("MEDIA_CACHE_DIR", "/var/lib/ingestor/media_cache"),
		HTTPListenAddr:      getEnvOrDefault("HTTP_LISTEN_ADDR", ":8081"),
		TGAppHash:           os.Getenv("TG_APP_HASH"),
		Socks5ProxyURL:      os.Getenv("SOCKS5_PROXY_URL"),
		MediaCacheTTL:       24 * time.Hour,
		AlbumBufferWindow:   1500 * time.Millisecond,
		PollInterval:        900 * time.Second,
		ChannelDiffInterval: 900 * time.Second,
		ResolveMinInterval:  5 * time.Second,
	}

	if v := os.Getenv("REDIS_DB"); v != "" {
		db, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("invalid REDIS_DB: %w", err)
		}
		cfg.RedisDB = db
	}

	if v := os.Getenv("MEDIA_CACHE_TTL_HOURS"); v != "" {
		hours, err := strconv.Atoi(v)
		if err != nil || hours <= 0 {
			return Config{}, fmt.Errorf("invalid MEDIA_CACHE_TTL_HOURS %q", v)
		}
		cfg.MediaCacheTTL = time.Duration(hours) * time.Hour
	}

	if v := os.Getenv("ALBUM_BUFFER_MS"); v != "" {
		ms, err := strconv.Atoi(v)
		if err != nil || ms <= 0 {
			return Config{}, fmt.Errorf("invalid ALBUM_BUFFER_MS %q", v)
		}
		cfg.AlbumBufferWindow = time.Duration(ms) * time.Millisecond
	}

	if v := os.Getenv("POLL_INTERVAL_SEC"); v != "" {
		sec, err := strconv.Atoi(v)
		if err != nil || sec <= 0 {
			return Config{}, fmt.Errorf("invalid POLL_INTERVAL_SEC %q", v)
		}
		cfg.PollInterval = time.Duration(sec) * time.Second
	}

	if v := os.Getenv("CHANNEL_DIFF_INTERVAL_SEC"); v != "" {
		sec, err := strconv.Atoi(v)
		if err != nil || sec <= 0 {
			return Config{}, fmt.Errorf("invalid CHANNEL_DIFF_INTERVAL_SEC %q", v)
		}
		cfg.ChannelDiffInterval = time.Duration(sec) * time.Second
	}

	if v := os.Getenv("RESOLVE_MIN_INTERVAL_MS"); v != "" {
		ms, err := strconv.Atoi(v)
		if err != nil || ms <= 0 {
			return Config{}, fmt.Errorf("invalid RESOLVE_MIN_INTERVAL_MS %q", v)
		}
		cfg.ResolveMinInterval = time.Duration(ms) * time.Millisecond
	}

	if v := os.Getenv("TG_APP_ID"); v != "" {
		appID, err := strconv.Atoi(v)
		if err != nil || appID <= 0 {
			return Config{}, fmt.Errorf("invalid TG_APP_ID %q", v)
		}
		cfg.TGAppID = appID
	}

	if raw := strings.TrimSpace(os.Getenv("TG_SESSION_KEYS")); raw != "" {
		for _, item := range strings.Split(raw, ",") {
			trimmed := strings.TrimSpace(item)
			if trimmed != "" {
				cfg.SessionKeys = append(cfg.SessionKeys, trimmed)
			}
		}
	}
	if len(cfg.SessionKeys) == 0 {
		cfg.SessionKeys = []string{"dev-session-1"}
	}

	return cfg, nil
}

func getEnvOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
