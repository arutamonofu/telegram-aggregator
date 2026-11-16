package mtproto

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"time"

	"github.com/gotd/td/telegram/dcs"
	"golang.org/x/net/proxy"
)

// NewSOCKS5Resolver builds a gotd dcs.Resolver that dials MTProto connections through proxyURL
// when provided (e.g. socks5://user:pass@host:1080), or uses a direct TCP dialer otherwise.
func NewSOCKS5Resolver(proxyURL string) (dcs.Resolver, error) {
	baseDialer := &net.Dialer{
		Timeout:   15 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	if proxyURL == "" {
		return dcs.Plain(dcs.PlainOptions{
			Dial: baseDialer.DialContext,
		}), nil
	}

	parsed, err := url.Parse(proxyURL)
	if err != nil {
		return nil, fmt.Errorf("parse SOCKS5_PROXY_URL %q: %w", proxyURL, err)
	}

	dialer, err := proxy.FromURL(parsed, baseDialer)
	if err != nil {
		return nil, fmt.Errorf("create SOCKS5 dialer: %w", err)
	}

	dialFn := func(ctx context.Context, network, addr string) (net.Conn, error) {
		if ctxDialer, ok := dialer.(proxy.ContextDialer); ok {
			return ctxDialer.DialContext(ctx, network, addr)
		}

		type dialResult struct {
			conn net.Conn
			err  error
		}
		ch := make(chan dialResult, 1)
		go func() {
			c, dialErr := dialer.Dial(network, addr)
			ch <- dialResult{conn: c, err: dialErr}
		}()

		select {
		case <-ctx.Done():
			go func() {
				res := <-ch
				if res.conn != nil {
					_ = res.conn.Close()
				}
			}()
			return nil, ctx.Err()
		case res := <-ch:
			return res.conn, res.err
		}
	}

	return dcs.Plain(dcs.PlainOptions{
		Dial: dialFn,
	}), nil
}
