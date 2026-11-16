FROM golang:1.26-alpine AS builder

WORKDIR /src

RUN apk add --no-cache ca-certificates tzdata

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/ingestor ./cmd/ingestor

FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata && \
    mkdir -p /var/lib/ingestor/bbolt /var/lib/ingestor/media_cache && \
    adduser -D -u 10001 ingestor && \
    chown -R ingestor:ingestor /var/lib/ingestor

USER ingestor
WORKDIR /app

COPY --from=builder /out/ingestor /usr/local/bin/ingestor

EXPOSE 8081

ENTRYPOINT ["/usr/local/bin/ingestor"]
