# KCS Deployment Guide

## Overview

This guide covers deploying Kernel Context Server (KCS) in various environments,
from development to production. KCS consists of multiple components that need to
be properly configured and coordinated.

## Architecture Overview

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   KCS MCP API   │    │   PostgreSQL    │
│  (nginx/HAProxy)│◄──►│   (Python)      │◄──►│   + pgvector    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                        ┌─────────────────┐
                        │  Rust Libraries │
                        │  (kcs-parser,   │
                        │   kcs-graph,    │
                        │   kcs-impact)   │
                        └─────────────────┘
```

## Prerequisites

### System Requirements

**Minimum**:

- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps

**Recommended**:

- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ NVMe SSD
- Network: 10Gbps

### Software Dependencies

- **OS**: Ubuntu 20.04+ / RHEL 8+ / Debian 11+
- **Python**: 3.11+
- **Rust**: 1.75+
- **PostgreSQL**: 15+ with pgvector extension
- **Git**: For repository access
- **Docker**: Optional, for containerized deployment

## Deployment Methods

### Method 1: Native Installation

#### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    postgresql-15 \
    postgresql-contrib \
    nginx \
    supervisor

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### 2. Database Setup

```bash
# Install pgvector extension
sudo apt install -y postgresql-15-pgvector

# Configure PostgreSQL
sudo -u postgres psql << EOF
CREATE USER kcs WITH PASSWORD 'secure_password_here';
CREATE DATABASE kcs OWNER kcs;
\\c kcs
CREATE EXTENSION IF NOT EXISTS pgvector;
GRANT ALL PRIVILEGES ON DATABASE kcs TO kcs;
EOF

# Configure PostgreSQL settings
sudo tee -a /etc/postgresql/15/main/postgresql.conf << EOF
# KCS Configuration
shared_preload_libraries = 'pgvector'
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 128MB
work_mem = 64MB
EOF

sudo systemctl restart postgresql
```

#### 3. Application Setup

```bash
# Create KCS user
sudo useradd -r -m -s /bin/bash kcs

# Clone repository
sudo -u kcs git clone https://github.com/your-org/kcs.git /opt/kcs
cd /opt/kcs

# Build Rust components
sudo -u kcs cargo build --release --workspace

# Set up Python environment
sudo -u kcs python3 -m venv venv
sudo -u kcs ./venv/bin/pip install -r requirements.txt

# Run database migrations
sudo -u kcs ./tools/setup/migrate.sh

# Generate configuration
sudo -u kcs cp config/kcs.example.yaml config/kcs.yaml
sudo -u kcs editor config/kcs.yaml  # Edit configuration
```

#### 4. Service Configuration

```bash
# Create systemd service
sudo tee /etc/systemd/system/kcs.service << EOF
[Unit]
Description=Kernel Context Server
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=exec
User=kcs
Group=kcs
WorkingDirectory=/opt/kcs
Environment=PYTHONPATH=/opt/kcs/src/python
ExecStart=/opt/kcs/venv/bin/python -m kcs_mcp.app
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/kcs/logs /opt/kcs/data

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable kcs
sudo systemctl start kcs
```

#### 5. Reverse Proxy Setup

```bash
# Configure nginx
sudo tee /etc/nginx/sites-available/kcs << EOF
upstream kcs_backend {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name kcs.example.com;
    
    # Redirect to HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name kcs.example.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/kcs.crt;
    ssl_certificate_key /etc/ssl/private/kcs.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://kcs_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://kcs_backend;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/kcs/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/kcs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Method 2: Docker Deployment

#### 1. Docker Compose Setup

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: kcs
      POSTGRES_USER: kcs
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U kcs"]
      interval: 30s
      timeout: 10s
      retries: 5

  kcs:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://kcs:${DB_PASSWORD}@postgres:5432/kcs
      - KCS_AUTH_TOKEN=${KCS_AUTH_TOKEN}
      - RUST_LOG=info
    volumes:
      - ./config:/app/config:ro
      - kernel_data:/app/data
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - kcs
    restart: unless-stopped

volumes:
  postgres_data:
  kernel_data:
```

#### 2. Dockerfile

```dockerfile
# Dockerfile
FROM rust:1.89 AS rust-builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/rust ./src/rust

# Build Rust components
RUN cargo build --release --workspace

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust binaries
COPY --from=rust-builder /app/target/release /app/bin

# Copy Python code
COPY src/python ./src/python
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration and scripts
COPY config ./config
COPY tools ./tools

# Create non-root user
RUN useradd -r -u 1000 kcs && \\
    chown -R kcs:kcs /app

USER kcs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "kcs_mcp.app"]
```

#### 3. Environment Configuration

```bash
# .env
DB_PASSWORD=secure_database_password
KCS_AUTH_TOKEN=secure_jwt_token
KCS_LOG_LEVEL=info
KCS_PORT=8080
KCS_WORKERS=4
```

#### 4. Deploy with Docker Compose

```bash
# Deploy
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f kcs

# Scale if needed
docker compose up -d --scale kcs=3
```

### Method 3: Kubernetes Deployment

#### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: kcs

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kcs-config
  namespace: kcs
data:
  kcs.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
      workers: 4
    
    database:
      url: "postgresql://kcs:password@postgres:5432/kcs"
      pool_size: 20
    
    logging:
      level: "info"
      format: "json"
    
    performance:
      query_timeout: 30
      index_timeout: 1200
```text

#### 2. PostgreSQL Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: kcs
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg15
        env:
        - name: POSTGRES_DB
          value: kcs
        - name: POSTGRES_USER
          value: kcs
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: kcs-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - kcs
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: kcs
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: kcs
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

#### 3. KCS Application Deployment

```yaml
# k8s/kcs.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kcs
  namespace: kcs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kcs
  template:
    metadata:
      labels:
        app: kcs
    spec:
      containers:
      - name: kcs
        image: kcs:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: kcs-secrets
              key: database-url
        - name: KCS_AUTH_TOKEN
          valueFrom:
            secretKeyRef:
              name: kcs-secrets
              key: auth-token
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: kernel-data
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: config
        configMap:
          name: kcs-config
      - name: kernel-data
        persistentVolumeClaim:
          claimName: kernel-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: kcs
  namespace: kcs
spec:
  selector:
    app: kcs
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: kernel-data-pvc
  namespace: kcs
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: shared-storage
```

#### 4. Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: kcs-ingress
  namespace: kcs
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - kcs.example.com
    secretName: kcs-tls
  rules:
  - host: kcs.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kcs
            port:
              number: 80
```

#### 5. Deploy to Kubernetes

```bash
# Create secrets
kubectl create secret generic kcs-secrets \
  --from-literal=db-password='secure_password' \
  --from-literal=database-url=\
    'postgresql://kcs:secure_password@postgres:5432/kcs' \
  --from-literal=auth-token='jwt_token_here' \
  -n kcs

# Deploy
kubectl apply -f k8s/

# Check status
kubectl get pods -n kcs
kubectl get services -n kcs
kubectl logs -f deployment/kcs -n kcs
```

## Configuration

### Core Configuration File

```yaml
# config/kcs.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  max_request_size: 10485760  # 10MB

database:
  url: "postgresql://kcs:password@localhost:5432/kcs"
  pool_size: 20
  max_overflow: 10
  pool_timeout: 30

auth:
  jwt_secret: "your-jwt-secret-here"
  token_expiry: 86400  # 24 hours

logging:
  level: "info"
  format: "json"
  file: "/var/log/kcs/kcs.log"

performance:
  query_timeout: 30
  index_timeout: 1200  # 20 minutes
  cache_ttl: 3600  # 1 hour

kernel:
  default_config: "x86_64:defconfig"
  supported_configs:
    - "x86_64:defconfig"
    - "x86_64:allmodconfig"
    - "arm64:defconfig"

features:
  enable_metrics: true
  enable_tracing: true
  enable_caching: true

security:
  cors_origins:
    - "https://your-domain.com"
  rate_limit:
    requests_per_minute: 100
    burst_size: 20
```

### Environment Variables

```bash
# Core settings
export KCS_CONFIG_FILE="/etc/kcs/kcs.yaml"
export KCS_LOG_LEVEL="info"
export KCS_PORT="8080"

# Database
export DATABASE_URL="postgresql://kcs:password@localhost:5432/kcs"

# Authentication
export KCS_AUTH_TOKEN="your-jwt-token"
export JWT_SECRET="your-jwt-secret"

# Performance
export KCS_WORKERS="4"
export KCS_MAX_CONNECTIONS="100"

# Features
export KCS_ENABLE_METRICS="true"
export KCS_ENABLE_CACHING="true"
```

## Performance Optimization

### Database Optimization

```sql
-- postgresql.conf optimizations
shared_buffers = 25% of RAM
effective_cache_size = 75% of RAM
maintenance_work_mem = 1GB
work_mem = 256MB
max_connections = 200

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_symbols_name ON symbols(name);
CREATE INDEX CONCURRENTLY idx_symbols_file_path ON symbols(file_path);
CREATE INDEX CONCURRENTLY idx_call_edges_caller ON call_edges(caller_id);
CREATE INDEX CONCURRENTLY idx_call_edges_callee ON call_edges(callee_id);

-- Vacuum and analyze regularly
VACUUM ANALYZE;
```

### Application Optimization

```python
# gunicorn configuration (gunicorn.conf.py)
bind = "0.0.0.0:8080"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 30
keepalive = 5

# Memory optimization
preload_app = True
worker_tmp_dir = "/dev/shm"
```

### Caching Strategy

```yaml
# Redis configuration for caching
cache:
  backend: "redis"
  redis_url: "redis://localhost:6379/0"
  default_ttl: 3600
  max_memory: "1gb"
  
  # Cache specific queries
  cache_patterns:
    - pattern: "search_code:*"
      ttl: 1800
    - pattern: "get_symbol:*"
      ttl: 3600
    - pattern: "who_calls:*"
      ttl: 7200
```

## Monitoring and Observability

### Metrics Collection

```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kcs'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Health Checks

```bash
#!/bin/bash
# health-check.sh

# Basic health check
curl -f http://localhost:8080/health || exit 1

# Database connectivity
curl -f -H "Authorization: Bearer $KCS_TOKEN" \\
  http://localhost:8080/mcp/resources || exit 1

# Performance check (should respond within 1 second)
timeout 1 curl -f http://localhost:8080/health || exit 1
```

### Logging Configuration

```yaml
# Structured logging
logging:
  version: 1
  formatters:
    json:
      format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s",
                "message": "%(message)s", "module": "%(name)s"}'
  
  handlers:
    console:
      class: logging.StreamHandler
      formatter: json
      stream: ext://sys.stdout
    
    file:
      class: logging.handlers.RotatingFileHandler
      formatter: json
      filename: /var/log/kcs/kcs.log
      maxBytes: 100000000  # 100MB
      backupCount: 5
  
  root:
    level: INFO
    handlers: [console, file]
```

## Security Hardening

### Network Security

```bash
# Firewall configuration (ufw)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# PostgreSQL access (only from KCS servers)
sudo ufw allow from 10.0.1.0/24 to any port 5432
```

### SSL/TLS Configuration

```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload";

# Other security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
```

### Application Security

```python
# Security middleware configuration
SECURITY_HEADERS = {
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Referrer-Policy': 'strict-origin-when-cross-origin'
}

# Rate limiting
RATE_LIMITS = {
    'default': '100/minute',
    'search_code': '50/minute',
    'who_calls': '30/minute'
}
```

## Backup and Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# backup-database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/kcs"
DB_NAME="kcs"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Full database backup
pg_dump -h localhost -U kcs -Fc "$DB_NAME" > "$BACKUP_DIR/kcs_backup_$DATE.dump"

# Compress backup
gzip "$BACKUP_DIR/kcs_backup_$DATE.dump"

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp "$BACKUP_DIR/kcs_backup_$DATE.dump.gz" s3://your-backup-bucket/kcs/
```

### Application Backup

```bash
#!/bin/bash
# backup-application.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/kcs-app"
APP_DIR="/opt/kcs"

# Create backup
tar -czf "$BACKUP_DIR/kcs_app_$DATE.tar.gz" \
  -C "$APP_DIR" \
  --exclude=venv \
  --exclude=target \
  --exclude=logs \
  --exclude=.git \
  .

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/kcs_app_$DATE.tar.gz" s3://your-backup-bucket/kcs-app/
```

### Disaster Recovery Plan

1. **RTO**: 4 hours (Recovery Time Objective)
2. **RPO**: 1 hour (Recovery Point Objective)

**Recovery Steps**:

1. Provision new infrastructure
2. Restore database from latest backup
3. Deploy application from backup or repository
4. Update DNS to point to new servers
5. Verify functionality

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
journalctl -u kcs -f

# Check configuration
kcs --check-config

# Test database connection
psql -h localhost -U kcs -d kcs -c "SELECT 1;"
```

#### Performance Issues

```bash
# Check system resources
htop
iotop
nethogs

# Check database performance
psql -h localhost -U kcs -d kcs -c "SELECT * FROM pg_stat_activity;"

# Check application metrics
curl http://localhost:8080/metrics
```

#### Memory Issues

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full kcs-parser

# Adjust memory limits
ulimit -m 8388608  # 8GB limit
```

### Log Analysis

```bash
# Search for errors
grep -i error /var/log/kcs/kcs.log

# Performance analysis
grep "request_duration" /var/log/kcs/kcs.log | awk '{print $8}' | sort -n

# Connection issues
grep -i "connection" /var/log/kcs/kcs.log
```

## Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# maintenance.sh

# Database maintenance
psql -h localhost -U kcs -d kcs << EOF
VACUUM ANALYZE;
REINDEX DATABASE kcs;
EOF

# Log rotation
logrotate /etc/logrotate.d/kcs

# Update system packages
apt update && apt upgrade -y

# Restart services if needed
systemctl restart kcs
systemctl restart nginx
```

### Monitoring Checklist

- [ ] Service health checks passing
- [ ] Database performance metrics normal
- [ ] Disk space usage < 80%
- [ ] Memory usage < 80%
- [ ] Response times < 600ms (p95)
- [ ] Error rate < 1%
- [ ] Backup jobs completing successfully
- [ ] SSL certificates valid (> 30 days remaining)

## Support and Contact

For deployment support:

- Documentation: <https://docs.kcs.example.com>
- Support Email: <support@kcs.example.com>
- Emergency: +1-555-KCS-HELP

## Appendix

### Sample Scripts

All deployment scripts are available in the `tools/deployment/` directory:

- `deploy.sh`: Automated deployment script
- `health-check.sh`: Health monitoring script
- `backup.sh`: Backup automation
- `maintenance.sh`: Regular maintenance tasks

### Configuration Templates

Configuration templates for various environments are in `config/templates/`:

- `development.yaml`: Development environment
- `staging.yaml`: Staging environment  
- `production.yaml`: Production environment
- `kubernetes.yaml`: Kubernetes deployment

### Security Checklist

- [ ] SSL/TLS configured with strong ciphers
- [ ] Authentication tokens rotated regularly
- [ ] Database access restricted by IP
- [ ] Firewall rules configured
- [ ] Security headers enabled
- [ ] Rate limiting implemented
- [ ] Logs monitored for security events
- [ ] Backup encryption enabled
- [ ] Access logs retained for audit
- [ ] Vulnerability scanning scheduled
