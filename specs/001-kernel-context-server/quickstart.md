# KCS Quickstart Guide

**Version**: 1.0.0
**Prerequisites**: Linux x86_64, Docker, 8GB RAM, 20GB disk

## Prerequisites

### Kernel Source Setup

KCS analyzes existing Linux kernel repositories. Set up your kernel source:

```bash
# Option 1: Use existing kernel source
export KCS_KERNEL_PATH="$HOME/src/linux"

# Option 2: Clone a fresh kernel (if needed)
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git ~/src/linux
export KCS_KERNEL_PATH="$HOME/src/linux"

# Option 3: Use stable kernel
git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git ~/src/linux-stable
export KCS_KERNEL_PATH="$HOME/src/linux-stable"

# Make environment variable persistent
echo 'export KCS_KERNEL_PATH="$HOME/src/linux"' >> ~/.bashrc
```

> **Note**: KCS does **not** include kernel source. It's designed to work with your existing
> kernel development setup or any kernel repository you want to analyze.

## 1-Minute Setup

```bash
# Clone and setup KCS
git clone https://github.com/your-org/kcs.git
cd kcs
./tools/setup/install.sh

# Start services
docker-compose up -d

# Index your kernel repository
tools/index_kernel.sh "$KCS_KERNEL_PATH"

# Start MCP server
kcs-mcp --serve --port 8080 --auth-token dev-token
```

## Verification Steps

### 1. Check Service Health

```bash
curl http://localhost:8080/health
# Expected: {"status": "healthy", "version": "1.0.0", ...}
```text

### 2. Test Symbol Query

```bash
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "sys_read"}' \
  http://localhost:8080/mcp/tools/get_symbol

# Expected: Symbol info with file/line citations
```text

### 3. Test Impact Analysis

```bash
# Create a sample diff
cat > test.diff << 'EOF'
diff --git a/fs/read_write.c b/fs/read_write.c
index abc123..def456 100644
--- a/fs/read_write.c
+++ b/fs/read_write.c
@@ -451,7 +451,7 @@ ssize_t vfs_read(struct file *file, char __user *buf,
+size_t count, loff_t *pos
-       if (!ret)
+       if (!ret && count > 0)
                ret = __vfs_read(file, buf, count, pos);
EOF

# Analyze impact
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d "{\"diff\": \"$(cat test.diff | jq -Rs .)\"}" \
  http://localhost:8080/mcp/tools/impact_of

# Expected: List of affected configs, modules, tests, owners
```text

### 4. Test Entry Point Flow

```bash
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"entry": "__NR_read"}' \
  http://localhost:8080/mcp/tools/entrypoint_flow

# Expected: Step-by-step flow from syscall to implementation
```text

## Common Use Cases

### Find Who Calls a Function

```bash
# Who calls vfs_read?
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "vfs_read", "depth": 2}' \
  http://localhost:8080/mcp/tools/who_calls
```text

### Search for Code Patterns

```bash
# Find memory barrier usage
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "smp_mb memory barrier", "topK": 5}' \
  http://localhost:8080/mcp/tools/search_code
```text

### Check for Drift

```bash
# Check if implementation matches spec
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"feature_id": "vfs-read-interface"}' \
  http://localhost:8080/mcp/tools/diff_spec_vs_code
```text

### Find Maintainers

```bash
# Who maintains VFS code?
curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"paths": ["fs/", "include/linux/fs.h"]}' \
  http://localhost:8080/mcp/tools/owners_for
```text

## CI Integration

### GitHub Actions

```yaml
name: KCS Impact Analysis
on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Get diff
        run: |
          git diff origin/main...HEAD > pr.diff

      - name: Analyze impact
        run: |
          curl -H "Authorization: Bearer ${{ secrets.KCS_TOKEN }}" \
            -H "Content-Type: application/json" \
            -d "{\"diff\": \"$(cat pr.diff | jq -Rs .)\"}" \
            https://kcs.your-org.com/mcp/tools/impact_of > impact.json

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            const impact = require('./impact.json');
            const comment = `
            ## KCS Impact Analysis

            **Affected Configurations**: ${impact.configs.join(', ')}
            **Affected Modules**: ${impact.modules.join(', ')}
            **Tests to Run**: ${impact.tests.length} tests
            **Risk Factors**: ${impact.risks.join(', ') || 'None detected'}

            cc: ${impact.owners.join(', ')}
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```text

## IDE Integration

### VS Code with Claude

1. Install Claude Code extension
2. Add to settings.json:

```json
{
  "claude.mcp.servers": {
    "kcs": {
      "url": "http://localhost:8080",
      "token": "dev-token"
    }
  }
}
```text

3. Use in chat:

```text
@kcs What functions call vfs_read in the kernel?
@kcs What's the impact if I change copy_page_to_iter?
```text

## Performance Tuning

### Indexing Optimization

```bash
# Parallel indexing with 8 workers
kcs-parser --parse ~/linux --config x86_64:defconfig --workers 8

# Incremental update
kcs-parser --parse ~/linux --incremental --since yesterday
```text

### Query Optimization

```bash
# Warm cache for common queries
kcs-graph --warmup --top-symbols 1000

# Monitor slow queries
kcs-mcp --serve --slow-query-log /var/log/kcs/slow.log
```text

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs kcs-mcp
docker-compose logs postgres

# Verify database connection
psql postgresql://localhost/kcs -c "SELECT COUNT(*) FROM symbol;"
```text

### Slow Queries

```bash
# Check index health
kcs-graph --analyze --db postgresql://localhost/kcs

# Rebuild indexes if needed
kcs-graph --reindex --db postgresql://localhost/kcs
```text

### Missing Symbols

```bash
# Verify configuration was indexed
kcs-parser --list-configs --db postgresql://localhost/kcs

# Re-index specific configuration
kcs-parser --parse ~/linux --config x86_64:allmodconfig --force
```text

## Architecture Overview

```text
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  AI Agent   │────▶│  MCP Server  │────▶│  PostgreSQL  │
│(Claude/etc) │     │   (Python)   │     │  + pgvector  │
└─────────────┘     └──────────────┘     └──────────────┘
                            │                     ▲
                            ▼                     │
                    ┌──────────────┐     ┌──────────────┐
                    │   Extractor  │────▶│    Parser    │
                    │    (Rust)    │     │    (Rust)    │
                    └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────┐
                                         │ Linux Kernel │
                                         │  Repository  │
                                         └──────────────┘
```text

## Development & Testing

### Using Mini-Kernel Fixture

For development and CI, KCS includes a mini-kernel fixture:

```bash
# Fast testing with mini-kernel (< 1 second)
export KCS_KERNEL_PATH="./tests/fixtures/mini-kernel-v6.1"
make test-mini-kernel

# Test indexing with mini-kernel
tools/index_kernel.sh "$KCS_KERNEL_PATH" --dry-run

# CI-friendly testing
make test-mini-kernel-fast
```

The mini-kernel fixture:

- ✅ **Size**: <100KB (vs 5GB+ full kernel)
- ✅ **Speed**: <1s parsing (vs 20min full kernel)
- ✅ **Coverage**: Representative kernel patterns
- ✅ **CI-Ready**: No external dependencies

### Full Kernel Development

```bash
# Use your real kernel for development
export KCS_KERNEL_PATH="$HOME/src/linux"
tools/index_kernel.sh "$KCS_KERNEL_PATH"
```

## Next Steps

1. **Start with mini-kernel**: Test KCS with the included fixture
2. **Index your kernel**: Full indexing takes ~20 minutes
3. **Configure authentication**: Replace dev-token with JWT
4. **Set up monitoring**: Enable Prometheus metrics
5. **Join community**: <https://github.com/your-org/kcs/discussions>

## Support

- Documentation: <https://docs.kcs.dev>
- Issues: <https://github.com/your-org/kcs/issues>
- Discord: <https://discord.gg/kcs>

---

*For production deployment, see [deployment guide](./docs/deployment.md)*
