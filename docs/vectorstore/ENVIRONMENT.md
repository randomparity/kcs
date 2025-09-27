# VectorStore Environment Variables Configuration

This document describes the environment variables required for the VectorStore semantic search system.

## Required Variables

These environment variables **MUST** be set for the system to function:

### `POSTGRES_USER`
- **Description**: PostgreSQL database username
- **Required**: Yes
- **Example**: `kcs`
- **Error if missing**: `OSError: Missing required environment variables: POSTGRES_USER: Database username is required`

### `POSTGRES_PASSWORD`
- **Description**: PostgreSQL database password
- **Required**: Yes
- **Example**: `kcs_dev_password_change_in_production`
- **Security Note**: Use a strong password in production environments
- **Error if missing**: `OSError: Missing required environment variables: POSTGRES_PASSWORD: Database password is required`

## Optional Variables

These variables have sensible defaults but can be customized:

### `POSTGRES_HOST`
- **Description**: PostgreSQL server hostname or IP address
- **Required**: No
- **Default**: `localhost`
- **Example**: `db.example.com`, `192.168.1.10`

### `POSTGRES_PORT`
- **Description**: PostgreSQL server port
- **Required**: No
- **Default**: `5432`
- **Example**: `5433`, `6543`

### `POSTGRES_DB`
- **Description**: PostgreSQL database name
- **Required**: No
- **Default**: `kcs`
- **Example**: `kcs_production`, `semantic_search`

## Configuration Methods

### Method 1: Using a `.env` File (Recommended for Development)

Create a `.env` file in the project root:

```bash
# Required variables
POSTGRES_USER=kcs
POSTGRES_PASSWORD=your_secure_password_here

# Optional variables (uncomment to override defaults)
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=kcs
```

The system automatically loads the `.env` file using `python-dotenv`.

### Method 2: Environment Variables (Recommended for Production)

Set environment variables in your shell or deployment configuration:

```bash
# Bash/Linux
export POSTGRES_USER=kcs
export POSTGRES_PASSWORD=your_secure_password_here
export POSTGRES_HOST=db.production.com
export POSTGRES_PORT=5432
export POSTGRES_DB=kcs_production

# Or as a one-liner
POSTGRES_USER=kcs POSTGRES_PASSWORD=secret python your_app.py
```

### Method 3: Docker/Docker Compose

In `docker-compose.yml`:

```yaml
services:
  app:
    environment:
      - POSTGRES_USER=kcs
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}  # From .env or shell
      - POSTGRES_HOST=postgres  # Docker service name
      - POSTGRES_PORT=5432
      - POSTGRES_DB=kcs
    env_file:
      - .env  # Or use an env file
```

### Method 4: Kubernetes ConfigMap/Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-credentials
type: Opaque
data:
  POSTGRES_USER: a2Nz  # base64 encoded
  POSTGRES_PASSWORD: <base64-encoded-password>

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
data:
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "kcs"
```

## Verification

### Check Environment Variables

```python
import os
from src.python.semantic_search.database.connection import verify_environment

# Check if all required variables are set
try:
    verify_environment()
    print("✅ All required environment variables are set")
except OSError as e:
    print(f"❌ {e}")
```

### Test Database Connection

```python
import asyncio
from src.python.semantic_search.database.connection import (
    DatabaseConfig,
    DatabaseConnection
)

async def test_connection():
    try:
        config = DatabaseConfig.from_env()
        db = DatabaseConnection(config)
        await db.connect()
        print(f"✅ Connected to {config.host}:{config.port}/{config.database}")
        await db.disconnect()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test_connection())
```

## Security Best Practices

1. **Never commit `.env` files** containing real passwords to version control
2. **Use strong passwords** in production (min 16 characters, mixed case, numbers, symbols)
3. **Rotate credentials regularly** in production environments
4. **Use secret management tools** like:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Kubernetes Secrets
   - Azure Key Vault
5. **Restrict database user permissions** to only what's needed:
   ```sql
   -- Grant only necessary permissions
   GRANT CONNECT ON DATABASE kcs TO kcs_user;
   GRANT USAGE ON SCHEMA public TO kcs_user;
   GRANT CREATE, SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO kcs_user;
   ```

## Troubleshooting

### Error: "Missing required environment variables"
- **Cause**: `POSTGRES_USER` or `POSTGRES_PASSWORD` not set
- **Solution**: Ensure both variables are set in your environment or `.env` file

### Error: "Failed to connect to database"
- **Cause**: Database is not running or connection parameters are incorrect
- **Solutions**:
  1. Check PostgreSQL is running: `pg_isready -h localhost -p 5432`
  2. Verify connection parameters are correct
  3. Check network connectivity to database host
  4. Ensure PostgreSQL is configured to accept connections

### Error: "pgvector extension not found"
- **Cause**: pgvector extension not installed in PostgreSQL
- **Solution**: Install pgvector:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

## Example Configurations

### Local Development
```bash
# .env file
POSTGRES_USER=dev_user
POSTGRES_PASSWORD=dev_password
# Using defaults for host/port/db
```

### Testing/CI
```bash
# .env.test
POSTGRES_USER=test_user
POSTGRES_PASSWORD=test_password
POSTGRES_DB=kcs_test
```

### Production
```bash
# Set via deployment platform (don't use .env files)
POSTGRES_USER=kcs_prod_user
POSTGRES_PASSWORD=<strong-random-password>
POSTGRES_HOST=prod-db-cluster.aws.com
POSTGRES_PORT=5432
POSTGRES_DB=kcs_production
```

## Related Documentation

- [Database Connection Setup](./setup.md#database-setup)
- [Migration Guide](./migration.md)
- [API Reference](./api.html)
- [Performance Tuning](./performance.md)