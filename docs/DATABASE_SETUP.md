# KCS Database Setup Guide

This guide covers how to set up KCS with different database configurations and user setups.

## Default Configuration (Docker Compose)

The default setup uses:

- **Database**: `kcs`
- **User**: `kcs`
- **Password**: `kcs_dev_password_change_in_production`

```bash
# Use the default setup
make docker-compose-up-app
```

## Custom Database Configuration

### Option 1: Environment Variables (Recommended)

Modify your `.env` file:

```bash
# Database settings
POSTGRES_DB=my_database
POSTGRES_USER=my_user
POSTGRES_PASSWORD=my_secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# This will be auto-constructed
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
```

### Option 2: External Database

If using an existing PostgreSQL database:

1. **Set connection in .env:**

   ```bash
   DATABASE_URL=postgresql://existing_user:password@db.company.com:5432/existing_db
   ```

2. **Run migrations manually:**

   ```bash
   # Export your database URL
   export DATABASE_URL="postgresql://existing_user:password@db.company.com:5432/existing_db"

   # Run migrations
   psql $DATABASE_URL -f src/sql/migrations/001_initial_schema.sql
   psql $DATABASE_URL -f src/sql/migrations/002_graph_tables.sql
   # ... continue for all migrations
   ```

3. **Handle permissions manually:**

   ```sql
   -- Connect to your database and run:
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO existing_user;
   GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO existing_user;

   -- For semantic search tables specifically:
   GRANT SELECT, INSERT, UPDATE, DELETE ON indexed_content TO existing_user;
   GRANT SELECT, INSERT, UPDATE, DELETE ON vector_embedding TO existing_user;
   GRANT SELECT, INSERT, UPDATE, DELETE ON search_query TO existing_user;
   GRANT SELECT, INSERT, UPDATE, DELETE ON search_result TO existing_user;
   ```

## Migration Behavior

### Automatic Permission Grants

The migration `014_semantic_search_core.sql` automatically:

1. **Checks if `kcs` user exists**
2. **Grants permissions if found**
3. **Skips with helpful message if not found**

Example output:

```
NOTICE:  User "kcs" does not exist - skipping permission grants
NOTICE:  Please manually grant permissions to your database user:
NOTICE:  GRANT SELECT, INSERT, UPDATE, DELETE ON indexed_content, vector_embedding, search_query, search_result TO <your_user>;
```

### Manual Permission Setup

For non-`kcs` users, run these grants after migrations:

```sql
-- Replace 'your_user' with your actual database user
GRANT SELECT, INSERT, UPDATE, DELETE ON indexed_content TO your_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON vector_embedding TO your_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON search_query TO your_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON search_result TO your_user;
GRANT USAGE ON SEQUENCE indexed_content_id_seq TO your_user;
GRANT USAGE ON SEQUENCE vector_embedding_id_seq TO your_user;
GRANT USAGE ON SEQUENCE search_query_id_seq TO your_user;
GRANT USAGE ON SEQUENCE search_result_id_seq TO your_user;

-- Grant on existing KCS tables too
GRANT SELECT, INSERT, UPDATE, DELETE ON file TO your_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON symbol TO your_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON symbol_embedding TO your_user;
-- ... (continue for other tables as needed)
```

## Production Considerations

### Security Best Practices

1. **Change default passwords:**

   ```bash
   POSTGRES_PASSWORD=your_very_secure_random_password_here
   JWT_SECRET=your_64_character_random_jwt_secret_here
   ```

2. **Use dedicated database users:**

   ```sql
   -- Create application-specific user
   CREATE USER kcs_app WITH PASSWORD 'secure_app_password';

   -- Grant only necessary permissions
   GRANT CONNECT ON DATABASE kcs TO kcs_app;
   GRANT USAGE ON SCHEMA public TO kcs_app;
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO kcs_app;
   GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO kcs_app;
   ```

3. **Network security:**

   ```bash
   # Only bind to localhost in production
   POSTGRES_HOST=127.0.0.1

   # Use non-default ports
   POSTGRES_EXTERNAL_PORT=5433
   ```

### High Availability Setup

For production deployments:

1. **External managed database** (AWS RDS, Google Cloud SQL, etc.)
2. **Connection pooling** (PgBouncer, built-in pooling)
3. **Read replicas** for search-heavy workloads
4. **Backup strategies** (automated backups, point-in-time recovery)

## Troubleshooting

### Permission Errors

**Error:** `permission denied for table indexed_content`

**Solution:**

```sql
-- Check current user
SELECT current_user;

-- Grant missing permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON indexed_content TO current_user;
```

### Connection Errors

**Error:** `could not connect to server: Connection refused`

**Solutions:**

1. Check PostgreSQL is running: `pg_isready -h localhost -p 5432`
2. Verify connection string in `.env`
3. Check firewall/network settings
4. Verify Docker containers: `docker compose ps`

### Migration Failures

**Error:** `role "some_user" does not exist`

**Solution:**

1. Create the user first:

   ```sql
   CREATE USER some_user WITH PASSWORD 'password';
   ```

2. Or modify migration to use existing user

### Schema Conflicts

**Error:** `relation "indexed_content" already exists`

**Solution:**

```bash
# Check existing schema
psql $DATABASE_URL -c "\dt"

# If needed, drop and recreate (WARNING: destroys data)
psql $DATABASE_URL -c "DROP TABLE IF EXISTS indexed_content CASCADE;"

# Or use migration rollback if available
```

## Testing Database Setup

Verify your database setup:

```bash
# Test connection
psql $DATABASE_URL -c "SELECT current_user, current_database();"

# Check tables exist
psql $DATABASE_URL -c "\dt"

# Test permissions
psql $DATABASE_URL -c "SELECT COUNT(*) FROM indexed_content;"

# Check KCS server connectivity
curl http://localhost:8080/health
```

## Migration Development

When creating new migrations:

1. **Avoid hard-coded usernames**
2. **Use conditional grants** with user existence checks
3. **Document permission requirements**
4. **Test with different user configurations**

Example template:

```sql
-- Check if specific user exists before granting
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'target_user') THEN
        GRANT permissions TO target_user;
        RAISE NOTICE 'Permissions granted to: target_user';
    ELSE
        RAISE NOTICE 'User target_user not found - manual grants required';
    END IF;
END
$$;
```
