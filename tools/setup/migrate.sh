#!/bin/bash
# KCS Database Migration Runner
# Applies SQL migrations in order and tracks schema version

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MIGRATIONS_DIR="$PROJECT_ROOT/src/sql/migrations"
PGPASSWORD="${POSTGRES_PASSWORD:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATABASE_URL="${DATABASE_URL:-postgresql://kcs:kcs_dev_password@localhost:5432/kcs}"
DRY_RUN=false
VERBOSE=false
FORCE=false

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Apply KCS database migrations in order.

OPTIONS:
    -h, --help              Show this help message
    -n, --dry-run          Show what would be executed without running
    -v, --verbose          Verbose output
    -f, --force            Force migration even if schema version is ahead
    -u, --url URL          Database URL (default: \$DATABASE_URL)
    --reset                Drop all tables and reapply all migrations (DANGEROUS)
    --status               Show current migration status
    --rollback VERSION     Rollback to specific migration version (not implemented)

EXAMPLES:
    $0                     Apply all pending migrations
    $0 --dry-run          Show pending migrations without applying
    $0 --status           Show current schema version
    $0 --reset            Reset database and apply all migrations

ENVIRONMENT:
    DATABASE_URL          PostgreSQL connection string
    POSTGRES_PASSWORD     Password for PostgreSQL (if not in URL)
EOF
}

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

error() {
    echo -e "${RED}ERROR:${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}WARNING:${NC} $*" >&2
}

success() {
    echo -e "${GREEN}SUCCESS:${NC} $*" >&2
}

verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}VERBOSE:${NC} $*" >&2
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -u|--url)
            DATABASE_URL="$2"
            shift 2
            ;;
        --reset)
            RESET=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --rollback)
            ROLLBACK_VERSION="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate database URL
if [[ -z "$DATABASE_URL" ]]; then
    error "DATABASE_URL is required"
    exit 1
fi

# Extract connection parameters
DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
DB_NAME=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')
DB_USER=$(echo "$DATABASE_URL" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
DB_PASS=$(echo "$DATABASE_URL" | sed -n 's/.*\/\/[^:]*:\([^@]*\)@.*/\1/p')

# Export for psql
export PGHOST="${DB_HOST:-localhost}"
export PGPORT="${DB_PORT:-5432}"
export PGDATABASE="${DB_NAME:-kcs}"
export PGUSER="${DB_USER:-kcs}"
export PGPASSWORD="${DB_PASS:-$PGPASSWORD}"

verbose "Database connection: $PGUSER@$PGHOST:$PGPORT/$PGDATABASE"

# Test database connection
test_connection() {
    log "Testing database connection..."
    if ! psql -c "SELECT 1;" > /dev/null 2>&1; then
        error "Cannot connect to database"
        error "URL: $DATABASE_URL"
        error "Check that PostgreSQL is running and connection parameters are correct"
        exit 1
    fi
    verbose "Database connection successful"
}

# Create schema_migrations table if it doesn't exist
ensure_migrations_table() {
    log "Ensuring schema_migrations table exists..."
    psql -q -c "
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            filename TEXT NOT NULL,
            checksum TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at
        ON schema_migrations(applied_at);
    "
    verbose "Schema migrations table ready"
}

# Get current schema version
get_current_version() {
    psql -t -c "SELECT COALESCE(MAX(version), '000') FROM schema_migrations;" | xargs
}

# Get list of applied migrations
get_applied_migrations() {
    psql -t -c "SELECT version FROM schema_migrations ORDER BY version;" | xargs
}

# Calculate checksum of migration file
calculate_checksum() {
    local file="$1"
    if command -v sha256sum > /dev/null; then
        sha256sum "$file" | cut -d' ' -f1
    elif command -v shasum > /dev/null; then
        shasum -a 256 "$file" | cut -d' ' -f1
    else
        error "No checksum tool available (sha256sum or shasum)"
        exit 1
    fi
}

# Show migration status
show_status() {
    log "Current migration status:"

    local current_version
    current_version=$(get_current_version)
    echo "Current schema version: $current_version"

    echo "Applied migrations:"
    psql -c "
        SELECT
            version,
            filename,
            applied_at,
            LEFT(checksum, 8) || '...' as checksum_preview
        FROM schema_migrations
        ORDER BY version;
    "

    echo "Available migrations:"
    for file in "$MIGRATIONS_DIR"/*.sql; do
        if [[ -f "$file" ]]; then
            local filename basename version checksum
            filename=$(basename "$file")
            basename="${filename%.sql}"
            version="${basename:0:3}"
            checksum=$(calculate_checksum "$file")

            if [[ $(psql -t -c "SELECT 1 FROM schema_migrations WHERE version = '$version';" | xargs) == "1" ]]; then
                echo "  ✓ $filename (applied)"
            else
                echo "  - $filename (pending)"
            fi
        fi
    done
}

# Apply a single migration
apply_migration() {
    local file="$1"
    local filename basename version checksum

    filename=$(basename "$file")
    basename="${filename%.sql}"
    version="${basename:0:3}"
    checksum=$(calculate_checksum "$file")

    log "Applying migration $version: $filename"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: Would apply $filename"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "--- Migration content ---"
            cat "$file"
            echo "--- End migration content ---"
        fi
        return 0
    fi

    # Check if already applied
    if [[ $(psql -t -c "SELECT 1 FROM schema_migrations WHERE version = '$version';" | xargs) == "1" ]]; then
        # Verify checksum matches
        local existing_checksum
        existing_checksum=$(psql -t -c "SELECT checksum FROM schema_migrations WHERE version = '$version';" | xargs)

        if [[ "$existing_checksum" != "$checksum" ]]; then
            if [[ "$FORCE" == "true" ]]; then
                warn "Migration $version checksum mismatch, forcing re-application"
                psql -q -c "DELETE FROM schema_migrations WHERE version = '$version';"
            else
                error "Migration $version already applied with different checksum"
                error "Existing: $existing_checksum"
                error "Current:  $checksum"
                error "Use --force to override (dangerous)"
                exit 1
            fi
        else
            verbose "Migration $version already applied (checksum matches)"
            return 0
        fi
    fi

    # Apply migration in transaction
    verbose "Executing migration $filename..."
    if psql -v ON_ERROR_STOP=1 -f "$file"; then
        # Record successful application
        psql -q -c "
            INSERT INTO schema_migrations (version, filename, checksum)
            VALUES ('$version', '$filename', '$checksum');
        "
        success "Applied migration $version successfully"
    else
        error "Failed to apply migration $version"
        exit 1
    fi
}

# Reset database (drop all tables)
reset_database() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: Would reset database (drop all tables)"
        return 0
    fi

    warn "This will DROP ALL TABLES in the database!"
    read -p "Are you sure? Type 'yes' to continue: " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Aborted."
        exit 0
    fi

    log "Resetting database..."
    psql -c "
        DROP SCHEMA public CASCADE;
        CREATE SCHEMA public;
        GRANT ALL ON SCHEMA public TO $PGUSER;
        GRANT ALL ON SCHEMA public TO public;
    "
    success "Database reset complete"
}

# Main execution
main() {
    test_connection

    if [[ "${RESET:-false}" == "true" ]]; then
        reset_database
    fi

    ensure_migrations_table

    if [[ "${STATUS_ONLY:-false}" == "true" ]]; then
        show_status
        exit 0
    fi

    if [[ -n "${ROLLBACK_VERSION:-}" ]]; then
        error "Rollback not implemented yet"
        exit 1
    fi

    # Find all migration files
    local migration_files=()
    while IFS= read -r -d '' file; do
        migration_files+=("$file")
    done < <(find "$MIGRATIONS_DIR" -name "*.sql" -print0 | sort -z)

    if [[ ${#migration_files[@]} -eq 0 ]]; then
        warn "No migration files found in $MIGRATIONS_DIR"
        exit 0
    fi

    local current_version pending_count=0
    current_version=$(get_current_version)

    log "Current schema version: $current_version"

    # Apply pending migrations
    for file in "${migration_files[@]}"; do
        local filename basename version
        filename=$(basename "$file")
        basename="${filename%.sql}"
        version="${basename:0:3}"

        # Skip if version format is invalid
        if [[ ! "$version" =~ ^[0-9]{3}$ ]]; then
            warn "Skipping file with invalid version format: $filename"
            continue
        fi

        # Skip if already applied (unless force)
        if [[ $(psql -t -c "SELECT 1 FROM schema_migrations WHERE version = '$version';" | xargs) == "1" ]]; then
            if [[ "$FORCE" == "false" ]]; then
                verbose "Skipping already applied migration: $filename"
                continue
            fi
        fi

        apply_migration "$file"
        ((pending_count++))
    done

    if [[ $pending_count -eq 0 ]]; then
        success "No pending migrations found - database is up to date"
    else
        success "Applied $pending_count migration(s) successfully"
    fi

    # Show final status
    if [[ "$VERBOSE" == "true" ]]; then
        echo
        show_status
    fi
}

# Run main function
main "$@"
