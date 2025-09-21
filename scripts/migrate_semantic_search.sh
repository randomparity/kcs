#!/bin/bash
#
# Semantic Search Database Migration Script
#
# This script applies the semantic search database migrations to set up
# the required tables and indexes for semantic search functionality.
#
# Usage: ./migrate_semantic_search.sh [database_url]
#
# If database_url is not provided, it will try to use environment variables.

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MIGRATIONS_DIR="$PROJECT_ROOT/src/sql/migrations"

# Default database connection
DEFAULT_DB_URL="postgresql://kcs:kcs_dev_password@localhost:5432/kcs"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Help function
show_help() {
    echo "Semantic Search Database Migration Script"
    echo ""
    echo "Usage: $0 [OPTIONS] [DATABASE_URL]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help         Show this help message"
    echo "  -c, --check        Check if migrations are needed"
    echo "  -f, --force        Force re-apply migrations (USE WITH CAUTION)"
    echo "  --dry-run          Show what would be done without executing"
    echo ""
    echo "DATABASE_URL:"
    echo "  PostgreSQL connection string (optional)"
    echo "  Format: postgresql://user:password@host:port/database"
    echo ""
    echo "Environment Variables:"
    echo "  DATABASE_URL       - Database connection string"
    echo "  POSTGRES_HOST      - Database host (default: localhost)"
    echo "  POSTGRES_PORT      - Database port (default: 5432)"
    echo "  POSTGRES_DB        - Database name (default: kcs)"
    echo "  POSTGRES_USER      - Database user (default: kcs)"
    echo "  POSTGRES_PASSWORD  - Database password"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 postgresql://kcs:password@localhost:5432/kcs"
    echo "  $0 --check"
    echo "  $0 --dry-run"
}

# Get database URL from environment or arguments
get_database_url() {
    local db_url="$1"

    if [[ -n "$db_url" ]]; then
        echo "$db_url"
        return
    fi

    # Try environment variables
    if [[ -n "${DATABASE_URL:-}" ]]; then
        echo "$DATABASE_URL"
        return
    fi

    # Build from individual components
    local host="${POSTGRES_HOST:-localhost}"
    local port="${POSTGRES_PORT:-5432}"
    local database="${POSTGRES_DB:-kcs}"
    local user="${POSTGRES_USER:-kcs}"
    local password="${POSTGRES_PASSWORD:-}"

    if [[ -n "$password" ]]; then
        echo "postgresql://$user:$password@$host:$port/$database"
    else
        log_warning "No database password found, using default"
        echo "$DEFAULT_DB_URL"
    fi
}

# Check if psql is available
check_psql() {
    if ! command -v psql &> /dev/null; then
        log_error "psql command not found. Please install PostgreSQL client tools."
        exit 1
    fi
}

# Test database connection
test_connection() {
    local db_url="$1"

    log_info "Testing database connection..."

    if psql "$db_url" -c "SELECT 1;" &> /dev/null; then
        log_success "Database connection successful"
        return 0
    else
        log_error "Failed to connect to database"
        log_error "URL: $db_url"
        return 1
    fi
}

# Check if pgvector extension is available
check_pgvector() {
    local db_url="$1"

    log_info "Checking for pgvector extension..."

    local has_vector
    has_vector=$(psql "$db_url" -t -c "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector');" 2>/dev/null || echo "f")

    if [[ "$has_vector" =~ "t" ]]; then
        log_success "pgvector extension is available"
        return 0
    else
        log_error "pgvector extension is not available"
        log_error "Please install pgvector: https://github.com/pgvector/pgvector"
        return 1
    fi
}

# Check if migration is already applied
check_migration_status() {
    local db_url="$1"
    local migration_name="$2"

    # Check if the main table exists
    local table_exists
    table_exists=$(psql "$db_url" -t -c "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'indexed_content');" 2>/dev/null || echo "f")

    if [[ "$table_exists" =~ "t" ]]; then
        log_info "Semantic search tables already exist"
        return 0
    else
        log_info "Semantic search tables not found, migration needed"
        return 1
    fi
}

# Apply migration
apply_migration() {
    local db_url="$1"
    local migration_file="$2"
    local dry_run="$3"

    if [[ ! -f "$migration_file" ]]; then
        log_error "Migration file not found: $migration_file"
        return 1
    fi

    log_info "Applying migration: $(basename "$migration_file")"

    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN: Would execute SQL from $migration_file"
        echo "--- Migration content preview ---"
        head -20 "$migration_file"
        echo "--- (truncated) ---"
        return 0
    fi

    if psql "$db_url" -f "$migration_file"; then
        log_success "Migration applied successfully"
        return 0
    else
        log_error "Migration failed"
        return 1
    fi
}

# Main function
main() {
    local db_url=""
    local check_only=false
    local force=false
    local dry_run=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                db_url="$1"
                shift
                ;;
        esac
    done

    # Get database URL
    db_url=$(get_database_url "$db_url")

    log_info "Starting semantic search migration"
    log_info "Database URL: ${db_url/\/\/[^@]*@/\/\/***:***@}"  # Hide credentials

    # Check prerequisites
    check_psql

    if ! test_connection "$db_url"; then
        exit 1
    fi

    if ! check_pgvector "$db_url"; then
        exit 1
    fi

    # Check migration status
    if check_migration_status "$db_url" "014_semantic_search_core"; then
        if [[ "$check_only" == "true" ]]; then
            log_success "Migration check complete: Tables exist"
            exit 0
        elif [[ "$force" == "false" ]]; then
            log_info "Semantic search tables already exist. Use --force to re-apply."
            exit 0
        else
            log_warning "Force flag specified, will re-apply migration"
        fi
    else
        if [[ "$check_only" == "true" ]]; then
            log_info "Migration check complete: Migration needed"
            exit 1
        fi
    fi

    # Apply migration
    local migration_file="$MIGRATIONS_DIR/014_semantic_search_core.sql"

    if apply_migration "$db_url" "$migration_file" "$dry_run"; then
        log_success "Semantic search migration completed successfully"

        if [[ "$dry_run" == "false" ]]; then
            log_info "Verifying migration..."
            if check_migration_status "$db_url" "014_semantic_search_core"; then
                log_success "Migration verification successful"
            else
                log_error "Migration verification failed"
                exit 1
            fi
        fi
    else
        log_error "Migration failed"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
