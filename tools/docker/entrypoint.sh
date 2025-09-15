#!/bin/bash
set -e

# KCS MCP Server Docker Entrypoint
# This script starts the KCS MCP server with proper configuration

# Default values
KCS_HOST=${KCS_HOST:-"0.0.0.0"}
KCS_PORT=${KCS_PORT:-"8080"}
KCS_LOG_LEVEL=${KCS_LOG_LEVEL:-"info"}

# Database configuration
if [ -z "$DATABASE_URL" ]; then
    echo "Warning: DATABASE_URL not set. Server may fail to start."
fi

# Redis configuration (optional)
if [ -z "$REDIS_URL" ]; then
    echo "Info: REDIS_URL not set. Running without Redis caching."
fi

# Authentication configuration
if [ -z "$JWT_SECRET" ]; then
    echo "Warning: JWT_SECRET not set. API will use insecure default token."
elif [ "$JWT_SECRET" = "dev_jwt_secret_change_in_production" ]; then
    echo "Warning: JWT_SECRET is using default value. Change for production use."
fi

# Log startup info
echo "Starting KCS MCP Server..."
echo "Host: $KCS_HOST"
echo "Port: $KCS_PORT"
echo "Log Level: $KCS_LOG_LEVEL"

# Execute the command passed to the container
exec "$@"
