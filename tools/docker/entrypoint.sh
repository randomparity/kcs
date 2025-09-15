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

# Authentication token
if [ -z "$KCS_AUTH_TOKEN" ]; then
    echo "Warning: KCS_AUTH_TOKEN not set. API will be unauthenticated."
fi

# Log startup info
echo "Starting KCS MCP Server..."
echo "Host: $KCS_HOST"
echo "Port: $KCS_PORT"
echo "Log Level: $KCS_LOG_LEVEL"

# Execute the command passed to the container
exec "$@"
