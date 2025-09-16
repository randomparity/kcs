# Kernel Context Server MCP API

Model Context Protocol API for Linux kernel analysis

**Version:** 1.0.0

## Overview

The Kernel Context Server (KCS) provides a Model Context Protocol (MCP) API for
analyzing Linux kernel source code. This API enables AI coding assistants to
understand kernel internals, trace function calls, analyze impact of changes,
and provide accurate kernel development guidance.

## Key Features

- **Read-Only Access**: KCS never modifies kernel source code
- **Constitutional Citations**: All results include file:line references  
- **Configuration Awareness**: Results tagged with kernel configuration
- **Performance Optimized**: Queries complete in <600ms (p95)
- **MCP Protocol**: Native integration with AI coding assistants

## Base URLs

- **Local development server**: `http://localhost:8080`

## Quick Start

1. **Authentication**: Obtain an auth token
2. **Health Check**: Verify server status at `/health`
3. **Search Code**: Use `/mcp/tools/search_code` for semantic search
4. **Explore Symbols**: Get symbol info with `/mcp/tools/get_symbol`
5. **Trace Calls**: Find callers with `/mcp/tools/who_calls`

## Constitutional Requirements

KCS operates under strict constitutional requirements:

1. **Citations Required**: Every claim must include file:line references
2. **Read-Only**: Never modifies kernel source code
3. **Performance**: Index ≤20min, queries p95 ≤600ms
4. **Config-Aware**: Results tagged with kernel configuration

See the [Integration Guide](integration.md) for complete setup instructions.
