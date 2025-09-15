# API Async Optimization Guide

## Overview

This guide covers implementing async optimizations for KCS API endpoints.

## Key Changes Required

### 1. FastAPI Async Endpoints

```python
# Before
@app.post("/mcp/tools/search_code")
def search_code(request: SearchRequest):
    return search_service.search(request.query)

# After  
@app.post("/mcp/tools/search_code")
async def search_code(request: SearchRequest):
    return await search_service.search(request.query)
```text

### 2. Async Database Operations

```python
# Use asyncpg for async PostgreSQL
import asyncpg

async def get_symbol(symbol_name: str):
    async with asyncpg.connect(DATABASE_URL) as conn:
        result = await conn.fetchrow(
            "SELECT * FROM symbols WHERE name = $1", 
            symbol_name
        )
        return result
```text

### 3. Async HTTP Client

```python
# Use aiohttp for external API calls
import aiohttp

async def fetch_external_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```text

## Implementation Priority

1. **High**: Convert database operations to async
2. **Medium**: Convert external API calls to async  
3. **Low**: Add async middleware for logging/metrics

## Testing

Use pytest-asyncio for testing async endpoints:

```bash
pip install pytest-asyncio
pytest -v tests/test_async_api.py
```text

## Performance Benefits

- 2-5x improvement in concurrent request handling
- Better resource utilization
- Reduced memory footprint under load
