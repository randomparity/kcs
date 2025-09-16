# Usage Examples

## Overview

This guide provides practical examples for common KCS API usage patterns.

## Authentication Setup

```bash
# Set your auth token
export KCS_TOKEN="your-auth-token-here"
export KCS_URL="http://localhost:8080"

# Test authentication
curl -H "Authorization: Bearer $KCS_TOKEN" $KCS_URL/health
```text

## Common Workflows

### 1. Search for Code

Find code related to file operations:

```bash
curl -X POST \
  -H "Authorization: Bearer $KCS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "read from file descriptor",
    "topK": 5
  }' \
  $KCS_URL/mcp/tools/search_code
```text

### 2. Get Symbol Information

Get details about a specific function:

```bash
curl -X POST \
  -H "Authorization: Bearer $KCS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "sys_read"
  }' \
  $KCS_URL/mcp/tools/get_symbol
```text

### 3. Find Function Callers

Discover what calls a function:

```bash
curl -X POST \
  -H "Authorization: Bearer $KCS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "vfs_read",
    "depth": 2
  }' \
  $KCS_URL/mcp/tools/who_calls
```text

### 4. Trace Entry Point Flow

Follow execution from syscall to implementation:

```bash
curl -X POST \
  -H "Authorization: Bearer $KCS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entry": "__NR_read"
  }' \
  $KCS_URL/mcp/tools/entrypoint_flow
```text

### 5. Analyze Change Impact

Assess impact of code changes:

```bash
curl -X POST \
  -H "Authorization: Bearer $KCS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["fs/read_write.c"],
    "config": "x86_64:defconfig"
  }' \
  $KCS_URL/mcp/tools/impact_of
```text

## Programming Language Examples

### Python

```python
import requests

class KCSClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def search_code(self, query: str, top_k: int = 10):
        response = requests.post(
            f'{self.base_url}/mcp/tools/search_code',
            headers=self.headers,
            json={'query': query, 'topK': top_k}
        )
        return response.json()
    
    def get_symbol(self, symbol: str):
        response = requests.post(
            f'{self.base_url}/mcp/tools/get_symbol',
            headers=self.headers,
            json={'symbol': symbol}
        )
        return response.json()

# Usage
client = KCSClient('http://localhost:8080', 'your-token')
results = client.search_code('memory allocation')
symbol_info = client.get_symbol('kmalloc')
```text

### JavaScript/TypeScript

```typescript
class KCSClient {
    constructor(private baseUrl: string, private token: string) {}
    
    private get headers() {
        return {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async searchCode(query: string, topK: number = 10) {
        const response = await fetch(`${this.baseUrl}/mcp/tools/search_code`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ query, topK })
        });
        return response.json();
    }
    
    async getSymbol(symbol: string) {
        const response = await fetch(`${this.baseUrl}/mcp/tools/get_symbol`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ symbol })
        });
        return response.json();
    }
}

// Usage
const client = new KCSClient('http://localhost:8080', 'your-token');
const results = await client.searchCode('network protocol');
const symbolInfo = await client.getSymbol('tcp_sendmsg');
```text

## Response Processing

### Working with Citations

All KCS responses include citations. Here's how to process them:

```python
def extract_file_references(response):
    """Extract file:line references from KCS response."""
    citations = response.get('cites', [])
    references = []
    
    for cite in citations:
        span = cite['span']
        file_path = span['path']
        start_line = span['start']
        end_line = span['end']
        sha = span['sha']
        
        if start_line == end_line:
            ref = f"{file_path}:{start_line}@{sha}"
        else:
            ref = f"{file_path}:{start_line}-{end_line}@{sha}"
        
        references.append(ref)
    
    return references

# Example usage
search_results = client.search_code('syscall definition')
file_refs = extract_file_references(search_results)
print("Found references:", file_refs)
```text

### Handling Errors

```python
def make_kcs_request(client, endpoint, data):
    """Make KCS request with proper error handling."""
    try:
        response = requests.post(
            f'{client.base_url}/mcp/tools/{endpoint}',
            headers=client.headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 401:
            raise Exception("Authentication failed - check token")
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded - wait and retry")
        elif response.status_code >= 400:
            error_info = response.json()
            raise Exception(f"API error: {error_info.get('message', 'Unknown error')}")
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out - try again")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to KCS server")
```text

## Best Practices

### 1. Efficient Querying

- Use specific search terms rather than broad queries
- Limit `topK` to reasonable values (5-20)
- Cache results when appropriate

### 2. Citation Handling

- Always process citations from responses
- Use citations for code navigation
- Verify file paths exist in your kernel version

### 3. Error Handling

- Handle authentication errors gracefully
- Implement retry logic for transient failures
- Respect rate limits

### 4. Performance

- Batch related queries when possible
- Use appropriate timeouts
- Monitor response times

## Troubleshooting

### Common Issues

1. **401 Unauthorized**: Check auth token
2. **404 Not Found**: Verify endpoint URL
3. **422 Validation Error**: Check request data format
4. **500 Server Error**: Server issue - check logs
5. **Timeout**: Increase timeout or check server load
