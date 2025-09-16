# Integration Guide

## Overview

This guide covers integrating KCS with various development environments and tools.

## Prerequisites

- KCS server running and accessible
- Valid authentication token
- Network access to KCS server

## IDE Integrations

### Visual Studio Code

Create a VS Code extension to integrate KCS:

```typescript
// extension.ts
import * as vscode from 'vscode';
import { KCSClient } from './kcs-client';

export function activate(context: vscode.ExtensionContext) {
    const client = new KCSClient(
        vscode.workspace.getConfiguration('kcs').get('serverUrl'),
        vscode.workspace.getConfiguration('kcs').get('authToken')
    );
    
    // Command to search for symbol under cursor
    const searchCommand = vscode.commands.registerCommand('kcs.searchSymbol', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        
        const document = editor.document;
        const position = editor.selection.active;
        const symbol = document.getText(document.getWordRangeAtPosition(position));
        
        try {
            const result = await client.getSymbol(symbol);
            // Display result in webview or output channel
            vscode.window.showInformationMessage(`Found symbol: ${result.name}`);
        } catch (error) {
            vscode.window.showErrorMessage(`KCS error: ${error.message}`);
        }
    });
    
    context.subscriptions.push(searchCommand);
}
```text

### Vim/Neovim

Create a Vim plugin for KCS integration:

```vim
" kcs.vim
function! KCSSearchSymbol()
    let symbol = expand('<cword>')
    let result = system('curl -s -H "Authorization: Bearer ' . g:kcs_token . '" -H "Content-Type: application/json" -d "{\"symbol\": \"' . symbol . '\"}" ' . g:kcs_url . '/mcp/tools/get_symbol')
    echo result
endfunction

command! KCSSearch call KCSSearchSymbol()
nnoremap <leader>ks :KCSSearch<CR>
```text

### Emacs

```elisp
;; kcs.el
(require 'request)
(require 'json)

(defvar kcs-server-url "http://localhost:8080")
(defvar kcs-auth-token nil)

(defun kcs-search-symbol (symbol)
  "Search for SYMBOL using KCS."
  (interactive (list (thing-at-point 'symbol)))
  (request
    (concat kcs-server-url "/mcp/tools/get_symbol")
    :type "POST"
    :headers `(("Authorization" . ,(concat "Bearer " kcs-auth-token))
               ("Content-Type" . "application/json"))
    :data (json-encode `((symbol . ,symbol)))
    :parser 'json-read
    :success (cl-function
              (lambda (&key data &allow-other-keys)
                (message "Symbol: %s" (alist-get 'name data))))
    :error (cl-function
            (lambda (&key error-thrown &allow-other-keys)
              (message "KCS error: %s" error-thrown)))))
```text

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/kcs-analysis.yml
name: KCS Analysis

on:
  pull_request:
    paths:
      - '**/*.c'
      - '**/*.h'

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run KCS Impact Analysis
      env:
        KCS_SERVER_URL: ${{ secrets.KCS_SERVER_URL }}
        KCS_AUTH_TOKEN: ${{ secrets.KCS_AUTH_TOKEN }}
      run: |
        python tools/ci-analysis.py \
          --pr-number ${{ github.event.number }} \
          --kcs-url "$KCS_SERVER_URL" \
          --kcs-token "$KCS_AUTH_TOKEN"
```text

### GitLab CI

```yaml
# .gitlab-ci.yml
kcs-analysis:
  stage: test
  script:
    - python tools/ci-analysis.py
      --merge-request $CI_MERGE_REQUEST_IID
      --kcs-url $KCS_SERVER_URL
      --kcs-token $KCS_AUTH_TOKEN
  only:
    - merge_requests
  variables:
    KCS_SERVER_URL: "https://kcs.example.com"
```text

## API Client Libraries

### Python Library

```python
# kcs_client.py
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class KCSConfig:
    base_url: str
    auth_token: str
    timeout: int = 30

class KCSClient:
    def __init__(self, config: KCSConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.auth_token}',
            'Content-Type': 'application/json'
        })
    
    def search_code(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        response = self.session.post(
            f'{self.config.base_url}/mcp/tools/search_code',
            json={'query': query, 'topK': top_k},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_symbol(self, symbol: str) -> Dict[str, Any]:
        response = self.session.post(
            f'{self.config.base_url}/mcp/tools/get_symbol',
            json={'symbol': symbol},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def who_calls(self, symbol: str, depth: int = 1) -> Dict[str, Any]:
        response = self.session.post(
            f'{self.config.base_url}/mcp/tools/who_calls',
            json={'symbol': symbol, 'depth': depth},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def impact_analysis(self, files: List[str], config: str = "x86_64:defconfig") -> Dict[str, Any]:
        response = self.session.post(
            f'{self.config.base_url}/mcp/tools/impact_of',
            json={'files': files, 'config': config},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
```text

## Configuration Management

### Environment Variables

```bash
# .env
KCS_SERVER_URL=https://kcs.example.com
KCS_AUTH_TOKEN=your-token-here
KCS_TIMEOUT=30
KCS_RETRY_COUNT=3
KCS_CACHE_TTL=3600
```text

### Configuration File

```yaml
# kcs-config.yaml
server:
  url: https://kcs.example.com
  timeout: 30
  
auth:
  token: ${KCS_AUTH_TOKEN}
  
client:
  retry_count: 3
  cache_ttl: 3600
  user_agent: "MyApp/1.0"
  
features:
  enable_caching: true
  enable_metrics: true
```text

## Monitoring and Observability

### Health Checks

```python
def check_kcs_health(client: KCSClient) -> bool:
    """Check if KCS server is healthy."""
    try:
        response = requests.get(
            f'{client.config.base_url}/health',
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False
```text

### Metrics Collection

```python
import time
from prometheus_client import Counter, Histogram

kcs_requests_total = Counter('kcs_requests_total', 'Total KCS requests', ['method', 'status'])
kcs_request_duration = Histogram('kcs_request_duration_seconds', 'KCS request duration')

def instrumented_request(method: str, *args, **kwargs):
    start_time = time.time()
    try:
        response = getattr(client, method)(*args, **kwargs)
        kcs_requests_total.labels(method=method, status='success').inc()
        return response
    except Exception as e:
        kcs_requests_total.labels(method=method, status='error').inc()
        raise
    finally:
        duration = time.time() - start_time
        kcs_request_duration.observe(duration)
```text

## Security Considerations

### Token Management

- Store tokens securely (environment variables, key vaults)
- Rotate tokens regularly
- Use different tokens for different environments
- Never commit tokens to version control

### Network Security

- Use HTTPS in production
- Implement proper certificate validation
- Consider VPN or private networks for sensitive deployments

### Access Control

- Implement least-privilege access
- Monitor API usage for anomalies
- Log all API requests for audit

## Performance Optimization

### Caching

```python
from functools import lru_cache
import hashlib

class CachedKCSClient(KCSClient):
    @lru_cache(maxsize=1000)
    def get_symbol_cached(self, symbol: str) -> Dict[str, Any]:
        return self.get_symbol(symbol)
    
    def search_code_cached(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        # Use query hash as cache key
        cache_key = hashlib.md5(f"{query}:{top_k}".encode()).hexdigest()
        # Implement your caching logic here
        return self.search_code(query, top_k)
```text

### Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_optimized_session():
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=100,
        pool_maxsize=100
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session
```text

## Troubleshooting

### Common Integration Issues

1. **Authentication Failures**
   - Verify token is correct and not expired
   - Check Authorization header format
   - Ensure token has required permissions

2. **Connection Issues**
   - Verify server URL is accessible
   - Check firewall and network configuration
   - Test with curl or similar tool

3. **Performance Issues**
   - Implement connection pooling
   - Use appropriate timeouts
   - Consider caching frequently accessed data

4. **Rate Limiting**
   - Implement exponential backoff
   - Monitor rate limit headers
   - Distribute requests across time

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add request/response logging
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1
```text

## Support

For integration support:
- Check the [Examples](examples.md) for code samples
- Review [Error Handling](error-handling.md) for common issues
- Monitor server logs for detailed error information
- Contact your KCS administrator for access issues
