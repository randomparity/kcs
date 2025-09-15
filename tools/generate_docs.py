#!/usr/bin/env python3
"""Generate API documentation from OpenAPI specification.

Creates comprehensive markdown documentation for the KCS MCP API,
including examples, authentication, and integration guides.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


class OpenAPIDocGenerator:
    """Generates documentation from OpenAPI specification."""

    def __init__(self, spec_file: Path, output_dir: Path):
        """Initialize generator.

        Args:
            spec_file: Path to OpenAPI specification file
            output_dir: Directory to write documentation files
        """
        self.spec_file = spec_file
        self.output_dir = output_dir
        self.spec = self._load_spec()

    def _load_spec(self) -> dict[str, Any]:
        """Load OpenAPI specification from file."""
        try:
            with open(self.spec_file) as f:
                if self.spec_file.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load OpenAPI spec: {e}") from e

    def generate_all(self):
        """Generate all documentation files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate main API documentation
        self._generate_api_overview()
        self._generate_authentication_guide()
        self._generate_endpoint_docs()
        self._generate_schema_docs()
        self._generate_examples()
        self._generate_integration_guide()
        self._generate_error_handling()

        print(f"Documentation generated in {self.output_dir}")

    def _generate_api_overview(self):
        """Generate API overview documentation."""
        info = self.spec.get("info", {})
        servers = self.spec.get("servers", [])

        content = f"""# {info.get("title", "API Documentation")}

{info.get("description", "API Documentation")}

**Version:** {info.get("version", "1.0.0")}

## Overview

The Kernel Context Server (KCS) provides a Model Context Protocol (MCP) API for analyzing Linux kernel source code. This API enables AI coding assistants to understand kernel internals, trace function calls, analyze impact of changes, and provide accurate kernel development guidance.

## Key Features

- **Read-Only Access**: KCS never modifies kernel source code
- **Constitutional Citations**: All results include file:line references
- **Configuration Awareness**: Results tagged with kernel configuration
- **Performance Optimized**: Queries complete in <600ms (p95)
- **MCP Protocol**: Native integration with AI coding assistants

## Base URLs

"""

        for server in servers:
            content += (
                f"- **{server.get('description', 'Server')}**: `{server.get('url')}`\n"
            )

        content += """
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
"""

        with open(self.output_dir / "api-overview.md", "w") as f:
            f.write(content)

    def _generate_authentication_guide(self):
        """Generate authentication documentation."""
        security_schemes = self.spec.get("components", {}).get("securitySchemes", {})

        content = """# Authentication

## Overview

KCS API uses JWT Bearer token authentication. All API requests (except `/health`) require a valid authentication token.

"""

        if "bearerAuth" in security_schemes:
            bearer_auth = security_schemes["bearerAuth"]
            content += f"""## Bearer Token Authentication

**Type**: {bearer_auth.get("type", "http")}
**Scheme**: {bearer_auth.get("scheme", "bearer")}
**Format**: {bearer_auth.get("bearerFormat", "JWT")}

### Request Headers

Include the authentication token in the `Authorization` header:

```http
Authorization: Bearer YOUR_TOKEN_HERE
```

### Example Request

```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
     -H "Content-Type: application/json" \\
     {self.spec.get("servers", [{}])[0].get("url", "http://localhost:8080")}/health
```

"""

        content += """## Obtaining Tokens

Contact your KCS administrator to obtain authentication tokens. Tokens should be:

- Kept secure and not shared
- Rotated regularly for security
- Used only over HTTPS in production

## Token Validation

The server validates tokens on each request. Invalid tokens return:

```json
{
  "error": "unauthorized",
  "message": "Invalid or expired token"
}
```

## Rate Limiting

API requests may be rate limited based on token. Check response headers:

- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests in window
- `X-RateLimit-Reset`: When the window resets

## Troubleshooting

### Common Authentication Errors

| Error | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Missing or invalid token | Check Authorization header |
| 403 Forbidden | Token lacks permissions | Contact administrator |
| 429 Too Many Requests | Rate limit exceeded | Wait and retry |

### Testing Authentication

Use the health endpoint to test authentication:

```bash
# This should work without auth
curl {self.spec.get('servers', [{}])[0].get('url', 'http://localhost:8080')}/health

# This requires auth
curl -H "Authorization: Bearer YOUR_TOKEN" \\
     {self.spec.get('servers', [{}])[0].get('url', 'http://localhost:8080')}/mcp/resources
```
"""

        with open(self.output_dir / "authentication.md", "w") as f:
            f.write(content)

    def _generate_endpoint_docs(self):
        """Generate endpoint documentation."""
        paths = self.spec.get("paths", {})

        content = """# API Endpoints

## Overview

The KCS API provides endpoints for kernel analysis, organized into categories:

- **Tools**: MCP tools for kernel analysis (`/mcp/tools/*`)
- **Resources**: MCP resources for data access (`/mcp/resources/*`)
- **System**: Health and metrics (`/health`, `/metrics`)

"""

        # Group endpoints by category
        tools_endpoints = []
        resources_endpoints = []
        system_endpoints = []

        for path, methods in paths.items():
            if "/mcp/tools/" in path:
                tools_endpoints.append((path, methods))
            elif "/mcp/resources" in path:
                resources_endpoints.append((path, methods))
            else:
                system_endpoints.append((path, methods))

        # Generate tools documentation
        content += "## MCP Tools\n\n"
        content += "MCP tools perform kernel analysis operations.\n\n"

        for path, methods in sorted(tools_endpoints):
            content += self._generate_endpoint_section(path, methods)

        # Generate resources documentation
        content += "\n## MCP Resources\n\n"
        content += "MCP resources provide access to indexed kernel data.\n\n"

        for path, methods in sorted(resources_endpoints):
            content += self._generate_endpoint_section(path, methods)

        # Generate system documentation
        content += "\n## System Endpoints\n\n"
        content += "System endpoints provide health and monitoring information.\n\n"

        for path, methods in sorted(system_endpoints):
            content += self._generate_endpoint_section(path, methods)

        with open(self.output_dir / "endpoints.md", "w") as f:
            f.write(content)

    def _generate_endpoint_section(self, path: str, methods: dict[str, Any]) -> str:
        """Generate documentation for a single endpoint."""
        content = ""

        for method, operation in methods.items():
            if method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                continue

            summary = operation.get("summary", "No summary available")
            description = operation.get("description", "")

            content += f"\n### {method.upper()} {path}\n\n"
            content += f"**Summary**: {summary}\n\n"

            if description:
                content += f"{description}\n\n"

            # Parameters
            parameters = operation.get("parameters", [])
            if parameters:
                content += "**Parameters**:\n\n"
                for param in parameters:
                    name = param.get("name", "unknown")
                    location = param.get("in", "unknown")
                    required = param.get("required", False)
                    param_type = param.get("schema", {}).get("type", "unknown")
                    param_desc = param.get("description", "No description")

                    req_text = " (required)" if required else " (optional)"
                    content += f"- `{name}` ({location}, {param_type}){req_text}: {param_desc}\n"
                content += "\n"

            # Request body
            request_body = operation.get("requestBody")
            if request_body:
                content += "**Request Body**:\n\n"
                content_types = request_body.get("content", {})

                for content_type, schema_info in content_types.items():
                    content += f"Content-Type: `{content_type}`\n\n"

                    schema = schema_info.get("schema", {})
                    if "properties" in schema:
                        content += "Properties:\n"
                        for prop_name, prop_schema in schema["properties"].items():
                            prop_type = prop_schema.get("type", "unknown")
                            prop_desc = prop_schema.get("description", "No description")
                            is_required = prop_name in schema.get("required", [])
                            req_text = " (required)" if is_required else " (optional)"
                            content += f"- `{prop_name}` ({prop_type}){req_text}: {prop_desc}\n"

                    # Add example if available
                    example = schema_info.get("example")
                    if example:
                        content += f"\nExample:\n```json\n{json.dumps(example, indent=2)}\n```\n"
                content += "\n"

            # Responses
            responses = operation.get("responses", {})
            if responses:
                content += "**Responses**:\n\n"
                for status_code, response_info in responses.items():
                    description = response_info.get("description", "No description")
                    content += f"- **{status_code}**: {description}\n"

                    # Response schema
                    response_content = response_info.get("content", {})
                    for content_type, schema_info in response_content.items():
                        schema = schema_info.get("schema", {})
                        if schema:
                            content += f"  - Content-Type: `{content_type}`\n"

                            # Add example response if available
                            example = schema_info.get("example")
                            if example:
                                content += f"  - Example:\n    ```json\n    {json.dumps(example, indent=4)}\n    ```\n"
                content += "\n"

            # Add curl example
            content += self._generate_curl_example(path, method, operation)
            content += "\n"

        return content

    def _generate_curl_example(
        self, path: str, method: str, operation: dict[str, Any]
    ) -> str:
        """Generate curl example for an endpoint."""
        base_url = self.spec.get("servers", [{}])[0].get("url", "http://localhost:8080")

        curl_cmd = f"curl -X {method.upper()}"

        # Add auth header if required
        security = operation.get("security", self.spec.get("security", []))
        if security and any("bearerAuth" in sec for sec in security):
            curl_cmd += ' \\\n     -H "Authorization: Bearer YOUR_TOKEN"'

        # Add content type for requests with body
        if method.upper() in ["POST", "PUT", "PATCH"] and operation.get("requestBody"):
            curl_cmd += ' \\\n     -H "Content-Type: application/json"'

        # Add example request body
        request_body = operation.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema", {})

            # Create example data
            example_data = self._create_example_from_schema(schema)
            if example_data:
                curl_cmd += f" \\\n     -d '{json.dumps(example_data, indent=2)}'"

        curl_cmd += f" \\\n     {base_url}{path}"

        return f"**Example**:\n```bash\n{curl_cmd}\n```"

    def _create_example_from_schema(
        self, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create example data from JSON schema."""
        if not schema:
            return None

        properties = schema.get("properties", {})
        example = {}

        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            prop_example = prop_schema.get("example")

            if prop_example is not None:
                example[prop_name] = prop_example
            elif prop_type == "string":
                if "symbol" in prop_name.lower():
                    example[prop_name] = "sys_read"
                elif "query" in prop_name.lower():
                    example[prop_name] = "read from file descriptor"
                elif "path" in prop_name.lower():
                    example[prop_name] = "fs/read_write.c"
                else:
                    example[prop_name] = f"example_{prop_name}"
            elif prop_type == "integer":
                example[prop_name] = 10
            elif prop_type == "number":
                example[prop_name] = 1.0
            elif prop_type == "boolean":
                example[prop_name] = True
            elif prop_type == "array":
                example[prop_name] = ["example_item"]
            elif prop_type == "object":
                example[prop_name] = {}

        return example if example else None

    def _generate_schema_docs(self):
        """Generate schema documentation."""
        components = self.spec.get("components", {})
        schemas = components.get("schemas", {})

        content = """# Data Schemas

## Overview

KCS API uses structured data schemas for requests and responses. All schemas follow OpenAPI 3.0 specification.

## Core Concepts

### Citations and Spans

All KCS responses include citations with file:line references per constitutional requirements.

"""

        # Document key schemas
        key_schemas = ["Span", "Citation", "SymbolInfo", "SearchHit", "ImpactResult"]

        for schema_name in key_schemas:
            if schema_name in schemas:
                content += self._generate_schema_section(
                    schema_name, schemas[schema_name]
                )

        content += "\n## Other Schemas\n\n"

        for schema_name, schema in schemas.items():
            if schema_name not in key_schemas:
                content += self._generate_schema_section(schema_name, schema)

        with open(self.output_dir / "schemas.md", "w") as f:
            f.write(content)

    def _generate_schema_section(self, name: str, schema: dict[str, Any]) -> str:
        """Generate documentation for a schema."""
        content = f"\n### {name}\n\n"

        description = schema.get("description", "No description available")
        content += f"{description}\n\n"

        schema_type = schema.get("type", "object")
        content += f"**Type**: {schema_type}\n\n"

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if properties:
            content += "**Properties**:\n\n"
            content += "| Property | Type | Required | Description |\n"
            content += "|----------|------|----------|-------------|\n"

            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "unknown")
                prop_format = prop_schema.get("format")
                if prop_format:
                    prop_type += f" ({prop_format})"

                is_required = "✓" if prop_name in required else ""
                prop_desc = prop_schema.get("description", "No description")

                # Handle references
                if "$ref" in prop_schema:
                    ref_name = prop_schema["$ref"].split("/")[-1]
                    prop_type = f"[{ref_name}](#{ref_name.lower()})"

                content += (
                    f"| `{prop_name}` | {prop_type} | {is_required} | {prop_desc} |\n"
                )

            content += "\n"

        # Add example if available
        example = schema.get("example")
        if example:
            content += (
                f"**Example**:\n```json\n{json.dumps(example, indent=2)}\n```\n\n"
            )
        elif properties:
            # Generate example from properties
            example_obj = {}
            for prop_name, prop_schema in properties.items():
                if prop_name in required[:3]:  # Include first 3 required properties
                    prop_example = prop_schema.get("example")
                    if prop_example is not None:
                        example_obj[prop_name] = prop_example
                    elif prop_schema.get("type") == "string":
                        example_obj[prop_name] = f"example_{prop_name}"
                    elif prop_schema.get("type") == "integer":
                        example_obj[prop_name] = 42

            if example_obj:
                content += f"**Example**:\n```json\n{json.dumps(example_obj, indent=2)}\n```\n\n"

        return content

    def _generate_examples(self):
        """Generate usage examples."""
        content = """# Usage Examples

## Overview

This guide provides practical examples for common KCS API usage patterns.

## Authentication Setup

```bash
# Set your auth token
export KCS_TOKEN="your-auth-token-here"
export KCS_URL="http://localhost:8080"

# Test authentication
curl -H "Authorization: Bearer $KCS_TOKEN" $KCS_URL/health
```

## Common Workflows

### 1. Search for Code

Find code related to file operations:

```bash
curl -X POST \\
  -H "Authorization: Bearer $KCS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "read from file descriptor",
    "topK": 5
  }' \\
  $KCS_URL/mcp/tools/search_code
```

### 2. Get Symbol Information

Get details about a specific function:

```bash
curl -X POST \\
  -H "Authorization: Bearer $KCS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "symbol": "sys_read"
  }' \\
  $KCS_URL/mcp/tools/get_symbol
```

### 3. Find Function Callers

Discover what calls a function:

```bash
curl -X POST \\
  -H "Authorization: Bearer $KCS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "symbol": "vfs_read",
    "depth": 2
  }' \\
  $KCS_URL/mcp/tools/who_calls
```

### 4. Trace Entry Point Flow

Follow execution from syscall to implementation:

```bash
curl -X POST \\
  -H "Authorization: Bearer $KCS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "entry": "__NR_read"
  }' \\
  $KCS_URL/mcp/tools/entrypoint_flow
```

### 5. Analyze Change Impact

Assess impact of code changes:

```bash
curl -X POST \\
  -H "Authorization: Bearer $KCS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "files": ["fs/read_write.c"],
    "config": "x86_64:defconfig"
  }' \\
  $KCS_URL/mcp/tools/impact_of
```

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
```

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
```

## Response Processing

### Working with Citations

All KCS responses include citations. Here's how to process them:

```python
def extract_file_references(response):
    \"\"\"Extract file:line references from KCS response.\"\"\"
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
```

### Handling Errors

```python
def make_kcs_request(client, endpoint, data):
    \"\"\"Make KCS request with proper error handling.\"\"\"
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
```

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
"""

        with open(self.output_dir / "examples.md", "w") as f:
            f.write(content)

    def _generate_integration_guide(self):
        """Generate integration guide."""
        content = """# Integration Guide

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
```

### Vim/Neovim

Create a Vim plugin for KCS integration:

```vim
\" kcs.vim
function! KCSSearchSymbol()
    let symbol = expand('<cword>')
    let result = system('curl -s -H "Authorization: Bearer ' . g:kcs_token . '" -H "Content-Type: application/json" -d "{\\"symbol\\": \\"' . symbol . '\\"}" ' . g:kcs_url . '/mcp/tools/get_symbol')
    echo result
endfunction

command! KCSSearch call KCSSearchSymbol()
nnoremap <leader>ks :KCSSearch<CR>
```

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
```

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
        python tools/ci-analysis.py \\
          --pr-number ${{ github.event.number }} \\
          --kcs-url "$KCS_SERVER_URL" \\
          --kcs-token "$KCS_AUTH_TOKEN"
```

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
```

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
```

## Configuration Management

### Environment Variables

```bash
# .env
KCS_SERVER_URL=https://kcs.example.com
KCS_AUTH_TOKEN=your-token-here
KCS_TIMEOUT=30
KCS_RETRY_COUNT=3
KCS_CACHE_TTL=3600
```

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
```

## Monitoring and Observability

### Health Checks

```python
def check_kcs_health(client: KCSClient) -> bool:
    \"\"\"Check if KCS server is healthy.\"\"\"
    try:
        response = requests.get(
            f'{client.config.base_url}/health',
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False
```

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
```

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
```

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
```

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
```

## Support

For integration support:
- Check the [Examples](examples.md) for code samples
- Review [Error Handling](error-handling.md) for common issues
- Monitor server logs for detailed error information
- Contact your KCS administrator for access issues
"""

        with open(self.output_dir / "integration.md", "w") as f:
            f.write(content)

    def _generate_error_handling(self):
        """Generate error handling documentation."""
        content = """# Error Handling

## Overview

KCS API uses standard HTTP status codes and provides detailed error information to help diagnose and resolve issues.

## HTTP Status Codes

### Success Codes (2xx)

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |

### Client Error Codes (4xx)

| Code | Description | Common Causes | Solution |
|------|-------------|---------------|----------|
| 400 | Bad Request | Invalid request format, missing required fields | Check request structure and required parameters |
| 401 | Unauthorized | Missing or invalid auth token | Verify Authorization header and token validity |
| 403 | Forbidden | Token lacks required permissions | Contact administrator for proper permissions |
| 404 | Not Found | Endpoint or resource doesn't exist | Check URL path and resource identifier |
| 422 | Unprocessable Entity | Valid format but semantic errors | Review request data for logical errors |
| 429 | Too Many Requests | Rate limit exceeded | Implement backoff and retry logic |

### Server Error Codes (5xx)

| Code | Description | Response |
|------|-------------|----------|
| 500 | Internal Server Error | Server-side issue - check logs, contact support |
| 502 | Bad Gateway | Proxy/gateway issue - retry or check infrastructure |
| 503 | Service Unavailable | Server overloaded or maintenance - retry later |
| 504 | Gateway Timeout | Request timed out - increase timeout or retry |

## Error Response Format

All error responses follow a consistent format:

```json
{
  "error": "error_type",
  "message": "Human-readable error description",
  "details": {
    "field": "specific error details",
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_123456789"
  }
}
```

### Error Types

- `invalid_request`: Request format or structure is invalid
- `authentication_failed`: Authentication token is missing or invalid
- `authorization_failed`: Token lacks required permissions
- `validation_error`: Request data fails validation
- `not_found`: Requested resource doesn't exist
- `rate_limit_exceeded`: Too many requests in time window
- `server_error`: Internal server error
- `service_unavailable`: Service temporarily unavailable

## Common Error Scenarios

### Authentication Errors

#### Missing Authorization Header

```json
{
  "error": "authentication_failed",
  "message": "Authorization header required",
  "details": {
    "expected_header": "Authorization: Bearer <token>"
  }
}
```

**Solution**: Add Authorization header to request.

#### Invalid Token Format

```json
{
  "error": "authentication_failed",
  "message": "Invalid token format",
  "details": {
    "expected_format": "Bearer <jwt_token>"
  }
}
```

**Solution**: Ensure token uses "Bearer " prefix.

#### Expired Token

```json
{
  "error": "authentication_failed",
  "message": "Token has expired",
  "details": {
    "expired_at": "2024-01-01T12:00:00Z",
    "current_time": "2024-01-01T13:00:00Z"
  }
}
```

**Solution**: Obtain a new token.

### Validation Errors

#### Missing Required Field

```json
{
  "error": "validation_error",
  "message": "Missing required field",
  "details": {
    "field": "symbol",
    "location": "request_body"
  }
}
```

#### Invalid Field Value

```json
{
  "error": "validation_error",
  "message": "Invalid field value",
  "details": {
    "field": "topK",
    "value": -5,
    "expected": "positive integer"
  }
}
```

### Rate Limiting

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded",
  "details": {
    "limit": 100,
    "window": "1 minute",
    "reset_at": "2024-01-01T12:01:00Z"
  }
}
```

**Response Headers**:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when window resets

### Constitutional Violations

KCS enforces constitutional requirements and will return errors for violations:

#### Missing Citations

```json
{
  "error": "constitutional_violation",
  "message": "Response contains claims without citations",
  "details": {
    "requirement": "All results must include file:line references",
    "missing_citations": ["search_results"]
  }
}
```

#### Read-Only Violation

```json
{
  "error": "constitutional_violation",
  "message": "Attempted to modify kernel source",
  "details": {
    "requirement": "KCS is read-only",
    "attempted_action": "file_write"
  }
}
```

## Error Handling Best Practices

### 1. Implement Retry Logic

```python
import time
import random

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    \"\"\"Retry function with exponential backoff.\"\"\"
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code < 500:
                # Client error - don't retry
                raise

            if attempt == max_retries - 1:
                # Last attempt failed
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

### 2. Handle Rate Limits

```python
def handle_rate_limit(response):
    \"\"\"Handle rate limit responses.\"\"\"
    if response.status_code == 429:
        reset_time = response.headers.get('X-RateLimit-Reset')
        if reset_time:
            wait_time = int(reset_time) - int(time.time())
            if wait_time > 0:
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return True
    return False
```

### 3. Parse Error Details

```python
def parse_error_response(response):
    \"\"\"Parse error response and extract details.\"\"\"
    try:
        error_data = response.json()
        error_type = error_data.get('error', 'unknown')
        message = error_data.get('message', 'No error message')
        details = error_data.get('details', {})

        return {
            'type': error_type,
            'message': message,
            'details': details,
            'status_code': response.status_code
        }
    except ValueError:
        # Response is not JSON
        return {
            'type': 'http_error',
            'message': f'HTTP {response.status_code}: {response.reason}',
            'details': {'body': response.text},
            'status_code': response.status_code
        }
```

### 4. Comprehensive Error Handler

```python
class KCSError(Exception):
    \"\"\"Base exception for KCS API errors.\"\"\"
    def __init__(self, error_type, message, details=None, status_code=None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

class KCSClient:
    def _handle_response(self, response):
        \"\"\"Handle API response and raise appropriate exceptions.\"\"\"
        if response.status_code == 200:
            return response.json()

        # Handle rate limiting
        if response.status_code == 429:
            if self.handle_rate_limit(response):
                # Retry handled, return None to indicate retry
                return None

        # Parse error response
        error_info = parse_error_response(response)

        # Raise specific exception based on error type
        if error_info['type'] == 'authentication_failed':
            raise KCSError(
                'authentication_failed',
                'Authentication failed - check token',
                error_info['details'],
                response.status_code
            )
        elif error_info['type'] == 'validation_error':
            raise KCSError(
                'validation_error',
                f"Validation failed: {error_info['message']}",
                error_info['details'],
                response.status_code
            )
        else:
            raise KCSError(
                error_info['type'],
                error_info['message'],
                error_info['details'],
                response.status_code
            )
```

## Debugging

### Enable Request/Response Logging

```python
import logging
import http.client as http_client

# Enable debug logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# Enable HTTP debug
http_client.HTTPConnection.debuglevel = 1
```

### Log Request Context

```python
def log_request_context(func):
    \"\"\"Decorator to log request context for debugging.\"\"\"
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())
        logger.info(f"Request {request_id}: {func.__name__} called with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.info(f"Request {request_id}: Success")
            return result
        except Exception as e:
            logger.error(f"Request {request_id}: Error - {str(e)}")
            raise

    return wrapper
```

## Monitoring and Alerting

### Health Check Implementation

```python
def health_check():
    \"\"\"Perform comprehensive health check.\"\"\"
    checks = {
        'server_reachable': False,
        'authentication_valid': False,
        'basic_functionality': False
    }

    try:
        # Test server reachability
        response = requests.get(f'{base_url}/health', timeout=5)
        checks['server_reachable'] = response.status_code == 200

        # Test authentication
        response = requests.get(
            f'{base_url}/mcp/resources',
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )
        checks['authentication_valid'] = response.status_code == 200

        # Test basic functionality
        response = requests.post(
            f'{base_url}/mcp/tools/search_code',
            headers={'Authorization': f'Bearer {token}'},
            json={'query': 'test', 'topK': 1},
            timeout=10
        )
        checks['basic_functionality'] = response.status_code == 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")

    return checks
```

### Error Rate Monitoring

```python
from collections import deque
import time

class ErrorRateMonitor:
    def __init__(self, window_size=100):
        self.requests = deque(maxlen=window_size)

    def record_request(self, success=True):
        self.requests.append({
            'timestamp': time.time(),
            'success': success
        })

    def get_error_rate(self):
        if not self.requests:
            return 0.0

        errors = sum(1 for req in self.requests if not req['success'])
        return errors / len(self.requests)

    def should_alert(self, threshold=0.1):
        return self.get_error_rate() > threshold
```

## Recovery Strategies

### Circuit Breaker Pattern

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

## Support and Escalation

When encountering persistent errors:

1. **Check Status Page**: Monitor KCS service status
2. **Review Logs**: Check both client and server logs
3. **Test Connectivity**: Verify network path to server
4. **Validate Configuration**: Ensure correct URLs and tokens
5. **Contact Support**: Provide error details and request ID

### Error Report Template

```
Error Report:
- Timestamp: 2024-01-01T12:00:00Z
- Request ID: req_123456789
- Error Type: authentication_failed
- Status Code: 401
- Endpoint: /mcp/tools/search_code
- Request Body: {"query": "example"}
- Response: {"error": "authentication_failed", "message": "Invalid token"}
- Client Version: kcs-client/1.0.0
- Server Version: kcs-server/1.0.0
```
"""

        with open(self.output_dir / "error-handling.md", "w") as f:
            f.write(content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate API documentation from OpenAPI spec"
    )
    parser.add_argument("spec_file", help="Path to OpenAPI specification file")
    parser.add_argument(
        "-o", "--output", default="docs", help="Output directory (default: docs)"
    )

    args = parser.parse_args()

    spec_path = Path(args.spec_file)
    output_path = Path(args.output)

    if not spec_path.exists():
        print(f"Error: Specification file not found: {spec_path}")
        sys.exit(1)

    try:
        generator = OpenAPIDocGenerator(spec_path, output_path)
        generator.generate_all()
        print(f"✅ Documentation generated successfully in {output_path}")

        # List generated files
        print("\nGenerated files:")
        for file_path in sorted(output_path.glob("*.md")):
            print(f"  - {file_path.name}")

    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
