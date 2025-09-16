# Authentication

## Overview

KCS API uses JWT Bearer token authentication. All API requests (except `/health`)
require a valid authentication token.

## Bearer Token Authentication

**Type**: http
**Scheme**: bearer
**Format**: JWT

### Request Headers

Include the authentication token in the `Authorization` header:

```http
Authorization: Bearer YOUR_TOKEN_HERE
```text

### Example Request

```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
     -H "Content-Type: application/json" \
     http://localhost:8080/health
```text

## Obtaining Tokens

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
```text

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
curl -H "Authorization: Bearer YOUR_TOKEN" \
     {self.spec.get('servers', [{}])[0].get('url', 'http://localhost:8080')}/mcp/resources
```text
