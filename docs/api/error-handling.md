# Error Handling

## Overview

KCS API uses standard HTTP status codes and provides detailed error information
to help diagnose and resolve issues.

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

## Error Handling Best Practices

### 1. Implement Retry Logic

```python
import time
import random

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff."""
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
```text

### 2. Handle Rate Limits

```python
def handle_rate_limit(response):
    """Handle rate limit responses."""
    if response.status_code == 429:
        reset_time = response.headers.get('X-RateLimit-Reset')
        if reset_time:
            wait_time = int(reset_time) - int(time.time())
            if wait_time > 0:
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return True
    return False
```text

### 3. Parse Error Details

```python
def parse_error_response(response):
    """Parse error response and extract details."""
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
```text

### 4. Comprehensive Error Handler

```python
class KCSError(Exception):
    """Base exception for KCS API errors."""
    def __init__(self, error_type, message, details=None, status_code=None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

class KCSClient:
    def _handle_response(self, response):
        """Handle API response and raise appropriate exceptions."""
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
```text

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
```text

### Log Request Context

```python
def log_request_context(func):
    """Decorator to log request context for debugging."""
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
```text

## Monitoring and Alerting

### Health Check Implementation

```python
def health_check():
    """Perform comprehensive health check."""
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
```text

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
```text

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
```text

## Support and Escalation

When encountering persistent errors:

1. **Check Status Page**: Monitor KCS service status
2. **Review Logs**: Check both client and server logs
3. **Test Connectivity**: Verify network path to server
4. **Validate Configuration**: Ensure correct URLs and tokens
5. **Contact Support**: Provide error details and request ID

### Error Report Template

```text
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
```text
