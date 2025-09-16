# Data Schemas

## Overview

KCS API uses structured data schemas for requests and responses. All schemas follow OpenAPI 3.0 specification.

## Core Concepts

### Citations and Spans

All KCS responses include citations with file:line references per constitutional requirements.

### Span

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `path` | string | ✓ | File path relative to repository root |
| `sha` | string | ✓ | Git SHA of file version |
| `start` | integer | ✓ | Starting line number |
| `end` | integer | ✓ | Ending line number |

**Example**:

```json
{
  "path": "fs/read_write.c",
  "sha": "a1b2c3d4e5f6",
  "start": 451
}
```text


### Citation

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `span` | [Span](#span) | ✓ | No description |
| `context` | string |  | Optional context around citation |


### SymbolInfo

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `name` | string | ✓ | No description |
| `kind` | string | ✓ | No description |
| `decl` | [Span](#span) | ✓ | No description |
| `summary` | object |  | Optional AI-generated summary |

**Example**:
```json
{
  "name": "sys_read",
  "kind": "example_kind"
}
```text


### SearchHit

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `span` | [Span](#span) | ✓ | No description |
| `snippet` | string | ✓ | Code snippet with match highlighted |
| `score` | number (float) |  | Relevance score |

**Example**:
```json
{
  "snippet": "example_snippet"
}
```text


### ImpactResult

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `configs` | array | ✓ | No description |
| `modules` | array | ✓ | No description |
| `tests` | array | ✓ | No description |
| `owners` | array | ✓ | No description |
| `risks` | array | ✓ | No description |
| `cites` | array | ✓ | No description |

**Example**:
```json
{
  "configs": [
    "x86_64:defconfig",
    "x86_64:allmodconfig"
  ],
  "modules": [
    "ext4",
    "btrfs"
  ],
  "tests": [
    "fs/ext4/tests/test_read.c"
  ]
}
```text


## Other Schemas


### CallerInfo

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `symbol` | string | ✓ | No description |
| `span` | [Span](#span) | ✓ | No description |
| `call_type` | string |  | No description |

**Example**:
```json
{
  "symbol": "example_symbol"
}
```text


### FlowStep

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `edge` | string | ✓ | Edge type |
| `from` | string | ✓ | Source symbol |
| `to` | string | ✓ | Target symbol |
| `span` | [Span](#span) | ✓ | No description |

**Example**:
```json
{
  "edge": "example_edge",
  "from": "example_from",
  "to": "example_to"
}
```text


### DriftMismatch

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `kind` | string | ✓ | No description |
| `detail` | string | ✓ | No description |
| `span` | [Span](#span) |  | No description |

**Example**:
```json
{
  "kind": "example_kind",
  "detail": "example_detail"
}
```text


### Error

No description available

**Type**: object

**Properties**:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `error` | string | ✓ | No description |
| `message` | string | ✓ | No description |

**Example**:
```json
{
  "error": "example_error",
  "message": "example_message"
}
```text
