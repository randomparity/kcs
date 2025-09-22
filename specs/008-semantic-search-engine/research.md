# Research: Semantic Search Engine Technical Components

## BAAI/bge-small-en-v1.5 Embedding Model

**Decision**: Use BAAI/bge-small-en-v1.5 as the primary embedding model

**Rationale**:

- Optimized for CPU-only operation with 133MB model size
- 384-dimensional embeddings provide good balance of accuracy and performance
- Excellent integration with sentence-transformers library
- Strong performance on code-related tasks and technical text
- ONNX optimization support for faster inference

**Alternatives considered**:

- all-MiniLM-L6-v2: Smaller but lower accuracy on technical content
- all-mpnet-base-v2: Better accuracy but 438MB size and slower inference
- OpenAI embeddings: API dependency and cost concerns

## pgvector with PostgreSQL

**Decision**: Use pgvector extension with PostgreSQL for vector storage

**Rationale**:

- Production-ready with proven scalability to 50M+ vectors
- Multiple indexing strategies (IVFFlat for recall, HNSW for speed)
- 28x better performance than some specialized vector databases
- Integrates with existing KCS PostgreSQL infrastructure
- Supports hybrid queries combining vector and metadata filtering

**Alternatives considered**:

- Chroma: Good for prototyping but less mature for production scale
- Qdrant: Excellent performance but adds infrastructure complexity
- Weaviate: Full-featured but overkill for our read-only use case

## Query Preprocessing Techniques

**Decision**: Multi-stage preprocessing pipeline

**Rationale**:

- Handle code-specific patterns (CamelCase, snake_case, technical abbreviations)
- Preserve semantic relationships while normalizing syntax variations
- Query expansion for kernel-specific terminology
- Maintain original query for fallback exact matching

**Approach**:

1. Tokenization preserving code identifiers
2. Technical abbreviation expansion (e.g., "mem" → "memory")
3. Case normalization while preserving semantic boundaries
4. Query enrichment with domain-specific synonyms

**Alternatives considered**:

- Simple lowercasing: Too aggressive, loses semantic information
- Stemming/lemmatization: Breaks technical terminology
- No preprocessing: Poor recall on variant terminology

## Result Ranking Algorithms

**Decision**: Hybrid ranking combining BM25 (30%) and semantic similarity (70%)

**Rationale**:

- BM25 provides exact keyword matching for precise technical terms
- Semantic similarity captures conceptual relationships
- Weighted combination optimizes for both precision and recall
- Confidence scoring based on distance metrics and contextual boosting

**Ranking Components**:

1. Semantic similarity score (cosine distance)
2. BM25 score for exact term matching
3. File type and location boosting (headers vs implementation)
4. Recency and modification frequency signals

**Alternatives considered**:

- Pure semantic ranking: Misses exact technical matches
- Pure keyword ranking: Poor conceptual understanding
- Equal weighting: Suboptimal for technical code search

## MCP Integration

**Decision**: Extend existing KCS MCP patterns for search endpoints

**Rationale**:

- Maintains constitutional MCP-first interface requirement
- Leverages existing authentication and error handling
- Consistent with current KCS architecture patterns
- Standard tool interfaces for AI agent integration

**Implementation**:

- Search tools following MCP 2025-09-20 specification
- OAuth 2.0 authentication integration
- Structured JSON responses with file citations
- Error handling with detailed context

**Alternatives considered**:

- Custom API: Breaks constitutional MCP-first requirement
- GraphQL: Added complexity without clear benefits
- REST-only: Insufficient for agent integration needs

## Integration Strategy

**Decision**: Enhance existing KCS components rather than standalone service

**Rationale**:

- Existing `/src/rust/kcs-search/` provides foundation
- Maintains architectural consistency
- Leverages existing database and MCP infrastructure
- Reduces operational complexity

**Key Integration Points**:

1. Replace hash-based embeddings with BGE model via Python bridge
2. Enhance pgvector implementation with optimized indexing
3. Extend query preprocessing in Rust components
4. Implement hybrid ranking in existing search pipeline
5. Extend MCP endpoints for semantic search capabilities

## Performance Considerations

**Target Metrics** (per constitution):

- Query response p95 ≤ 600ms
- Indexing throughput: 10k files/hour
- Concurrent users: 10+

**Optimization Strategies**:

- ONNX model optimization for CPU inference
- pgvector HNSW indexing for sub-linear search
- Query result caching with LRU eviction
- Batch embedding generation for indexing
- Connection pooling for database access

**Memory Requirements**:

- Model loading: ~200MB RAM
- Index size: ~50 bytes per document + vectors
- Query processing: ~10MB per concurrent request
