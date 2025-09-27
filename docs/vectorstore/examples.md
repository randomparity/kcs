# VectorStore API Usage Examples

This document provides comprehensive usage examples for all VectorStore API methods, including basic usage, advanced parameters, error handling, and real-world use cases.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [API Methods](#api-methods)
   - [store_content()](#store_content)
   - [store_embedding()](#store_embedding)
   - [similarity_search()](#similarity_search)
   - [get_content_by_id()](#get_content_by_id)
   - [get_embedding_by_content_id()](#get_embedding_by_content_id)
   - [list_content()](#list_content)
   - [update_content_status()](#update_content_status)
   - [delete_content()](#delete_content)
   - [get_storage_stats()](#get_storage_stats)
4. [Common Error Patterns](#common-error-patterns)
5. [Best Practices](#best-practices)

## Overview

The VectorStore provides high-performance vector storage and similarity search operations using PostgreSQL with pgvector extension. All methods are async and require proper error handling.

## Setup

```python
from semantic_search.database.vector_store import VectorStore, ContentFilter, SimilaritySearchFilter

# Initialize vector store
vector_store = VectorStore()
```

## API Methods

### store_content()

Store content for indexing with optional metadata.

#### Basic Usage

```python
async def basic_store_content():
    """Store a simple source file."""
    try:
        content_id = await vector_store.store_content(
            content_type="source_file",
            source_path="/home/project/src/main.c",
            content="int main() { return 0; }"
        )
        print(f"Stored content with ID: {content_id}")
        return content_id
    except ValueError as e:
        print(f"Validation error: {e}")
    except RuntimeError as e:
        print(f"Storage error: {e}")
```

#### Advanced Usage

```python
async def advanced_store_content():
    """Store content with complete metadata."""
    metadata = {
        "language": "c",
        "functions": ["main"],
        "includes": ["stdio.h"],
        "complexity": "simple",
        "author": "developer",
        "version": "1.0"
    }

    try:
        content_id = await vector_store.store_content(
            content_type="source_file",
            source_path="/home/project/src/complex.c",
            content="""#include <stdio.h>

int calculate_factorial(int n) {
    if (n <= 1) return 1;
    return n * calculate_factorial(n - 1);
}

int main() {
    int result = calculate_factorial(5);
    printf("5! = %d\\n", result);
    return 0;
}""",
            title="Factorial Calculator",
            metadata=metadata
        )
        print(f"Stored complex content with ID: {content_id}")
        return content_id
    except Exception as e:
        print(f"Error storing content: {e}")
        raise
```

#### Real-world Use Case

```python
async def index_project_files(project_path: str):
    """Index all C files in a project directory."""
    import os
    from pathlib import Path

    results = []
    for file_path in Path(project_path).rglob("*.c"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Determine content type
            if file_path.name.endswith('.h'):
                content_type = "header_file"
            else:
                content_type = "source_file"

            # Extract metadata
            metadata = {
                "file_size": os.path.getsize(file_path),
                "last_modified": os.path.getmtime(file_path),
                "relative_path": str(file_path.relative_to(project_path))
            }

            content_id = await vector_store.store_content(
                content_type=content_type,
                source_path=str(file_path.absolute()),
                content=content,
                title=file_path.name,
                metadata=metadata
            )
            results.append((str(file_path), content_id))

        except Exception as e:
            print(f"Failed to index {file_path}: {e}")

    return results
```

#### Expected Output

```python
# Success
Stored content with ID: 42

# Error cases
Validation error: Content cannot be empty
Storage error: Failed to store content: duplicate key value violates unique constraint
```

---

### store_embedding()

Store vector embeddings for content chunks with model information.

#### Basic Usage

```python
async def basic_store_embedding():
    """Store a simple embedding for content."""
    # Assume we have a content_id from store_content()
    content_id = 42

    # Sample 384-dimensional embedding (normally from embedding model)
    sample_embedding = [0.1] * 384
    chunk_text = "int main() { return 0; }"

    try:
        embedding_id = await vector_store.store_embedding(
            content_id=content_id,
            embedding=sample_embedding,
            chunk_text=chunk_text
        )
        print(f"Stored embedding with ID: {embedding_id}")
        return embedding_id
    except ValueError as e:
        print(f"Validation error: {e}")
    except RuntimeError as e:
        print(f"Storage error: {e}")
```

#### Advanced Usage

```python
async def advanced_store_embedding():
    """Store embeddings with custom model and chunking."""
    content_id = 42

    # Multiple chunks with different embeddings
    chunks = [
        {
            "text": "#include <stdio.h>",
            "embedding": generate_embedding("#include <stdio.h>"),
            "index": 0
        },
        {
            "text": "int main() { printf(\"Hello\\n\"); return 0; }",
            "embedding": generate_embedding("int main() { printf(\"Hello\\n\"); return 0; }"),
            "index": 1
        }
    ]

    embedding_ids = []
    for chunk in chunks:
        try:
            embedding_id = await vector_store.store_embedding(
                content_id=content_id,
                embedding=chunk["embedding"],
                chunk_text=chunk["text"],
                chunk_index=chunk["index"],
                model_name="BAAI/bge-large-en-v1.5",
                model_version="1.5"
            )
            embedding_ids.append(embedding_id)
            print(f"Stored chunk {chunk['index']} with embedding ID: {embedding_id}")
        except Exception as e:
            print(f"Failed to store chunk {chunk['index']}: {e}")

    return embedding_ids

def generate_embedding(text: str) -> list[float]:
    """Mock function to generate embeddings."""
    # In real implementation, use actual embedding model
    import hashlib
    import random

    # Seed random with text for consistent results
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    return [random.random() for _ in range(384)]
```

#### Real-world Use Case

```python
async def process_large_file_chunks(content_id: int, file_path: str, chunk_size: int = 512):
    """Process large files in chunks and store embeddings."""
    from sentence_transformers import SentenceTransformer

    # Initialize embedding model
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    with open(file_path, 'r') as f:
        content = f.read()

    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i + chunk_size]
        if chunk_text.strip():  # Skip empty chunks
            chunks.append(chunk_text)

    print(f"Processing {len(chunks)} chunks for content ID {content_id}")

    embedding_ids = []
    for idx, chunk_text in enumerate(chunks):
        try:
            # Generate embedding
            embedding = model.encode(chunk_text).tolist()

            embedding_id = await vector_store.store_embedding(
                content_id=content_id,
                embedding=embedding,
                chunk_text=chunk_text,
                chunk_index=idx
            )

            embedding_ids.append(embedding_id)

            if idx % 10 == 0:  # Progress indicator
                print(f"Processed {idx + 1}/{len(chunks)} chunks")

        except Exception as e:
            print(f"Failed to process chunk {idx}: {e}")
            # Continue with next chunk
            continue

    print(f"Successfully stored {len(embedding_ids)} embeddings")
    return embedding_ids
```

#### Expected Output

```python
# Success
Stored embedding with ID: 123

# Error cases
Validation error: Expected 384 dimensions, got 256
Storage error: Failed to store embedding: content_id does not exist
```

---

### similarity_search()

Perform vector similarity search with flexible filtering options.

#### Basic Usage

```python
async def basic_similarity_search():
    """Basic similarity search."""
    # Query embedding (normally from embedding model)
    query_embedding = [0.2] * 384

    try:
        results = await vector_store.similarity_search(query_embedding)

        print(f"Found {len(results)} results")
        for result in results[:3]:  # Show top 3
            print(f"Content ID: {result['content_id']}")
            print(f"Similarity: {result['similarity_score']:.4f}")
            print(f"Source: {result['source_path']}")
            print("---")

        return results
    except ValueError as e:
        print(f"Validation error: {e}")
    except RuntimeError as e:
        print(f"Search error: {e}")
```

#### Advanced Usage

```python
async def advanced_similarity_search():
    """Advanced search with comprehensive filters."""
    from semantic_search.database.vector_store import SimilaritySearchFilter

    query_embedding = generate_embedding("malloc memory allocation")

    # Create advanced filter
    search_filter = SimilaritySearchFilter(
        similarity_threshold=0.7,  # Only high-confidence matches
        max_results=10,
        content_types=["source_file", "header_file"],
        file_paths=["/home/project/src/memory.c", "/home/project/src/utils.c"],
        include_content=True  # Include full content in results
    )

    try:
        results = await vector_store.similarity_search(
            query_embedding=query_embedding,
            filters=search_filter
        )

        print(f"Advanced search found {len(results)} results")
        for result in results:
            print(f"File: {result['source_path']}")
            print(f"Type: {result['content_type']}")
            print(f"Similarity: {result['similarity_score']:.4f}")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Chunk Index: {result['chunk_index']}")

            # Show metadata if available
            if result.get('metadata'):
                print(f"Metadata: {result['metadata']}")

            # Show content preview if included
            if result.get('content'):
                preview = result['content'][:200]
                print(f"Preview: {preview}...")

            print("=" * 50)

        return results
    except Exception as e:
        print(f"Search failed: {e}")
        return []
```

#### Real-world Use Case

```python
async def semantic_code_search(query: str, project_filter: str = None):
    """Semantic search across codebase with natural language queries."""
    from sentence_transformers import SentenceTransformer

    # Initialize embedding model
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Create context-aware filter
    search_filter = SimilaritySearchFilter(
        similarity_threshold=0.6,
        max_results=20,
        include_content=True
    )

    # Add project-specific filtering
    if project_filter:
        search_filter.path_patterns = [f"*{project_filter}*"]

    try:
        results = await vector_store.similarity_search(
            query_embedding=query_embedding,
            filters=search_filter
        )

        # Group results by file for better presentation
        files_results = {}
        for result in results:
            file_path = result['source_path']
            if file_path not in files_results:
                files_results[file_path] = []
            files_results[file_path].append(result)

        # Present results
        print(f"Query: '{query}'")
        print(f"Found matches in {len(files_results)} files:")
        print("=" * 60)

        for file_path, file_results in files_results.items():
            print(f"\n📁 {file_path}")
            print(f"   Found {len(file_results)} relevant chunks")

            # Show best match for this file
            best_match = max(file_results, key=lambda x: x['similarity_score'])
            print(f"   Best match (similarity: {best_match['similarity_score']:.3f}):")

            # Show content preview
            if best_match.get('content'):
                lines = best_match['content'].split('\n')
                for i, line in enumerate(lines[:5]):  # Show first 5 lines
                    print(f"   {i+1}: {line}")
                if len(lines) > 5:
                    print("   ...")

        return results

    except Exception as e:
        print(f"Semantic search failed: {e}")
        return []

# Example usage
async def search_examples():
    """Examples of semantic searches."""
    queries = [
        "memory allocation and deallocation",
        "error handling and return codes",
        "string manipulation functions",
        "file I/O operations",
        "mathematical calculations"
    ]

    for query in queries:
        print(f"\nSearching for: {query}")
        await semantic_code_search(query)
        print("\n" + "="*80)
```

#### Expected Output

```python
# Basic search results
Found 5 results
Content ID: 42
Similarity: 0.8756
Source: /home/project/src/main.c
---

# Advanced search with filters
Advanced search found 3 results
File: /home/project/src/memory.c
Type: source_file
Similarity: 0.9234
Title: Memory Management
Chunk Index: 2
Metadata: {'functions': ['malloc', 'free'], 'complexity': 'medium'}
Preview: void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}...
```

---

### get_content_by_id()

Retrieve specific content by its unique identifier.

#### Basic Usage

```python
async def basic_get_content():
    """Retrieve content by ID."""
    content_id = 42

    try:
        content = await vector_store.get_content_by_id(content_id)

        if content:
            print(f"Found content: {content.source_path}")
            print(f"Type: {content.content_type}")
            print(f"Status: {content.status}")
            print(f"Created: {content.created_at}")
            return content
        else:
            print(f"No content found with ID {content_id}")
            return None

    except Exception as e:
        print(f"Error retrieving content: {e}")
        return None
```

#### Advanced Usage

```python
async def advanced_get_content():
    """Retrieve and analyze content with full details."""
    content_id = 42

    try:
        content = await vector_store.get_content_by_id(content_id)

        if not content:
            print(f"Content ID {content_id} not found")
            return None

        # Display comprehensive information
        print("=" * 60)
        print(f"📄 Content Details (ID: {content.id})")
        print("=" * 60)
        print(f"Source Path: {content.source_path}")
        print(f"Content Type: {content.content_type}")
        print(f"Title: {content.title or 'N/A'}")
        print(f"Content Hash: {content.content_hash}")
        print(f"Status: {content.status}")
        print(f"Created: {content.created_at}")
        print(f"Updated: {content.updated_at}")
        print(f"Indexed: {content.indexed_at or 'Not indexed'}")

        # Show metadata
        if content.metadata:
            print(f"Metadata:")
            for key, value in content.metadata.items():
                print(f"  {key}: {value}")
        else:
            print("Metadata: None")

        # Show content preview
        if content.content:
            print(f"\nContent Preview (first 300 chars):")
            print("-" * 40)
            print(content.content[:300])
            if len(content.content) > 300:
                print("...")
            print("-" * 40)
            print(f"Total content length: {len(content.content)} characters")

        return content

    except Exception as e:
        print(f"Failed to retrieve content {content_id}: {e}")
        return None
```

#### Real-world Use Case

```python
async def content_analysis_pipeline(content_ids: list[int]):
    """Analyze multiple content items for processing status."""
    results = {
        'found': [],
        'missing': [],
        'by_status': {},
        'by_type': {}
    }

    for content_id in content_ids:
        try:
            content = await vector_store.get_content_by_id(content_id)

            if content:
                results['found'].append(content)

                # Group by status
                status = content.status
                if status not in results['by_status']:
                    results['by_status'][status] = []
                results['by_status'][status].append(content)

                # Group by type
                content_type = content.content_type
                if content_type not in results['by_type']:
                    results['by_type'][content_type] = []
                results['by_type'][content_type].append(content)

            else:
                results['missing'].append(content_id)

        except Exception as e:
            print(f"Error processing content ID {content_id}: {e}")
            results['missing'].append(content_id)

    # Generate report
    print(f"Content Analysis Report")
    print(f"=" * 50)
    print(f"Total requested: {len(content_ids)}")
    print(f"Found: {len(results['found'])}")
    print(f"Missing: {len(results['missing'])}")

    if results['missing']:
        print(f"Missing IDs: {results['missing']}")

    print(f"\nBy Status:")
    for status, items in results['by_status'].items():
        print(f"  {status}: {len(items)}")

    print(f"\nBy Type:")
    for content_type, items in results['by_type'].items():
        print(f"  {content_type}: {len(items)}")

    return results

async def validate_content_integrity(content_id: int):
    """Validate content integrity and relationships."""
    content = await vector_store.get_content_by_id(content_id)

    if not content:
        return {"valid": False, "error": "Content not found"}

    issues = []

    # Check file exists
    import os
    if not os.path.exists(content.source_path):
        issues.append(f"Source file does not exist: {content.source_path}")

    # Check content hash if file exists
    if os.path.exists(content.source_path):
        import hashlib
        with open(content.source_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()

        current_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
        if current_hash != content.content_hash:
            issues.append("Content hash mismatch - file may have been modified")

    # Check for orphaned content (no embeddings)
    # This would require checking embeddings table
    # For now, just placeholder
    if content.status == 'COMPLETED':
        # In real implementation, check if embeddings exist
        pass

    return {
        "valid": len(issues) == 0,
        "content_id": content_id,
        "issues": issues,
        "content": content
    }
```

#### Expected Output

```python
# Found content
Found content: /home/project/src/main.c
Type: source_file
Status: COMPLETED
Created: 2024-09-25 10:30:45

# Content not found
No content found with ID 999

# Detailed analysis
==============================================================
📄 Content Details (ID: 42)
==============================================================
Source Path: /home/project/src/main.c
Content Type: source_file
Title: Main Program Entry Point
Content Hash: a1b2c3d4e5f6...
Status: COMPLETED
Created: 2024-09-25 10:30:45.123456
Updated: 2024-09-25 10:35:12.654321
Indexed: 2024-09-25 10:35:15.789012
Metadata:
  language: c
  functions: ['main']
  complexity: simple

Content Preview (first 300 chars):
----------------------------------------
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
----------------------------------------
Total content length: 67 characters
```

---

### get_embedding_by_content_id()

Retrieve vector embeddings for specific content and chunk indices.

#### Basic Usage

```python
async def basic_get_embedding():
    """Retrieve embedding for content."""
    content_id = 42

    try:
        embedding = await vector_store.get_embedding_by_content_id(content_id)

        if embedding:
            print(f"Found embedding for content {content_id}")
            print(f"Embedding ID: {embedding.id}")
            print(f"Chunk index: {embedding.chunk_index}")
            print(f"Vector dimensions: {len(embedding.embedding)}")
            print(f"Created: {embedding.created_at}")
            return embedding
        else:
            print(f"No embedding found for content ID {content_id}")
            return None

    except Exception as e:
        print(f"Error retrieving embedding: {e}")
        return None
```

#### Advanced Usage

```python
async def advanced_get_embedding():
    """Retrieve and analyze embedding with specific chunk."""
    content_id = 42
    chunk_index = 2  # Get specific chunk

    try:
        embedding = await vector_store.get_embedding_by_content_id(
            content_id=content_id,
            chunk_index=chunk_index
        )

        if not embedding:
            print(f"No embedding found for content {content_id}, chunk {chunk_index}")
            return None

        # Analyze embedding
        print(f"🔍 Embedding Analysis")
        print(f"=" * 50)
        print(f"Content ID: {embedding.content_id}")
        print(f"Embedding ID: {embedding.id}")
        print(f"Chunk Index: {embedding.chunk_index}")
        print(f"Created: {embedding.created_at}")

        # Analyze vector properties
        import numpy as np
        vec = np.array(embedding.embedding)

        print(f"\n📊 Vector Statistics:")
        print(f"Dimensions: {len(embedding.embedding)}")
        print(f"Min value: {vec.min():.6f}")
        print(f"Max value: {vec.max():.6f}")
        print(f"Mean: {vec.mean():.6f}")
        print(f"Std dev: {vec.std():.6f}")
        print(f"L2 norm: {np.linalg.norm(vec):.6f}")

        # Show first few dimensions
        print(f"\nFirst 10 dimensions:")
        for i, val in enumerate(embedding.embedding[:10]):
            print(f"  [{i}]: {val:.6f}")

        return embedding

    except Exception as e:
        print(f"Failed to retrieve embedding: {e}")
        return None
```

#### Real-world Use Case

```python
async def analyze_content_embeddings(content_id: int):
    """Analyze all embeddings for a piece of content."""
    # First get the content to understand context
    content = await vector_store.get_content_by_id(content_id)
    if not content:
        print(f"Content {content_id} not found")
        return None

    print(f"Analyzing embeddings for: {content.source_path}")
    print(f"Content status: {content.status}")

    # Try to get embeddings for different chunk indices
    # Start with chunk 0 and keep going until we find no more
    embeddings = []
    chunk_index = 0

    while True:
        try:
            embedding = await vector_store.get_embedding_by_content_id(
                content_id=content_id,
                chunk_index=chunk_index
            )

            if embedding:
                embeddings.append(embedding)
                chunk_index += 1
            else:
                break  # No more chunks

        except Exception as e:
            print(f"Error retrieving chunk {chunk_index}: {e}")
            break

    if not embeddings:
        print("No embeddings found for this content")
        return []

    print(f"\nFound {len(embeddings)} embedding chunks")

    # Analyze embedding patterns
    import numpy as np

    all_vectors = [emb.embedding for emb in embeddings]
    vectors_array = np.array(all_vectors)

    print(f"\n📊 Embedding Collection Analysis:")
    print(f"Total chunks: {len(embeddings)}")
    print(f"Vector dimensions: {vectors_array.shape[1]}")
    print(f"Average L2 norm: {np.linalg.norm(vectors_array, axis=1).mean():.6f}")

    # Calculate inter-chunk similarities
    if len(embeddings) > 1:
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                vec1 = np.array(embeddings[i].embedding)
                vec2 = np.array(embeddings[j].embedding)

                # Cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities.append(similarity)

        if similarities:
            print(f"Inter-chunk similarity stats:")
            print(f"  Mean: {np.mean(similarities):.4f}")
            print(f"  Min:  {np.min(similarities):.4f}")
            print(f"  Max:  {np.max(similarities):.4f}")

    return embeddings

async def compare_embeddings(content_id1: int, content_id2: int, chunk_index: int = 0):
    """Compare embeddings between two pieces of content."""
    try:
        emb1 = await vector_store.get_embedding_by_content_id(content_id1, chunk_index)
        emb2 = await vector_store.get_embedding_by_content_id(content_id2, chunk_index)

        if not emb1:
            print(f"No embedding found for content {content_id1}")
            return None

        if not emb2:
            print(f"No embedding found for content {content_id2}")
            return None

        # Calculate similarity
        import numpy as np
        vec1 = np.array(emb1.embedding)
        vec2 = np.array(emb2.embedding)

        # Cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Euclidean distance
        euclidean_dist = np.linalg.norm(vec1 - vec2)

        print(f"🔄 Embedding Comparison")
        print(f"Content {content_id1} vs Content {content_id2}")
        print(f"Chunk index: {chunk_index}")
        print(f"Cosine similarity: {cosine_sim:.6f}")
        print(f"Euclidean distance: {euclidean_dist:.6f}")

        # Interpretation
        if cosine_sim > 0.9:
            print("📈 Very high similarity - likely very similar content")
        elif cosine_sim > 0.7:
            print("📊 High similarity - related content")
        elif cosine_sim > 0.5:
            print("📉 Moderate similarity - some relationship")
        else:
            print("📋 Low similarity - different content")

        return {
            'content_id1': content_id1,
            'content_id2': content_id2,
            'chunk_index': chunk_index,
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'embedding1': emb1,
            'embedding2': emb2
        }

    except Exception as e:
        print(f"Error comparing embeddings: {e}")
        return None
```

#### Expected Output

```python
# Basic retrieval
Found embedding for content 42
Embedding ID: 123
Chunk index: 0
Vector dimensions: 384
Created: 2024-09-25 10:35:15

# Advanced analysis
🔍 Embedding Analysis
==================================================
Content ID: 42
Embedding ID: 123
Chunk Index: 2
Created: 2024-09-25 10:35:15.789012

📊 Vector Statistics:
Dimensions: 384
Min value: -0.892341
Max value: 0.934521
Mean: -0.001234
Std dev: 0.234567
L2 norm: 2.456789

First 10 dimensions:
  [0]: 0.123456
  [1]: -0.234567
  [2]: 0.345678
  ...
```

---

### list_content()

List content with flexible filtering and pagination.

#### Basic Usage

```python
async def basic_list_content():
    """List all content with default filters."""
    try:
        content_list = await vector_store.list_content()

        print(f"Found {len(content_list)} content items")
        for content in content_list[:5]:  # Show first 5
            print(f"ID: {content.id} | {content.source_path} | {content.status}")

        return content_list
    except RuntimeError as e:
        print(f"Error listing content: {e}")
        return []
```

#### Advanced Usage

```python
async def advanced_list_content():
    """List content with comprehensive filters."""
    from semantic_search.database.vector_store import ContentFilter

    # Create advanced filter
    content_filter = ContentFilter(
        content_types=["source_file", "header_file"],
        path_patterns=["*.c", "*/src/*"],
        status_filter=["COMPLETED", "PENDING"],
        max_results=50
    )

    try:
        content_list = await vector_store.list_content(filters=content_filter)

        print(f"Advanced filter found {len(content_list)} items")
        print("=" * 80)

        # Group by status for analysis
        by_status = {}
        by_type = {}

        for content in content_list:
            # Group by status
            status = content.status
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(content)

            # Group by type
            content_type = content.content_type
            if content_type not in by_type:
                by_type[content_type] = []
            by_type[content_type].append(content)

        print("📊 Results Summary:")
        print(f"By Status:")
        for status, items in by_status.items():
            print(f"  {status}: {len(items)}")

        print(f"By Type:")
        for content_type, items in by_type.items():
            print(f"  {content_type}: {len(items)}")

        # Show detailed results
        print(f"\n📄 Detailed Results:")
        for content in content_list:
            print(f"ID: {content.id:4d} | {content.status:10s} | {content.content_type:12s} | {content.source_path}")
            if content.title:
                print(f"      Title: {content.title}")
            if content.metadata:
                print(f"      Metadata: {dict(list(content.metadata.items())[:3])}...")  # Show first 3 metadata items

        return content_list

    except Exception as e:
        print(f"Advanced listing failed: {e}")
        return []
```

#### Real-world Use Case

```python
async def project_status_report():
    """Generate comprehensive project status report."""
    from semantic_search.database.vector_store import ContentFilter

    # Get all content
    all_content = await vector_store.list_content(
        filters=ContentFilter(max_results=1000)
    )

    if not all_content:
        print("No content found in database")
        return

    print("🚀 Project Indexing Status Report")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total files indexed: {len(all_content)}")

    # Analyze by status
    status_counts = {}
    failed_items = []

    for content in all_content:
        status = content.status
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == 'FAILED':
            failed_items.append(content)

    print(f"\n📊 Status Distribution:")
    for status, count in sorted(status_counts.items()):
        percentage = (count / len(all_content)) * 100
        print(f"  {status:12s}: {count:4d} ({percentage:5.1f}%)")

    # Analyze by file type
    type_counts = {}
    for content in all_content:
        content_type = content.content_type
        type_counts[content_type] = type_counts.get(content_type, 0) + 1

    print(f"\n📁 File Type Distribution:")
    for content_type, count in sorted(type_counts.items()):
        percentage = (count / len(all_content)) * 100
        print(f"  {content_type:15s}: {count:4d} ({percentage:5.1f}%)")

    # Show recent activity
    recent_content = sorted(all_content, key=lambda x: x.updated_at, reverse=True)[:10]
    print(f"\n🕒 Recent Activity (Last 10 updates):")
    for content in recent_content:
        print(f"  {content.updated_at.strftime('%Y-%m-%d %H:%M')} | {content.status:10s} | {content.source_path}")

    # Show failed items details
    if failed_items:
        print(f"\n❌ Failed Items ({len(failed_items)}):")
        for content in failed_items[:5]:  # Show first 5 failures
            print(f"  {content.source_path}")
            print(f"    Error: {getattr(content, 'error_message', 'No error message')}")
        if len(failed_items) > 5:
            print(f"  ... and {len(failed_items) - 5} more")

    # Project health score
    completed = status_counts.get('COMPLETED', 0)
    health_score = (completed / len(all_content)) * 100

    print(f"\n💚 Project Health Score: {health_score:.1f}%")
    if health_score > 90:
        print("   ✅ Excellent - Most files are properly indexed")
    elif health_score > 70:
        print("   ⚠️  Good - Some files need attention")
    else:
        print("   ❌ Poor - Many files failed or are pending")

async def find_stale_content(days_threshold: int = 7):
    """Find content that hasn't been updated recently."""
    from datetime import datetime, timedelta
    from semantic_search.database.vector_store import ContentFilter

    # Get all content
    all_content = await vector_store.list_content(
        filters=ContentFilter(max_results=1000)
    )

    threshold_date = datetime.now() - timedelta(days=days_threshold)
    stale_items = []

    for content in all_content:
        if content.updated_at < threshold_date:
            stale_items.append(content)

    print(f"📅 Stale Content Report (older than {days_threshold} days)")
    print("=" * 60)
    print(f"Found {len(stale_items)} stale items out of {len(all_content)} total")

    if stale_items:
        # Sort by oldest first
        stale_items.sort(key=lambda x: x.updated_at)

        print(f"\nOldest items:")
        for content in stale_items[:10]:
            days_old = (datetime.now() - content.updated_at).days
            print(f"  {days_old:3d} days | {content.status:10s} | {content.source_path}")

    return stale_items
```

#### Expected Output

```python
# Basic listing
Found 25 content items
ID:   1 | /home/project/src/main.c | COMPLETED
ID:   2 | /home/project/src/utils.c | COMPLETED
ID:   3 | /home/project/include/utils.h | PENDING
ID:   4 | /home/project/src/parser.c | FAILED
ID:   5 | /home/project/docs/README.md | COMPLETED

# Advanced filtering
Advanced filter found 18 items
================================================================================
📊 Results Summary:
By Status:
  COMPLETED: 15
  PENDING: 2
  FAILED: 1

By Type:
  source_file: 12
  header_file: 6

📄 Detailed Results:
ID:   1 | COMPLETED  | source_file  | /home/project/src/main.c
      Title: Main Program Entry Point
      Metadata: {'language': 'c', 'functions': ['main'], 'complexity': 'simple'}...
```

---

### update_content_status()

Update the processing status of content items.

#### Basic Usage

```python
async def basic_update_status():
    """Update content status."""
    content_id = 42
    new_status = "COMPLETED"

    try:
        success = await vector_store.update_content_status(
            content_id=content_id,
            status=new_status
        )

        if success:
            print(f"Successfully updated content {content_id} to {new_status}")
        else:
            print(f"Failed to update content {content_id} - may not exist")

        return success
    except Exception as e:
        print(f"Error updating status: {e}")
        return False
```

#### Advanced Usage

```python
async def advanced_update_status():
    """Update status with timestamp and validation."""
    from datetime import datetime

    content_id = 42
    new_status = "COMPLETED"
    indexed_timestamp = datetime.now()

    # First verify content exists
    content = await vector_store.get_content_by_id(content_id)
    if not content:
        print(f"Content {content_id} not found")
        return False

    print(f"Updating content {content_id}:")
    print(f"  Current status: {content.status}")
    print(f"  New status: {new_status}")
    print(f"  Indexed at: {indexed_timestamp}")

    try:
        success = await vector_store.update_content_status(
            content_id=content_id,
            status=new_status,
            indexed_at=indexed_timestamp
        )

        if success:
            print(f"✅ Status update successful")

            # Verify the update
            updated_content = await vector_store.get_content_by_id(content_id)
            if updated_content:
                print(f"Verification:")
                print(f"  Status: {updated_content.status}")
                print(f"  Indexed at: {updated_content.indexed_at}")
                print(f"  Updated at: {updated_content.updated_at}")
        else:
            print(f"❌ Status update failed")

        return success

    except Exception as e:
        print(f"Error during status update: {e}")
        return False
```

#### Real-world Use Case

```python
async def batch_status_update(content_updates: list[dict]):
    """Update multiple content statuses in batch."""
    """
    content_updates format:
    [
        {"content_id": 1, "status": "COMPLETED"},
        {"content_id": 2, "status": "FAILED", "indexed_at": datetime.now()},
        ...
    ]
    """
    results = {
        'success': [],
        'failed': [],
        'not_found': []
    }

    print(f"Processing batch status update for {len(content_updates)} items")

    for i, update in enumerate(content_updates):
        content_id = update['content_id']
        status = update['status']
        indexed_at = update.get('indexed_at')

        try:
            # Verify content exists first
            content = await vector_store.get_content_by_id(content_id)
            if not content:
                results['not_found'].append(content_id)
                print(f"[{i+1}/{len(content_updates)}] Content {content_id} not found")
                continue

            # Update status
            success = await vector_store.update_content_status(
                content_id=content_id,
                status=status,
                indexed_at=indexed_at
            )

            if success:
                results['success'].append(content_id)
                print(f"[{i+1}/{len(content_updates)}] ✅ Updated {content_id} to {status}")
            else:
                results['failed'].append(content_id)
                print(f"[{i+1}/{len(content_updates)}] ❌ Failed to update {content_id}")

        except Exception as e:
            results['failed'].append(content_id)
            print(f"[{i+1}/{len(content_updates)}] ❌ Error updating {content_id}: {e}")

    # Summary report
    print(f"\n📊 Batch Update Summary:")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Not found: {len(results['not_found'])}")

    if results['failed']:
        print(f"Failed IDs: {results['failed']}")
    if results['not_found']:
        print(f"Not found IDs: {results['not_found']}")

    return results

async def process_failed_content():
    """Identify and reprocess failed content."""
    from semantic_search.database.vector_store import ContentFilter

    # Get all failed content
    failed_filter = ContentFilter(
        status_filter=["FAILED"],
        max_results=100
    )

    failed_content = await vector_store.list_content(filters=failed_filter)

    if not failed_content:
        print("No failed content found")
        return

    print(f"Found {len(failed_content)} failed content items")
    print("Attempting to reset status to PENDING for reprocessing...")

    reset_updates = []
    for content in failed_content:
        reset_updates.append({
            'content_id': content.id,
            'status': 'PENDING'
        })

    # Batch update to reset status
    results = await batch_status_update(reset_updates)

    if results['success']:
        print(f"✅ Reset {len(results['success'])} items for reprocessing")

    return results

async def mark_stale_as_pending(days_threshold: int = 30):
    """Mark stale content as pending for re-indexing."""
    from datetime import datetime, timedelta
    from semantic_search.database.vector_store import ContentFilter

    # Get all completed content
    completed_filter = ContentFilter(
        status_filter=["COMPLETED"],
        max_results=1000
    )

    completed_content = await vector_store.list_content(filters=completed_filter)
    threshold_date = datetime.now() - timedelta(days=days_threshold)

    stale_items = []
    for content in completed_content:
        if content.indexed_at and content.indexed_at < threshold_date:
            stale_items.append(content)

    if not stale_items:
        print(f"No stale content found (older than {days_threshold} days)")
        return

    print(f"Found {len(stale_items)} stale content items")
    print("Marking as PENDING for re-indexing...")

    updates = []
    for content in stale_items:
        updates.append({
            'content_id': content.id,
            'status': 'PENDING'
        })

    results = await batch_status_update(updates)
    return results
```

#### Expected Output

```python
# Basic update
Successfully updated content 42 to COMPLETED

# Advanced update
Updating content 42:
  Current status: PENDING
  New status: COMPLETED
  Indexed at: 2024-09-25 15:30:45.123456
✅ Status update successful
Verification:
  Status: COMPLETED
  Indexed at: 2024-09-25 15:30:45.123456
  Updated at: 2024-09-25 15:30:45.234567

# Batch update
Processing batch status update for 5 items
[1/5] ✅ Updated 1 to COMPLETED
[2/5] ✅ Updated 2 to COMPLETED
[3/5] ❌ Content 3 not found
[4/5] ❌ Failed to update 4
[5/5] ✅ Updated 5 to PENDING

📊 Batch Update Summary:
Successful: 3
Failed: 1
Not found: 1
Failed IDs: [4]
Not found IDs: [3]
```

---

### delete_content()

Delete content and all associated embeddings.

#### Basic Usage

```python
async def basic_delete_content():
    """Delete content by ID."""
    content_id = 42

    try:
        success = await vector_store.delete_content(content_id)

        if success:
            print(f"Successfully deleted content {content_id}")
        else:
            print(f"Failed to delete content {content_id} - may not exist")

        return success
    except Exception as e:
        print(f"Error deleting content: {e}")
        return False
```

#### Advanced Usage

```python
async def advanced_delete_content():
    """Delete content with comprehensive verification."""
    content_id = 42

    # First get content details for verification
    try:
        content = await vector_store.get_content_by_id(content_id)

        if not content:
            print(f"Content {content_id} not found")
            return False

        print(f"🗑️  Preparing to delete content {content_id}")
        print(f"Source: {content.source_path}")
        print(f"Type: {content.content_type}")
        print(f"Status: {content.status}")
        print(f"Created: {content.created_at}")

        # Check for associated embeddings
        embedding = await vector_store.get_embedding_by_content_id(content_id)
        has_embeddings = embedding is not None

        if has_embeddings:
            print(f"⚠️  Content has associated embeddings that will also be deleted")

        # Confirm deletion (in real app, might prompt user)
        print(f"Proceeding with deletion...")

        success = await vector_store.delete_content(content_id)

        if success:
            print(f"✅ Successfully deleted content {content_id}")

            # Verify deletion
            deleted_content = await vector_store.get_content_by_id(content_id)
            deleted_embedding = await vector_store.get_embedding_by_content_id(content_id)

            if not deleted_content and not deleted_embedding:
                print(f"✅ Deletion verified - content and embeddings removed")
            else:
                print(f"⚠️  Deletion incomplete - some data may remain")
        else:
            print(f"❌ Failed to delete content {content_id}")

        return success

    except Exception as e:
        print(f"Error during deletion process: {e}")
        return False
```

#### Real-world Use Case

```python
async def cleanup_failed_content(confirm: bool = False):
    """Clean up all failed content items."""
    from semantic_search.database.vector_store import ContentFilter

    # Get all failed content
    failed_filter = ContentFilter(
        status_filter=["FAILED"],
        max_results=1000
    )

    failed_content = await vector_store.list_content(filters=failed_filter)

    if not failed_content:
        print("No failed content found")
        return {'deleted': 0, 'failed': 0}

    print(f"🧹 Cleanup Process - Found {len(failed_content)} failed items")

    if not confirm:
        print("⚠️  This is a dry run. Set confirm=True to actually delete.")
        for content in failed_content[:10]:  # Show first 10
            print(f"  Would delete: {content.source_path} (ID: {content.id})")
        if len(failed_content) > 10:
            print(f"  ... and {len(failed_content) - 10} more")
        return {'would_delete': len(failed_content)}

    # Actually delete items
    results = {'deleted': 0, 'failed': 0}

    for i, content in enumerate(failed_content):
        try:
            success = await vector_store.delete_content(content.id)
            if success:
                results['deleted'] += 1
                print(f"[{i+1}/{len(failed_content)}] ✅ Deleted {content.source_path}")
            else:
                results['failed'] += 1
                print(f"[{i+1}/{len(failed_content)}] ❌ Failed to delete {content.source_path}")
        except Exception as e:
            results['failed'] += 1
            print(f"[{i+1}/{len(failed_content)}] ❌ Error deleting {content.source_path}: {e}")

    print(f"\n📊 Cleanup Summary:")
    print(f"Successfully deleted: {results['deleted']}")
    print(f"Failed to delete: {results['failed']}")

    return results

async def delete_by_path_pattern(pattern: str, confirm: bool = False):
    """Delete content matching a path pattern."""
    from semantic_search.database.vector_store import ContentFilter

    # Get content matching pattern
    filter_config = ContentFilter(
        path_patterns=[pattern],
        max_results=1000
    )

    matching_content = await vector_store.list_content(filters=filter_config)

    if not matching_content:
        print(f"No content found matching pattern: {pattern}")
        return {'deleted': 0, 'failed': 0}

    print(f"🎯 Found {len(matching_content)} items matching pattern: {pattern}")

    if not confirm:
        print("⚠️  This is a dry run. Set confirm=True to actually delete.")
        for content in matching_content:
            print(f"  Would delete: {content.source_path} (ID: {content.id}, Status: {content.status})")
        return {'would_delete': len(matching_content)}

    # Delete matching items
    results = {'deleted': 0, 'failed': 0}

    for content in matching_content:
        try:
            success = await vector_store.delete_content(content.id)
            if success:
                results['deleted'] += 1
                print(f"✅ Deleted {content.source_path}")
            else:
                results['failed'] += 1
                print(f"❌ Failed to delete {content.source_path}")
        except Exception as e:
            results['failed'] += 1
            print(f"❌ Error deleting {content.source_path}: {e}")

    return results

async def safe_delete_with_backup(content_id: int):
    """Delete content with backup information."""
    # Get content details first
    content = await vector_store.get_content_by_id(content_id)
    if not content:
        print(f"Content {content_id} not found")
        return False

    # Create backup record
    backup_info = {
        'content_id': content.id,
        'source_path': content.source_path,
        'content_type': content.content_type,
        'content_hash': content.content_hash,
        'title': content.title,
        'metadata': content.metadata,
        'status': content.status,
        'created_at': content.created_at,
        'updated_at': content.updated_at,
        'indexed_at': content.indexed_at,
        'deleted_at': datetime.now()
    }

    # Save backup (in real implementation, might save to file or separate table)
    import json
    backup_filename = f"content_backup_{content_id}_{int(datetime.now().timestamp())}.json"

    print(f"💾 Creating backup: {backup_filename}")
    with open(backup_filename, 'w') as f:
        json.dump(backup_info, f, indent=2, default=str)

    # Proceed with deletion
    try:
        success = await vector_store.delete_content(content_id)

        if success:
            print(f"✅ Content {content_id} deleted successfully")
            print(f"💾 Backup saved to: {backup_filename}")
        else:
            print(f"❌ Failed to delete content {content_id}")
            # Remove backup file if deletion failed
            import os
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
                print(f"🗑️  Removed backup file (deletion failed)")

        return success

    except Exception as e:
        print(f"Error during safe deletion: {e}")
        return False
```

#### Expected Output

```python
# Basic deletion
Successfully deleted content 42

# Advanced deletion with verification
🗑️  Preparing to delete content 42
Source: /home/project/src/old_code.c
Type: source_file
Status: FAILED
Created: 2024-09-20 10:30:45
⚠️  Content has associated embeddings that will also be deleted
Proceeding with deletion...
✅ Successfully deleted content 42
✅ Deletion verified - content and embeddings removed

# Cleanup failed content (dry run)
🧹 Cleanup Process - Found 5 failed items
⚠️  This is a dry run. Set confirm=True to actually delete.
  Would delete: /home/project/src/broken.c (ID: 10)
  Would delete: /home/project/src/corrupt.c (ID: 15)
  Would delete: /home/project/include/missing.h (ID: 23)
  Would delete: /home/project/docs/invalid.md (ID: 31)
  Would delete: /home/project/test/error.c (ID: 38)

# Actual cleanup
📊 Cleanup Summary:
Successfully deleted: 4
Failed to delete: 1
```

---

### get_storage_stats()

Get comprehensive statistics about vector storage.

#### Basic Usage

```python
async def basic_get_stats():
    """Get basic storage statistics."""
    try:
        stats = await vector_store.get_storage_stats()

        if 'error' in stats:
            print(f"Error retrieving stats: {stats['error']}")
            return None

        print(f"📊 Storage Statistics:")
        print(f"Total content: {stats.get('total_content', 0)}")
        print(f"Total embeddings: {stats.get('total_embeddings', 0)}")

        return stats
    except Exception as e:
        print(f"Error getting storage stats: {e}")
        return None
```

#### Advanced Usage

```python
async def advanced_get_stats():
    """Get comprehensive storage analysis."""
    try:
        stats = await vector_store.get_storage_stats()

        if 'error' in stats:
            print(f"❌ Error retrieving stats: {stats['error']}")
            return None

        print("=" * 70)
        print("🗄️  VECTOR STORE STATISTICS REPORT")
        print("=" * 70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Overall totals
        total_content = stats.get('total_content', 0)
        total_embeddings = stats.get('total_embeddings', 0)

        print(f"\n📈 Overall Statistics:")
        print(f"Total Content Items: {total_content:,}")
        print(f"Total Embeddings: {total_embeddings:,}")

        if total_content > 0:
            embeddings_per_content = total_embeddings / total_content
            print(f"Average Embeddings per Content: {embeddings_per_content:.2f}")

        # Content by type and status
        content_by_type_status = stats.get('content_by_type_status', {})

        if content_by_type_status:
            print(f"\n📋 Content Distribution:")

            # Calculate totals by type and status
            type_totals = {}
            status_totals = {}

            for content_type, status_counts in content_by_type_status.items():
                type_total = sum(status_counts.values())
                type_totals[content_type] = type_total

                for status, count in status_counts.items():
                    status_totals[status] = status_totals.get(status, 0) + count

            # Show by content type
            print(f"\n  By Content Type:")
            for content_type, total in sorted(type_totals.items()):
                percentage = (total / total_content) * 100 if total_content > 0 else 0
                print(f"    {content_type:15s}: {total:5d} ({percentage:5.1f}%)")

                # Show status breakdown for this type
                for status, count in sorted(content_by_type_status[content_type].items()):
                    status_percentage = (count / total) * 100 if total > 0 else 0
                    print(f"      └─ {status:10s}: {count:4d} ({status_percentage:4.0f}%)")

            # Show by status
            print(f"\n  By Processing Status:")
            for status, total in sorted(status_totals.items()):
                percentage = (total / total_content) * 100 if total_content > 0 else 0
                print(f"    {status:12s}: {total:5d} ({percentage:5.1f}%)")

        # Embedding models
        embedding_models = stats.get('embedding_models', [])

        if embedding_models:
            print(f"\n🤖 Embedding Models:")
            for model_info in embedding_models:
                model_name = model_info['model']
                count = model_info['count']
                percentage = (count / total_embeddings) * 100 if total_embeddings > 0 else 0
                print(f"    {model_name:30s}: {count:6d} ({percentage:5.1f}%)")

        # Calculate health metrics
        completed_count = status_totals.get('COMPLETED', 0)
        failed_count = status_totals.get('FAILED', 0)
        pending_count = status_totals.get('PENDING', 0)

        health_score = (completed_count / total_content) * 100 if total_content > 0 else 0

        print(f"\n💚 Health Metrics:")
        print(f"Processing Success Rate: {health_score:.1f}%")

        if health_score >= 95:
            health_status = "🟢 Excellent"
        elif health_score >= 85:
            health_status = "🟡 Good"
        elif health_score >= 70:
            health_status = "🟠 Fair"
        else:
            health_status = "🔴 Poor"

        print(f"Overall Health: {health_status}")

        if failed_count > 0:
            failure_rate = (failed_count / total_content) * 100
            print(f"Failure Rate: {failure_rate:.1f}% ({failed_count} items)")

        if pending_count > 0:
            print(f"Pending Processing: {pending_count} items")

        return stats

    except Exception as e:
        print(f"Error getting advanced storage stats: {e}")
        return None
```

#### Real-world Use Case

```python
async def generate_monitoring_report():
    """Generate monitoring report for ops team."""
    stats = await vector_store.get_storage_stats()

    if not stats or 'error' in stats:
        return {"status": "error", "message": "Failed to retrieve stats"}

    total_content = stats.get('total_content', 0)
    total_embeddings = stats.get('total_embeddings', 0)
    content_by_type_status = stats.get('content_by_type_status', {})

    # Calculate key metrics
    status_totals = {}
    for content_type, status_counts in content_by_type_status.items():
        for status, count in status_counts.items():
            status_totals[status] = status_totals.get(status, 0) + count

    completed = status_totals.get('COMPLETED', 0)
    failed = status_totals.get('FAILED', 0)
    pending = status_totals.get('PENDING', 0)
    processing = status_totals.get('PROCESSING', 0)

    # Calculate alerts
    alerts = []
    warnings = []

    # Check for high failure rate
    if total_content > 0:
        failure_rate = (failed / total_content) * 100
        if failure_rate > 10:
            alerts.append(f"High failure rate: {failure_rate:.1f}% ({failed} items)")
        elif failure_rate > 5:
            warnings.append(f"Elevated failure rate: {failure_rate:.1f}% ({failed} items)")

    # Check for stuck processing
    if processing > 0:
        warnings.append(f"Items stuck in PROCESSING status: {processing}")

    # Check for large pending queue
    if pending > 100:
        warnings.append(f"Large pending queue: {pending} items")

    # Check embedding coverage
    if total_content > 0 and total_embeddings > 0:
        avg_embeddings = total_embeddings / total_content
        if avg_embeddings < 0.5:
            warnings.append(f"Low embedding coverage: {avg_embeddings:.2f} embeddings per content item")

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_content": total_content,
            "total_embeddings": total_embeddings,
            "processing_health": {
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "processing": processing,
                "success_rate": (completed / total_content * 100) if total_content > 0 else 0
            }
        },
        "alerts": alerts,
        "warnings": warnings,
        "raw_stats": stats
    }

    # Print formatted report
    print("🔍 SYSTEM MONITORING REPORT")
    print("=" * 50)
    print(f"Generated: {report['timestamp']}")

    print(f"\n📊 Summary:")
    print(f"  Content Items: {total_content:,}")
    print(f"  Embeddings: {total_embeddings:,}")
    print(f"  Success Rate: {report['summary']['processing_health']['success_rate']:.1f}%")

    if alerts:
        print(f"\n🚨 ALERTS ({len(alerts)}):")
        for alert in alerts:
            print(f"  • {alert}")

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  • {warning}")

    if not alerts and not warnings:
        print(f"\n✅ All systems nominal")

    return report

async def storage_capacity_analysis():
    """Analyze storage capacity and growth trends."""
    stats = await vector_store.get_storage_stats()

    if not stats or 'error' in stats:
        return None

    total_content = stats.get('total_content', 0)
    total_embeddings = stats.get('total_embeddings', 0)

    # Estimate storage usage
    # Assuming average content size of 2KB and embedding size of 1.5KB (384 floats * 4 bytes)
    avg_content_size = 2048  # bytes
    avg_embedding_size = 384 * 4  # bytes (float32)

    estimated_content_storage = total_content * avg_content_size
    estimated_embedding_storage = total_embeddings * avg_embedding_size
    estimated_total_storage = estimated_content_storage + estimated_embedding_storage

    def format_bytes(bytes_count):
        """Format bytes in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"

    print("💾 STORAGE CAPACITY ANALYSIS")
    print("=" * 50)

    print(f"Estimated Storage Usage:")
    print(f"  Content Data: {format_bytes(estimated_content_storage)}")
    print(f"  Embedding Data: {format_bytes(estimated_embedding_storage)}")
    print(f"  Total Estimated: {format_bytes(estimated_total_storage)}")

    # Growth projections
    if total_content > 0:
        embeddings_per_content = total_embeddings / total_content

        print(f"\nGrowth Metrics:")
        print(f"  Embeddings per Content: {embeddings_per_content:.2f}")
        print(f"  Storage per Content Item: {format_bytes(avg_content_size + (embeddings_per_content * avg_embedding_size))}")

        # Project growth scenarios
        growth_scenarios = [1000, 5000, 10000, 50000]
        print(f"\nGrowth Projections:")
        for new_items in growth_scenarios:
            projected_embeddings = new_items * embeddings_per_content
            projected_storage = new_items * (avg_content_size + (embeddings_per_content * avg_embedding_size))
            print(f"  +{new_items:,} items: +{format_bytes(projected_storage)} storage")

    return {
        "current_items": total_content,
        "current_embeddings": total_embeddings,
        "estimated_storage_bytes": estimated_total_storage,
        "estimated_storage_formatted": format_bytes(estimated_total_storage)
    }
```

#### Expected Output

```python
# Basic statistics
📊 Storage Statistics:
Total content: 1,234
Total embeddings: 5,678

# Advanced statistics
======================================================================
🗄️  VECTOR STORE STATISTICS REPORT
======================================================================
Generated: 2024-09-25 15:45:30

📈 Overall Statistics:
Total Content Items: 1,234
Total Embeddings: 5,678
Average Embeddings per Content: 4.60

📋 Content Distribution:

  By Content Type:
    source_file    :   856 ( 69.4%)
      └─ COMPLETED :  823 ( 96%)
      └─ FAILED    :   20 (  2%)
      └─ PENDING   :   13 (  2%)
    header_file    :   234 ( 19.0%)
      └─ COMPLETED :  230 ( 98%)
      └─ FAILED    :    4 (  2%)
    documentation  :   144 ( 11.7%)
      └─ COMPLETED :  140 ( 97%)
      └─ PENDING   :    4 (  3%)

  By Processing Status:
    COMPLETED   :  1193 ( 96.7%)
    FAILED      :    24 (  1.9%)
    PENDING     :    17 (  1.4%)

🤖 Embedding Models:
    BAAI/bge-small-en-v1.5:1.5       :   5678 (100.0%)

💚 Health Metrics:
Processing Success Rate: 96.7%
Overall Health: 🟢 Excellent

# Monitoring report
🔍 SYSTEM MONITORING REPORT
==================================================
Generated: 2024-09-25T15:45:30.123456

📊 Summary:
  Content Items: 1,234
  Embeddings: 5,678
  Success Rate: 96.7%

✅ All systems nominal
```

## Common Error Patterns

### Validation Errors

```python
# Common validation errors and handling
async def handle_common_errors():
    """Examples of common error patterns and handling."""

    # Empty content error
    try:
        await vector_store.store_content("source_file", "/path/file.c", "")
    except ValueError as e:
        print(f"Empty content error: {e}")  # "Content cannot be empty"

    # Invalid embedding dimensions
    try:
        invalid_embedding = [0.1] * 256  # Should be 384
        await vector_store.store_embedding(1, invalid_embedding, "text")
    except ValueError as e:
        print(f"Dimension error: {e}")  # "Expected 384 dimensions, got 256"

    # Invalid content ID
    try:
        await vector_store.get_content_by_id(99999)
        # Returns None, no exception
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Database connection errors
    try:
        await vector_store.similarity_search([0.1] * 384)
    except RuntimeError as e:
        print(f"Database error: {e}")  # Connection or query issues
```

### Network and Database Errors

```python
# Robust error handling for production
async def robust_operation(content_id: int, max_retries: int = 3):
    """Example of robust error handling with retries."""
    import asyncio

    for attempt in range(max_retries):
        try:
            content = await vector_store.get_content_by_id(content_id)
            return content

        except asyncio.TimeoutError:
            print(f"Attempt {attempt + 1}: Timeout error")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        except RuntimeError as e:
            if "connection" in str(e).lower():
                print(f"Attempt {attempt + 1}: Connection error")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
            else:
                raise  # Re-raise if not a connection error

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    raise RuntimeError(f"Failed after {max_retries} attempts")
```

## Best Practices

### 1. Batch Operations

```python
# Prefer batch operations for better performance
async def batch_best_practices():
    """Examples of efficient batch operations."""

    # Store multiple content items
    content_items = [
        {"path": "/src/file1.c", "content": "..."},
        {"path": "/src/file2.c", "content": "..."},
        {"path": "/src/file3.c", "content": "..."},
    ]

    content_ids = []
    for item in content_items:
        try:
            content_id = await vector_store.store_content(
                "source_file", item["path"], item["content"]
            )
            content_ids.append(content_id)
        except Exception as e:
            print(f"Failed to store {item['path']}: {e}")

    return content_ids
```

### 2. Resource Management

```python
# Proper resource management
async def resource_management_example():
    """Example of proper resource management."""

    # Use connection pooling and cleanup
    vector_store = VectorStore()

    try:
        # Perform operations
        stats = await vector_store.get_storage_stats()
        return stats
    finally:
        # Cleanup would go here if needed
        # vector_store.close()  # If such method existed
        pass
```

### 3. Performance Optimization

```python
# Optimize for performance
async def performance_optimized_search():
    """Performance-optimized search example."""

    from semantic_search.database.vector_store import SimilaritySearchFilter

    # Use specific filters to reduce result set
    optimized_filter = SimilaritySearchFilter(
        similarity_threshold=0.8,  # High threshold for precision
        max_results=10,  # Limit results
        include_content=False,  # Don't include content if not needed
        content_types=["source_file"]  # Specific type filter
    )

    query_embedding = [0.1] * 384
    results = await vector_store.similarity_search(query_embedding, optimized_filter)

    return results
```

This completes the comprehensive usage examples for all 9 VectorStore API methods. Each method includes basic usage, advanced usage with all parameters, error handling patterns, real-world use cases, and expected outputs.
