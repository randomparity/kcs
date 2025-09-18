//! pgvector database integration module
//!
//! Provides PostgreSQL pgvector integration for storing and searching
//! embedding vectors for semantic search functionality.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{Pool, Postgres, Row};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub struct VectorDatabase {
    pool: Pool<Postgres>,
    dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchParams {
    pub vector: Vec<f32>,
    pub limit: i32,
    pub threshold: Option<f32>,
    pub kernel_config: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    pub symbol_id: i64,
    pub name: String,
    pub file_path: String,
    pub line_number: i32,
    pub distance: f32,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub symbol_id: i64,
    pub embedding: Vec<f32>,
    pub metadata: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl VectorDatabase {
    pub async fn new(database_url: &str) -> Result<Self> {
        Self::with_config(database_url, 384, 5).await
    }

    pub async fn with_config(
        database_url: &str,
        dimension: usize,
        max_connections: u32,
    ) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL database")?;

        let db = Self { pool, dimension };

        // Ensure pgvector extension is installed
        db.ensure_extension().await?;

        // Create tables if they don't exist
        db.create_tables().await?;

        Ok(db)
    }

    /// Ensure pgvector extension is installed
    async fn ensure_extension(&self) -> Result<()> {
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .context("Failed to create pgvector extension")?;

        info!("pgvector extension ready");
        Ok(())
    }

    /// Create necessary tables for storing embeddings
    async fn create_tables(&self) -> Result<()> {
        let query = format!(
            r#"
            CREATE TABLE IF NOT EXISTS symbol_embeddings (
                symbol_id BIGINT PRIMARY KEY REFERENCES symbol(id) ON DELETE CASCADE,
                embedding vector({}),
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            self.dimension
        );

        sqlx::query(&query)
            .execute(&self.pool)
            .await
            .context("Failed to create symbol_embeddings table")?;

        // Create index for vector similarity search
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_symbol_embeddings_vector
            ON symbol_embeddings USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100)
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create vector index")?;

        // Create index on metadata for filtering
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_symbol_embeddings_metadata
            ON symbol_embeddings USING GIN (metadata)
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create metadata index")?;

        info!("Database tables and indexes ready");
        Ok(())
    }

    pub async fn store_embedding(
        &self,
        symbol_id: i64,
        embedding: &[f32],
        metadata: serde_json::Value,
    ) -> Result<()> {
        if embedding.len() != self.dimension {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            );
        }

        // Convert to pgvector format
        let embedding_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        sqlx::query(
            r#"
            INSERT INTO symbol_embeddings (symbol_id, embedding, metadata, updated_at)
            VALUES ($1, $2::vector, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (symbol_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(symbol_id)
        .bind(&embedding_str)
        .bind(&metadata)
        .execute(&self.pool)
        .await
        .context("Failed to store embedding")?;

        debug!("Stored embedding for symbol_id: {}", symbol_id);
        Ok(())
    }

    pub async fn search(&self, params: VectorSearchParams) -> Result<Vec<VectorSearchResult>> {
        if params.vector.len() != self.dimension {
            anyhow::bail!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dimension,
                params.vector.len()
            );
        }

        // Convert query vector to pgvector format
        let query_vector_str = format!(
            "[{}]",
            params
                .vector
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        // Build the query based on parameters
        let mut query_builder = String::from(
            r#"
            SELECT
                se.symbol_id,
                s.name,
                s.file_path,
                s.line_number,
                se.embedding <-> $1::vector as distance,
                se.metadata
            FROM symbol_embeddings se
            JOIN symbol s ON se.symbol_id = s.id
            WHERE 1=1
            "#,
        );

        let mut bind_values = vec![query_vector_str];
        let mut bind_index = 2;

        // Add threshold filter if specified
        if let Some(threshold) = params.threshold {
            query_builder.push_str(&format!(
                " AND se.embedding <-> $1::vector < ${}",
                bind_index
            ));
            bind_values.push(threshold.to_string());
            bind_index += 1;
        }

        // Add kernel config filter if specified
        if let Some(ref config) = params.kernel_config {
            query_builder.push_str(&format!(
                " AND se.metadata->>'kernel_config' = ${}",
                bind_index
            ));
            bind_values.push(config.clone());
            bind_index += 1;
        }

        query_builder.push_str(" ORDER BY distance ASC");
        query_builder.push_str(&format!(" LIMIT ${}", bind_index));
        bind_values.push(params.limit.to_string());

        // Execute the query
        let mut query = sqlx::query(&query_builder);
        for value in &bind_values {
            query = query.bind(value);
        }

        let rows = query
            .fetch_all(&self.pool)
            .await
            .context("Failed to execute vector search")?;

        // Map results
        let results: Vec<VectorSearchResult> = rows
            .into_iter()
            .map(|row: PgRow| VectorSearchResult {
                symbol_id: row.get("symbol_id"),
                name: row.get("name"),
                file_path: row.get("file_path"),
                line_number: row.get("line_number"),
                distance: row.get("distance"),
                metadata: row.get("metadata"),
            })
            .collect();

        debug!("Found {} results for vector search", results.len());
        Ok(results)
    }

    pub async fn update_embedding(&self, symbol_id: i64, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dimension {
            anyhow::bail!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            );
        }

        let embedding_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let result = sqlx::query(
            r#"
            UPDATE symbol_embeddings
            SET embedding = $2::vector, updated_at = CURRENT_TIMESTAMP
            WHERE symbol_id = $1
            "#,
        )
        .bind(symbol_id)
        .bind(&embedding_str)
        .execute(&self.pool)
        .await
        .context("Failed to update embedding")?;

        if result.rows_affected() == 0 {
            warn!("No embedding found for symbol_id: {}", symbol_id);
        } else {
            debug!("Updated embedding for symbol_id: {}", symbol_id);
        }

        Ok(())
    }

    pub async fn delete_embedding(&self, symbol_id: i64) -> Result<()> {
        let result = sqlx::query(
            r#"
            DELETE FROM symbol_embeddings
            WHERE symbol_id = $1
            "#,
        )
        .bind(symbol_id)
        .execute(&self.pool)
        .await
        .context("Failed to delete embedding")?;

        if result.rows_affected() == 0 {
            warn!("No embedding found to delete for symbol_id: {}", symbol_id);
        } else {
            info!("Deleted embedding for symbol_id: {}", symbol_id);
        }

        Ok(())
    }

    pub async fn create_index(&self) -> Result<()> {
        // Drop existing index if it exists
        sqlx::query("DROP INDEX IF EXISTS idx_symbol_embeddings_vector")
            .execute(&self.pool)
            .await
            .context("Failed to drop existing index")?;

        // Create new ivfflat index
        sqlx::query(
            r#"
            CREATE INDEX idx_symbol_embeddings_vector
            ON symbol_embeddings USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100)
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create vector index")?;

        info!("Vector index created successfully");
        Ok(())
    }

    /// Get embedding for a specific symbol
    pub async fn get_embedding(&self, symbol_id: i64) -> Result<Option<EmbeddingRecord>> {
        let row = sqlx::query(
            r#"
            SELECT symbol_id, embedding::text, metadata, created_at, updated_at
            FROM symbol_embeddings
            WHERE symbol_id = $1
            "#,
        )
        .bind(symbol_id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get embedding")?;

        if let Some(row) = row {
            let embedding_str: String = row.get(1);
            let embedding = parse_pgvector_string(&embedding_str)?;

            Ok(Some(EmbeddingRecord {
                symbol_id: row.get(0),
                embedding,
                metadata: row.get(2),
                created_at: row.get(3),
                updated_at: row.get(4),
            }))
        } else {
            Ok(None)
        }
    }

    /// Count total embeddings in the database
    pub async fn count_embeddings(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) FROM symbol_embeddings")
            .fetch_one(&self.pool)
            .await
            .context("Failed to count embeddings")?;

        Ok(row.get(0))
    }

    /// Batch store multiple embeddings
    pub async fn batch_store_embeddings(
        &self,
        embeddings: Vec<(i64, Vec<f32>, serde_json::Value)>,
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        let count = embeddings.len();

        // Start a transaction
        let mut tx = self.pool.begin().await?;

        for (symbol_id, embedding, metadata) in embeddings {
            if embedding.len() != self.dimension {
                error!(
                    "Skipping symbol_id {}: dimension mismatch ({} != {})",
                    symbol_id,
                    embedding.len(),
                    self.dimension
                );
                continue;
            }

            let embedding_str = format!(
                "[{}]",
                embedding
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );

            sqlx::query(
                r#"
                INSERT INTO symbol_embeddings (symbol_id, embedding, metadata, updated_at)
                VALUES ($1, $2::vector, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                "#,
            )
            .bind(symbol_id)
            .bind(&embedding_str)
            .bind(&metadata)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        info!("Batch stored {} embeddings", count);
        Ok(())
    }

    /// Find similar symbols to a given symbol
    pub async fn find_similar(
        &self,
        symbol_id: i64,
        limit: i32,
    ) -> Result<Vec<VectorSearchResult>> {
        // First get the embedding for the given symbol
        let embedding = self.get_embedding(symbol_id).await?;

        if let Some(record) = embedding {
            // Search for similar symbols excluding the original
            let params = VectorSearchParams {
                vector: record.embedding,
                limit: limit + 1, // Get one extra to exclude self
                threshold: None,
                kernel_config: None,
            };

            let mut results = self.search(params).await?;

            // Filter out the original symbol
            results.retain(|r| r.symbol_id != symbol_id);
            results.truncate(limit as usize);

            Ok(results)
        } else {
            warn!("No embedding found for symbol_id: {}", symbol_id);
            Ok(Vec::new())
        }
    }
}

/// Parse pgvector string format "[1.0,2.0,3.0]" into Vec<f32>
fn parse_pgvector_string(s: &str) -> Result<Vec<f32>> {
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(',')
        .map(|x| x.parse::<f32>().context("Failed to parse vector component"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_search_params() {
        let params = VectorSearchParams {
            vector: vec![0.1, 0.2, 0.3],
            limit: 10,
            threshold: Some(0.8),
            kernel_config: Some("x86_64:defconfig".to_string()),
        };

        assert_eq!(params.limit, 10);
        assert_eq!(params.vector.len(), 3);
    }

    #[test]
    fn test_parse_pgvector_string() {
        let input = "[1.0,2.5,-3.7,4.2]";
        let result = parse_pgvector_string(input).unwrap();
        assert_eq!(result, vec![1.0, 2.5, -3.7, 4.2]);
    }

    #[test]
    fn test_parse_pgvector_string_single() {
        let input = "[42.0]";
        let result = parse_pgvector_string(input).unwrap();
        assert_eq!(result, vec![42.0]);
    }

    #[tokio::test]
    async fn test_database_connection() {
        // Skip if no database URL is provided
        let db_url = std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs_test"
                .to_string()
        });

        // Try to connect (this might fail if no test database is available)
        if VectorDatabase::new(&db_url).await.is_ok() {
            // Connection successful - test passes
        } else {
            // Skip test if database is not available
            eprintln!("Skipping database test - no test database available");
        }
    }
}
