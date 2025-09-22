"""
Configuration management for semantic search engine.

Provides centralized configuration for embedding models, database connections,
search parameters, and performance settings with support for multiple sources.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .logging_integration import get_semantic_search_logger

logger = get_semantic_search_logger(__name__)


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: str | None = None
    dimension: int = 384
    device: str = "cpu"
    trust_remote_code: bool = False
    normalize_embeddings: bool = True
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_cache_dir is None:
            # Default cache directory
            self.model_cache_dir = str(
                Path.home() / ".cache" / "semantic_search" / "models"
            )

        # Ensure cache directory exists
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

        # Validate model configuration
        if self.dimension <= 0:
            raise ValueError("Model dimension must be positive")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.max_length <= 0:
            raise ValueError("Max length must be positive")


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""

    host: str = "localhost"
    port: int = 5432
    database: str = "kcs"
    username: str = "kcs"
    password: str = ""
    min_pool_size: int = 2
    max_pool_size: int = 10
    command_timeout: int = 30
    connection_timeout: int = 10
    ssl_mode: str = "prefer"

    def to_url(self) -> str:
        """Convert configuration to database URL."""
        if self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return (
                f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging (without password)."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size,
            "command_timeout": self.command_timeout,
            "ssl_mode": self.ssl_mode,
        }


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    default_max_results: int = 10
    default_similarity_threshold: float = 0.7
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    enable_bm25: bool = True
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    cache_query_results: bool = True
    cache_ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        """Validate search configuration."""
        if not (0.0 <= self.default_similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

        if (
            self.chunk_size < self.min_chunk_size
            or self.chunk_size > self.max_chunk_size
        ):
            raise ValueError(
                f"Chunk size must be between {self.min_chunk_size} and {self.max_chunk_size}"
            )

        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "Chunk overlap must be non-negative and less than chunk size"
            )

        if not (0.0 <= self.bm25_weight <= 1.0) or not (
            0.0 <= self.semantic_weight <= 1.0
        ):
            raise ValueError("Weights must be between 0.0 and 1.0")

        if abs(self.bm25_weight + self.semantic_weight - 1.0) > 0.001:
            raise ValueError("BM25 weight and semantic weight must sum to 1.0")


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""

    max_query_time_ms: int = 600
    max_indexing_time_ms: int = 30000
    enable_query_optimization: bool = True
    enable_parallel_processing: bool = True
    max_concurrent_requests: int = 10
    embedding_batch_size: int = 32
    vector_index_lists: int = 100
    cleanup_interval_hours: int = 24
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000

    def __post_init__(self) -> None:
        """Validate performance configuration."""
        if self.max_query_time_ms <= 0:
            raise ValueError("Max query time must be positive")

        if self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be positive")

        if self.embedding_batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")


@dataclass
class SemanticSearchConfig:
    """Main configuration class for semantic search engine."""

    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    log_level: str = "INFO"
    log_format: str = "json"
    enable_metrics: bool = True
    config_file_path: str | None = None

    @classmethod
    def from_env(cls, prefix: str = "SEMANTIC_SEARCH_") -> "SemanticSearchConfig":
        """
        Create configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            Configuration instance
        """
        # Embedding configuration
        embedding_config = EmbeddingModelConfig(
            model_name=os.getenv(f"{prefix}MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            model_cache_dir=os.getenv(f"{prefix}MODEL_CACHE_DIR"),
            dimension=int(os.getenv(f"{prefix}MODEL_DIMENSION", "384")),
            device=os.getenv(f"{prefix}MODEL_DEVICE", "cpu"),
            trust_remote_code=os.getenv(f"{prefix}TRUST_REMOTE_CODE", "false").lower()
            == "true",
            normalize_embeddings=os.getenv(
                f"{prefix}NORMALIZE_EMBEDDINGS", "true"
            ).lower()
            == "true",
            batch_size=int(os.getenv(f"{prefix}BATCH_SIZE", "32")),
            max_length=int(os.getenv(f"{prefix}MAX_LENGTH", "512")),
        )

        # Database configuration
        database_config = DatabaseConfig(
            host=os.getenv(f"{prefix}DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
            port=int(os.getenv(f"{prefix}DB_PORT", os.getenv("POSTGRES_PORT", "5432"))),
            database=os.getenv(f"{prefix}DB_NAME", os.getenv("POSTGRES_DB", "kcs")),
            username=os.getenv(f"{prefix}DB_USER", os.getenv("POSTGRES_USER", "kcs")),
            password=os.getenv(
                f"{prefix}DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "")
            ),
            min_pool_size=int(os.getenv(f"{prefix}DB_MIN_POOL_SIZE", "2")),
            max_pool_size=int(os.getenv(f"{prefix}DB_MAX_POOL_SIZE", "10")),
            command_timeout=int(os.getenv(f"{prefix}DB_COMMAND_TIMEOUT", "30")),
            connection_timeout=int(os.getenv(f"{prefix}DB_CONNECTION_TIMEOUT", "10")),
            ssl_mode=os.getenv(f"{prefix}DB_SSL_MODE", "prefer"),
        )

        # Search configuration
        search_config = SearchConfig(
            default_max_results=int(os.getenv(f"{prefix}DEFAULT_MAX_RESULTS", "10")),
            default_similarity_threshold=float(
                os.getenv(f"{prefix}DEFAULT_SIMILARITY_THRESHOLD", "0.7")
            ),
            chunk_size=int(os.getenv(f"{prefix}CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv(f"{prefix}CHUNK_OVERLAP", "50")),
            enable_bm25=os.getenv(f"{prefix}ENABLE_BM25", "true").lower() == "true",
            bm25_weight=float(os.getenv(f"{prefix}BM25_WEIGHT", "0.3")),
            semantic_weight=float(os.getenv(f"{prefix}SEMANTIC_WEIGHT", "0.7")),
            cache_query_results=os.getenv(f"{prefix}CACHE_RESULTS", "true").lower()
            == "true",
            cache_ttl_seconds=int(os.getenv(f"{prefix}CACHE_TTL_SECONDS", "3600")),
        )

        # Performance configuration
        performance_config = PerformanceConfig(
            max_query_time_ms=int(os.getenv(f"{prefix}MAX_QUERY_TIME_MS", "600")),
            max_indexing_time_ms=int(
                os.getenv(f"{prefix}MAX_INDEXING_TIME_MS", "30000")
            ),
            enable_query_optimization=os.getenv(
                f"{prefix}ENABLE_QUERY_OPTIMIZATION", "true"
            ).lower()
            == "true",
            enable_parallel_processing=os.getenv(
                f"{prefix}ENABLE_PARALLEL_PROCESSING", "true"
            ).lower()
            == "true",
            max_concurrent_requests=int(
                os.getenv(f"{prefix}MAX_CONCURRENT_REQUESTS", "10")
            ),
            embedding_batch_size=int(os.getenv(f"{prefix}EMBEDDING_BATCH_SIZE", "32")),
            vector_index_lists=int(os.getenv(f"{prefix}VECTOR_INDEX_LISTS", "100")),
            cleanup_interval_hours=int(
                os.getenv(f"{prefix}CLEANUP_INTERVAL_HOURS", "24")
            ),
            log_slow_queries=os.getenv(f"{prefix}LOG_SLOW_QUERIES", "true").lower()
            == "true",
            slow_query_threshold_ms=int(
                os.getenv(f"{prefix}SLOW_QUERY_THRESHOLD_MS", "1000")
            ),
        )

        return cls(
            embedding=embedding_config,
            database=database_config,
            search=search_config,
            performance=performance_config,
            log_level=os.getenv(f"{prefix}LOG_LEVEL", "INFO"),
            log_format=os.getenv(f"{prefix}LOG_FORMAT", "json"),
            enable_metrics=os.getenv(f"{prefix}ENABLE_METRICS", "true").lower()
            == "true",
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SemanticSearchConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration instance
        """
        # Extract embedding config
        embedding_dict = config_dict.get("embedding", {})
        embedding_config = EmbeddingModelConfig(**embedding_dict)

        # Extract database config
        database_dict = config_dict.get("database", {})
        database_config = DatabaseConfig(**database_dict)

        # Extract search config
        search_dict = config_dict.get("search", {})
        search_config = SearchConfig(**search_dict)

        # Extract performance config
        performance_dict = config_dict.get("performance", {})
        performance_config = PerformanceConfig(**performance_dict)

        return cls(
            embedding=embedding_config,
            database=database_config,
            search=search_config,
            performance=performance_config,
            log_level=config_dict.get("log_level", "INFO"),
            log_format=config_dict.get("log_format", "json"),
            enable_metrics=config_dict.get("enable_metrics", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary (without sensitive data)
        """
        return {
            "embedding": {
                "model_name": self.embedding.model_name,
                "model_cache_dir": self.embedding.model_cache_dir,
                "dimension": self.embedding.dimension,
                "device": self.embedding.device,
                "trust_remote_code": self.embedding.trust_remote_code,
                "normalize_embeddings": self.embedding.normalize_embeddings,
                "batch_size": self.embedding.batch_size,
                "max_length": self.embedding.max_length,
            },
            "database": self.database.to_dict(),
            "search": {
                "default_max_results": self.search.default_max_results,
                "default_similarity_threshold": self.search.default_similarity_threshold,
                "chunk_size": self.search.chunk_size,
                "chunk_overlap": self.search.chunk_overlap,
                "enable_bm25": self.search.enable_bm25,
                "bm25_weight": self.search.bm25_weight,
                "semantic_weight": self.search.semantic_weight,
                "cache_query_results": self.search.cache_query_results,
                "cache_ttl_seconds": self.search.cache_ttl_seconds,
            },
            "performance": {
                "max_query_time_ms": self.performance.max_query_time_ms,
                "max_indexing_time_ms": self.performance.max_indexing_time_ms,
                "enable_query_optimization": self.performance.enable_query_optimization,
                "enable_parallel_processing": self.performance.enable_parallel_processing,
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "embedding_batch_size": self.performance.embedding_batch_size,
                "vector_index_lists": self.performance.vector_index_lists,
                "cleanup_interval_hours": self.performance.cleanup_interval_hours,
                "log_slow_queries": self.performance.log_slow_queries,
                "slow_query_threshold_ms": self.performance.slow_query_threshold_ms,
            },
            "log_level": self.log_level,
            "log_format": self.log_format,
            "enable_metrics": self.enable_metrics,
        }

    def validate(self) -> list[str]:
        """
        Validate the entire configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        try:
            # Validate embedding config
            self.embedding.__post_init__()
        except ValueError as e:
            errors.append(f"Embedding config: {e}")

        try:
            # Validate search config
            self.search.__post_init__()
        except ValueError as e:
            errors.append(f"Search config: {e}")

        try:
            # Validate performance config
            self.performance.__post_init__()
        except ValueError as e:
            errors.append(f"Performance config: {e}")

        # Validate log level
        valid_log_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.log_level}")

        return errors


# Pydantic models for API validation
class EmbeddingModelConfigPydantic(BaseModel):
    """Pydantic model for embedding configuration validation."""

    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Hugging Face model name"
    )
    model_cache_dir: str | None = Field(None, description="Model cache directory")
    dimension: int = Field(default=384, gt=0, description="Embedding dimension")
    device: str = Field(
        default="cpu", pattern=r"^(cpu|cuda|auto)$", description="Device for inference"
    )
    trust_remote_code: bool = Field(
        default=False, description="Trust remote code in model"
    )
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
    batch_size: int = Field(default=32, gt=0, description="Batch size for embedding")
    max_length: int = Field(default=512, gt=0, description="Maximum sequence length")


class SearchConfigPydantic(BaseModel):
    """Pydantic model for search configuration validation."""

    default_max_results: int = Field(
        default=10, ge=1, le=100, description="Default max results"
    )
    default_similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Default similarity threshold"
    )
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: int = Field(default=50, ge=0, description="Chunk overlap size")
    enable_bm25: bool = Field(default=True, description="Enable BM25 scoring")
    bm25_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="BM25 weight in hybrid scoring"
    )
    semantic_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Semantic weight in hybrid scoring"
    )
    cache_query_results: bool = Field(default=True, description="Cache query results")
    cache_ttl_seconds: int = Field(
        default=3600, gt=0, description="Cache TTL in seconds"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        """Validate chunk overlap is less than chunk size."""
        if hasattr(info.data, "chunk_size") and v >= info.data["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


# Global configuration instance
_global_config: SemanticSearchConfig | None = None


def get_config() -> SemanticSearchConfig:
    """
    Get the global semantic search configuration.

    Returns:
        Global configuration instance
    """
    global _global_config

    if _global_config is None:
        _global_config = SemanticSearchConfig.from_env()
        logger.info("Loaded semantic search configuration from environment")

    return _global_config


def set_config(config: SemanticSearchConfig) -> None:
    """
    Set the global semantic search configuration.

    Args:
        config: Configuration instance to set as global
    """
    global _global_config

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    _global_config = config
    logger.info("Updated global semantic search configuration")


def reload_config() -> SemanticSearchConfig:
    """
    Reload configuration from environment variables.

    Returns:
        Reloaded configuration instance
    """
    global _global_config

    _global_config = SemanticSearchConfig.from_env()
    logger.info("Reloaded semantic search configuration from environment")

    return _global_config


def get_model_cache_path(model_name: str) -> Path:
    """
    Get the cache path for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Path to model cache directory
    """
    config = get_config()
    cache_dir = Path(
        config.embedding.model_cache_dir or "~/.cache/semantic_search/models"
    )
    cache_dir = cache_dir.expanduser()

    # Create safe directory name from model name
    safe_name = model_name.replace("/", "--").replace(":", "_")
    model_path = cache_dir / safe_name

    # Ensure directory exists
    model_path.mkdir(parents=True, exist_ok=True)

    return model_path
