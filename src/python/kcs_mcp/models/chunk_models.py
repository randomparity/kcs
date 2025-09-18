"""
Pydantic models for KCS chunk processing API.

Defines request/response schemas for chunk management and processing endpoints.
These models correspond to the OpenAPI specification in contracts/chunk-api.yaml.
"""

import typing
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChunkMetadata(BaseModel):
    """Metadata for an individual chunk."""

    id: str = Field(description="Unique chunk identifier", examples=["kernel_001"])
    sequence: int = Field(description="Chunk sequence number", ge=1)
    file: str = Field(
        description="Chunk file path", examples=["chunks/kernel_001.json"]
    )
    subsystem: str = Field(description="Kernel subsystem", examples=["kernel"])
    size_bytes: int = Field(description="Chunk size in bytes", ge=0)
    checksum_sha256: str = Field(
        description="SHA256 checksum of chunk file", pattern=r"^[a-f0-9]{64}$"
    )
    symbol_count: int | None = Field(
        default=None, description="Number of symbols in chunk", ge=0
    )
    entry_point_count: int | None = Field(
        default=None, description="Number of entry points in chunk", ge=0
    )
    file_count: int | None = Field(
        default=None, description="Number of files in chunk", ge=0
    )

    class Config:
        json_schema_extra: typing.ClassVar[dict[str, typing.Any]] = {
            "example": {
                "id": "kernel_001",
                "sequence": 1,
                "file": "chunks/kernel_001.json",
                "subsystem": "kernel",
                "size_bytes": 52428800,
                "checksum_sha256": "a1b2c3d4e5f6789012345678901234567890abcd1234567890abcdef123456789",
                "symbol_count": 1245,
                "entry_point_count": 18,
                "file_count": 156,
            }
        }


class ChunkManifest(BaseModel):
    """Manifest describing all chunks in an indexing operation."""

    version: str = Field(
        description="Manifest version in semver format",
        pattern=r"^\d+\.\d+\.\d+$",
        examples=["1.0.0"],
    )
    created: datetime = Field(description="Manifest creation timestamp")
    kernel_version: str | None = Field(
        default=None, description="Kernel version", examples=["6.7.0"]
    )
    kernel_path: str | None = Field(
        default=None,
        description="Path to kernel source",
        examples=["/home/user/src/linux"],
    )
    config: str | None = Field(
        default=None, description="Kernel configuration", examples=["x86_64:defconfig"]
    )
    total_chunks: int = Field(description="Total number of chunks", ge=1)
    total_size_bytes: int | None = Field(
        default=None, description="Total size of all chunks in bytes", ge=0
    )
    chunks: list[ChunkMetadata] = Field(description="List of chunk metadata")

    @model_validator(mode="after")
    def validate_chunk_count(self) -> "ChunkManifest":
        """Ensure chunks list matches total_chunks."""
        if len(self.chunks) != self.total_chunks:
            raise ValueError(
                f"chunks list length ({len(self.chunks)}) does not match total_chunks ({self.total_chunks})"
            )
        return self

    class Config:
        json_schema_extra: typing.ClassVar[dict[str, typing.Any]] = {
            "example": {
                "version": "1.0.0",
                "created": "2025-01-18T10:30:00Z",
                "kernel_version": "6.7.0",
                "kernel_path": "/home/user/src/linux",
                "config": "x86_64:defconfig",
                "total_chunks": 60,
                "total_size_bytes": 3145728000,
                "chunks": [
                    {
                        "id": "kernel_001",
                        "sequence": 1,
                        "file": "chunks/kernel_001.json",
                        "subsystem": "kernel",
                        "size_bytes": 52428800,
                        "checksum_sha256": "a1b2c3d4e5f6789012345678901234567890abcd1234567890abcdef123456789",
                        "symbol_count": 1245,
                        "entry_point_count": 18,
                        "file_count": 156,
                    }
                ],
            }
        }


class ProcessingStatus(BaseModel):
    """Processing status of a chunk."""

    chunk_id: str = Field(description="Chunk identifier")
    manifest_version: str = Field(
        ..., description="Version of the manifest this chunk belongs to"
    )
    status: str = Field(
        ...,
        description="Current processing state",
        pattern=r"^(pending|processing|completed|failed)$",
    )
    started_at: datetime | None = Field(None, description="Processing start timestamp")
    completed_at: datetime | None = Field(
        default=None, description="Processing completion timestamp"
    )
    error_message: str | None = Field(
        default=None, description="Error message if processing failed"
    )
    retry_count: int | None = Field(
        default=None, description="Number of retry attempts", ge=0, le=3
    )
    symbols_processed: int | None = Field(
        default=None, description="Number of symbols processed", ge=0
    )
    checksum_verified: bool | None = Field(
        default=None, description="Whether checksum was verified"
    )

    class Config:
        json_schema_extra: typing.ClassVar[dict[str, typing.Any]] = {
            "example": {
                "chunk_id": "kernel_001",
                "manifest_version": "1.0.0",
                "status": "completed",
                "started_at": "2025-01-18T10:45:00Z",
                "completed_at": "2025-01-18T10:47:30Z",
                "error_message": None,
                "retry_count": 0,
                "symbols_processed": 1245,
                "checksum_verified": True,
            }
        }


# Request models for chunk processing endpoints


class ProcessChunkRequest(BaseModel):
    """Request to process a single chunk."""

    force: bool = Field(
        False, description="Force reprocessing even if already completed"
    )


class ProcessBatchRequest(BaseModel):
    """Request to process multiple chunks in batch."""

    chunk_ids: list[str] = Field(
        description="List of chunk IDs to process", min_length=1, max_length=100
    )
    parallelism: int = Field(
        default=4, description="Number of chunks to process in parallel", ge=1, le=10
    )


# Response models for chunk processing endpoints


class ProcessChunkResponse(BaseModel):
    """Response from chunk processing request."""

    message: str = Field(description="Processing status message")
    chunk_id: str = Field(description="Chunk identifier")
    status: str = Field(description="Processing status")


class ProcessBatchResponse(BaseModel):
    """Response from batch processing request."""

    message: str = Field(description="Batch processing status message")
    total_chunks: int = Field(description="Total number of chunks to process")
    processing: list[str] = Field(description="List of chunk IDs being processed")


# Database models for internal use


class ChunkProcessingRecord(BaseModel):
    """Database record for chunk processing status."""

    chunk_id: str = Field(description="Primary key: chunk identifier")
    manifest_version: str = Field(description="Version of manifest")
    status: str = Field(
        "pending",
        description="Processing status",
        pattern=r"^(pending|processing|completed|failed)$",
    )
    started_at: datetime | None = Field(None, description="Processing start time")
    completed_at: datetime | None = Field(
        default=None, description="Processing completion time"
    )
    error_message: str | None = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retry attempts", ge=0)
    symbols_processed: int = Field(0, description="Number of symbols processed", ge=0)
    checksum_verified: bool = Field(False, description="Whether checksum was verified")
    created_at: datetime = Field(description="Record creation time")
    updated_at: datetime = Field(description="Record last update time")


class IndexingManifestRecord(BaseModel):
    """Database record for indexing manifest."""

    version: str = Field(description="Primary key: manifest version")
    created: datetime = Field(description="Manifest creation time")
    kernel_version: str | None = Field(None, description="Kernel version")
    kernel_path: str | None = Field(None, description="Path to kernel source")
    config: str | None = Field(None, description="Kernel configuration")
    total_chunks: int = Field(description="Total number of chunks", ge=1)
    total_size_bytes: int | None = Field(None, description="Total size in bytes", ge=0)
    manifest_data: dict[str, Any] = Field(description="Complete manifest as JSONB")
    created_at: datetime = Field(description="Database record creation time")
