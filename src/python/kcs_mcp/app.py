"""
FastAPI application for KCS MCP server.

Provides Model Context Protocol endpoints for kernel analysis.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .database import Database, set_database
from .models import ErrorResponse
from .resources import router as resources_router
from .tools import router as tools_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://kcs:kcs_dev_password@localhost:5432/kcs"
)
JWT_SECRET = os.getenv("JWT_SECRET", "dev_jwt_secret_change_in_production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting KCS MCP server")

    # Initialize database connection
    try:
        database = Database(DATABASE_URL)
        await database.connect()
        app.state.database = database
        set_database(database)
        logger.info("Database connected successfully")
    except Exception as e:
        logger.warning("Database connection failed, using mock mode", error=str(e))
        # Set a mock database for testing
        app.state.database = None

    logger.info("KCS MCP server started successfully")

    yield

    # Cleanup
    logger.info("Shutting down KCS MCP server")
    await database.disconnect()
    logger.info("KCS MCP server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Kernel Context Server MCP API",
    description="Model Context Protocol API for Linux kernel analysis",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Verify JWT token and return user info."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # For development, accept test tokens
    if token.startswith("test_token_") or token == "dev-token":
        return "test_user"

    # TODO: Implement proper JWT verification
    # For now, reject all other tokens
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> dict[str, Any]:
    """Global exception handler."""
    logger.error("Unhandled exception", exc_info=exc, path=request.url.path)

    error_response = ErrorResponse(
        error="internal_server_error", message="An internal server error occurred"
    )
    return {"error": error_response.error, "message": error_response.message}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "indexed_at": None,  # TODO: Get from database
    }


@app.get("/metrics")
async def metrics() -> str:
    """Prometheus metrics endpoint."""
    # TODO: Implement proper metrics
    return "# KCS metrics\nkcs_requests_total 0\n"


# Include routers
app.include_router(
    resources_router,
    prefix="/mcp/resources",
    tags=["Resources"],
    dependencies=[Depends(verify_token)],
)

app.include_router(
    tools_router,
    prefix="/mcp/tools",
    tags=["Tools"],
    dependencies=[Depends(verify_token)],
)


if __name__ == "__main__":
    import uvicorn

    # For development
    uvicorn.run(
        "kcs_mcp.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
