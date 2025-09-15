"""
MCP Resources implementation - read-only data access endpoints.

These endpoints provide direct access to kernel data without
requiring complex queries, supporting MCP resource protocol.
"""


import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from .database import Database, get_database
from .models import ErrorResponse, ResourceList

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=ResourceList)
async def list_resources(db: Database = Depends(get_database)) -> ResourceList:
    """
    List available MCP resources.

    Returns catalog of available read-only data resources
    for direct access by MCP clients.
    """
    logger.info("list_resources")

    # TODO: Implement dynamic resource discovery
    # For now, return static list of available resources

    resources = [
        "kernel://configs",
        "kernel://architectures",
        "kernel://subsystems",
        "kernel://syscalls",
        "kernel://ioctls",
        "kernel://symbols/exports",
        "kernel://modules",
        "kernel://maintainers",
    ]

    return ResourceList(resources=resources)


@router.get("/kernel/configs")
async def get_kernel_configs(db: Database = Depends(get_database)):
    """
    Get available kernel configurations.

    Returns list of supported kernel config variations
    (e.g., defconfig, allmodconfig) for each architecture.
    """
    logger.info("get_kernel_configs")

    try:
        async with db.acquire() as conn:
            # TODO: Implement actual config query
            # This would query the kernel_config table

            sql = """
            SELECT arch, config_name, path
            FROM kernel_config
            ORDER BY arch, config_name
            """

            rows = await conn.fetch(sql)

            configs = {}
            for row in rows:
                arch = row["arch"]
                if arch not in configs:
                    configs[arch] = []
                configs[arch].append({"name": row["config_name"], "path": row["path"]})

            return {"configurations": configs}

    except Exception as e:
        logger.error("get_kernel_configs_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="config_lookup_failed",
                message=f"Failed to retrieve kernel configs: {e!s}",
            ).dict(),
        )


@router.get("/kernel/architectures")
async def get_architectures(db: Database = Depends(get_database)):
    """
    Get supported architectures.

    Returns list of CPU architectures with kernel support.
    """
    logger.info("get_architectures")

    # TODO: Query from database
    architectures = [
        {
            "name": "x86_64",
            "description": "64-bit x86 architecture",
            "configs": ["defconfig", "allmodconfig", "allyesconfig"],
        },
        {
            "name": "arm64",
            "description": "64-bit ARM architecture",
            "configs": ["defconfig", "allmodconfig"],
        },
        {
            "name": "ppc64le",
            "description": "PowerPC 64-bit little endian",
            "configs": ["defconfig", "allmodconfig"],
        },
        {
            "name": "s390x",
            "description": "IBM System/390 64-bit",
            "configs": ["defconfig", "allmodconfig"],
        },
    ]

    return {"architectures": architectures}


@router.get("/kernel/subsystems")
async def get_subsystems(db: Database = Depends(get_database)):
    """
    Get kernel subsystems.

    Returns hierarchical view of kernel subsystem organization.
    """
    logger.info("get_subsystems")

    try:
        async with db.acquire() as conn:
            # TODO: Implement subsystem discovery from maintainers file
            # and directory structure analysis

            subsystems = [
                {
                    "name": "VFS",
                    "path": "fs/",
                    "maintainers": ["vfs@kernel.org"],
                    "description": "Virtual File System layer",
                },
                {
                    "name": "Networking",
                    "path": "net/",
                    "maintainers": ["netdev@vger.kernel.org"],
                    "description": "Network stack and protocols",
                },
                {
                    "name": "Block Layer",
                    "path": "block/",
                    "maintainers": ["linux-block@vger.kernel.org"],
                    "description": "Block device abstraction layer",
                },
                {
                    "name": "Memory Management",
                    "path": "mm/",
                    "maintainers": ["linux-mm@kvack.org"],
                    "description": "Virtual memory management",
                },
            ]

            return {"subsystems": subsystems}

    except Exception as e:
        logger.error("get_subsystems_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="subsystem_lookup_failed",
                message=f"Failed to retrieve subsystems: {e!s}",
            ).dict(),
        )


@router.get("/kernel/syscalls")
async def get_syscalls(arch: str | None = None, db: Database = Depends(get_database)):
    """
    Get system call definitions.

    Returns syscall numbers, names, and signatures by architecture.
    """
    logger.info("get_syscalls", arch=arch)

    try:
        async with db.acquire() as conn:
            # TODO: Implement syscall table query
            # This would parse arch/*/entry/syscalls/syscall_*.tbl

            if arch and arch not in ["x86_64", "arm64", "ppc64le", "s390x"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorResponse(
                        error="invalid_architecture",
                        message=f"Unsupported architecture: {arch}",
                    ).dict(),
                )

            # Mock syscall data
            syscalls = [
                {
                    "number": 0,
                    "name": "read",
                    "signature": "sys_read(unsigned int fd, char __user *buf, size_t count)",
                    "architecture": arch or "x86_64",
                },
                {
                    "number": 1,
                    "name": "write",
                    "signature": "sys_write(unsigned int fd, const char __user *buf, size_t count)",
                    "architecture": arch or "x86_64",
                },
                {
                    "number": 2,
                    "name": "open",
                    "signature": "sys_open(const char __user *filename, int flags, umode_t mode)",
                    "architecture": arch or "x86_64",
                },
            ]

            return {"syscalls": syscalls}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_syscalls_error", error=str(e), arch=arch)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="syscall_lookup_failed",
                message=f"Failed to retrieve syscalls: {e!s}",
            ).dict(),
        )


@router.get("/kernel/exports")
async def get_exported_symbols(
    module: str | None = None, db: Database = Depends(get_database)
):
    """
    Get EXPORT_SYMBOL declarations.

    Returns symbols exported by kernel modules for external use.
    """
    logger.info("get_exported_symbols", module=module)

    try:
        async with db.acquire() as conn:
            # TODO: Query EXPORT_SYMBOL patterns from database

            sql = """
            SELECT s.name, s.kind, f.path, m.name as module_name
            FROM symbol s
            JOIN file f ON s.file_id = f.id
            LEFT JOIN module m ON f.path LIKE CONCAT(m.path, '%')
            WHERE s.exported = true
            """

            if module:
                sql += " AND m.name = $1"
                rows = await conn.fetch(sql, module)
            else:
                rows = await conn.fetch(sql)

            exports = [
                {
                    "symbol": row["name"],
                    "kind": row["kind"],
                    "module": row["module_name"],
                    "path": row["path"],
                }
                for row in rows
            ]

            return {"exported_symbols": exports}

    except Exception as e:
        logger.error("get_exported_symbols_error", error=str(e), module=module)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="export_lookup_failed",
                message=f"Failed to retrieve exported symbols: {e!s}",
            ).dict(),
        )
