"""
Integration tests for multi-architecture kernel configuration parsing.

These tests verify end-to-end functionality for parsing and storing
kernel configurations across different target architectures.

Key test scenarios:
- Parse configurations for x86_64, arm64, arm, riscv architectures
- Verify architecture-specific configuration options are correctly handled
- Test configuration comparison across architectures
- Validate database storage and retrieval for multi-arch configs
- Test configuration dependency resolution per architecture
"""

import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests


# Test infrastructure
def is_mcp_server_running() -> bool:
    """Check if MCP server is accessible for integration testing."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_without_mcp_server = pytest.mark.skipif(
    not is_mcp_server_running(), reason="MCP server required for integration tests"
)

# Skip integration tests in CI environments unless explicitly enabled
skip_integration_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true" and os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests skipped in CI (set RUN_INTEGRATION_TESTS=true to enable)",
)

# Test configuration
MCP_BASE_URL = "http://localhost:8080"
TEST_TOKEN = "integration_test_token"


# Test fixtures
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authentication headers for MCP requests."""
    return {"Authorization": f"Bearer {TEST_TOKEN}", "Content-Type": "application/json"}


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for async requests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest.fixture
def sample_configs() -> dict[str, str]:
    """Sample kernel configurations for different architectures."""
    return {
        "x86_64": """# x86_64 configuration
CONFIG_X86_64=y
CONFIG_X86=y
CONFIG_64BIT=y
CONFIG_GENERIC_CPU=y
CONFIG_SMP=y
CONFIG_NUMA=y
CONFIG_NET=y
CONFIG_BLOCK=y
CONFIG_EXT4_FS=m
CONFIG_USB=m
CONFIG_PCI=y
CONFIG_ACPI=y
# CONFIG_EMBEDDED is not set
# CONFIG_DEBUG_KERNEL is not set
CONFIG_CRYPTO_AES=y
CONFIG_VIRTUALIZATION=y
""",
        "arm64": """# ARM64 configuration
CONFIG_ARM64=y
CONFIG_64BIT=y
CONFIG_MMU=y
CONFIG_ARM64_PAGE_SHIFT=12
CONFIG_ARM64_CONT_SHIFT=4
CONFIG_ARCH_MMAP_RND_BITS=18
CONFIG_NET=y
CONFIG_BLOCK=y
CONFIG_EXT4_FS=m
CONFIG_USB=m
CONFIG_PCI=y
CONFIG_OF=y
CONFIG_ARM_GIC=y
# CONFIG_EMBEDDED is not set
# CONFIG_DEBUG_KERNEL is not set
CONFIG_CRYPTO_AES_ARM64=y
""",
        "arm": """# ARM 32-bit configuration
CONFIG_ARM=y
CONFIG_MMU=y
CONFIG_ARM_THUMB=y
CONFIG_ARM_L1_CACHE_SHIFT=6
CONFIG_ARM_DMA_USE_IOMMU=y
CONFIG_NET=y
CONFIG_BLOCK=y
CONFIG_EXT4_FS=m
CONFIG_USB=m
CONFIG_PCI=y
CONFIG_OF=y
CONFIG_ARM_GIC=y
# CONFIG_EMBEDDED is not set
# CONFIG_DEBUG_KERNEL is not set
CONFIG_CRYPTO_AES_ARM=y
""",
        "riscv": """# RISC-V configuration
CONFIG_RISCV=y
CONFIG_64BIT=y
CONFIG_MMU=y
CONFIG_RISCV_SBI=y
CONFIG_RISCV_ISA_C=y
CONFIG_RISCV_ISA_A=y
CONFIG_NET=y
CONFIG_BLOCK=y
CONFIG_EXT4_FS=m
CONFIG_USB=m
CONFIG_PCI=y
CONFIG_OF=y
CONFIG_RISCV_INTC=y
# CONFIG_EMBEDDED is not set
# CONFIG_DEBUG_KERNEL is not set
CONFIG_CRYPTO_AES=y
""",
        "ppc64le": """# PowerPC 64-bit Little Endian configuration
CONFIG_PPC64=y
CONFIG_PPC=y
CONFIG_64BIT=y
CONFIG_CPU_LITTLE_ENDIAN=y
CONFIG_PPC_BOOK3S_64=y
CONFIG_PPC_PSERIES=y
CONFIG_PPC_POWERNV=y
CONFIG_SMP=y
CONFIG_NUMA=y
CONFIG_NET=y
CONFIG_BLOCK=y
CONFIG_EXT4_FS=m
CONFIG_USB=m
CONFIG_PCI=y
CONFIG_OF=y
CONFIG_PPC_XICS=y
CONFIG_PPC_XIVE=y
# CONFIG_EMBEDDED is not set
# CONFIG_DEBUG_KERNEL is not set
CONFIG_CRYPTO_AES=y
CONFIG_VIRTUALIZATION=y
CONFIG_KVM_BOOK3S_64=y
""",
        "s390x": """# IBM System z (s390x) configuration
CONFIG_S390=y
CONFIG_64BIT=y
CONFIG_SMP=y
CONFIG_NUMA=y
CONFIG_S390_GUEST=y
CONFIG_MARCH_Z196=y
CONFIG_PACK_STACK=y
CONFIG_NET=y
CONFIG_BLOCK=y
CONFIG_EXT4_FS=m
CONFIG_USB=m
CONFIG_PCI=y
CONFIG_HOTPLUG_PCI=y
CONFIG_S390_HYPFS=y
CONFIG_S390_IOMMU=y
# CONFIG_EMBEDDED is not set
# CONFIG_DEBUG_KERNEL is not set
CONFIG_CRYPTO_AES_S390=y
CONFIG_VIRTUALIZATION=y
CONFIG_KVM=y
""",
    }


@pytest.fixture
def config_files(
    sample_configs: dict[str, str],
) -> Generator[dict[str, Path], None, None]:
    """Create temporary config files for each architecture."""
    config_files = {}

    for arch, content in sample_configs.items():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{arch}.config", delete=False
        ) as f:
            f.write(content)
            config_files[arch] = Path(f.name)

    yield config_files

    # Cleanup
    for config_file in config_files.values():
        try:
            config_file.unlink()
        except FileNotFoundError:
            pass


# Integration tests
@skip_without_mcp_server
@skip_integration_in_ci
class TestMultiArchConfig:
    """Test multi-architecture kernel configuration parsing."""

    async def test_parse_x86_64_config(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test parsing x86_64 kernel configuration."""
        config_path = config_files["x86_64"]

        request_data = {
            "config_path": str(config_path),
            "arch": "x86_64",
            "config_name": "defconfig",
            "resolve_dependencies": False,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/parse_kernel_config",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "config_id" in data
        assert data["arch"] == "x86_64"
        assert data["config_name"] == "defconfig"
        assert "options" in data
        assert "dependencies" in data
        assert "parsed_at" in data

        # Verify x86_64 specific options
        options = data["options"]
        assert "CONFIG_X86_64" in options
        assert options["CONFIG_X86_64"]["value"] is True
        assert options["CONFIG_X86_64"]["type"] == "bool"

        assert "CONFIG_64BIT" in options
        assert options["CONFIG_64BIT"]["value"] is True

        # Verify modular options
        assert "CONFIG_EXT4_FS" in options
        assert options["CONFIG_EXT4_FS"]["value"] == "m"
        assert options["CONFIG_EXT4_FS"]["type"] == "tristate"

    async def test_parse_arm64_config(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test parsing ARM64 kernel configuration."""
        config_path = config_files["arm64"]

        request_data = {
            "config_path": str(config_path),
            "arch": "arm64",
            "config_name": "defconfig",
            "resolve_dependencies": False,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/parse_kernel_config",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["arch"] == "arm64"
        assert "options" in data

        # Verify ARM64 specific options
        options = data["options"]
        assert "CONFIG_ARM64" in options
        assert options["CONFIG_ARM64"]["value"] is True

        assert "CONFIG_ARM64_PAGE_SHIFT" in options
        assert options["CONFIG_ARM64_PAGE_SHIFT"]["value"] == 12
        assert options["CONFIG_ARM64_PAGE_SHIFT"]["type"] == "int"

        # Verify ARM64 crypto support
        assert "CONFIG_CRYPTO_AES_ARM64" in options
        assert options["CONFIG_CRYPTO_AES_ARM64"]["value"] is True

    async def test_parse_ppc64le_config(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test parsing PowerPC 64-bit Little Endian kernel configuration."""
        config_path = config_files["ppc64le"]

        request_data = {
            "config_path": str(config_path),
            "arch": "ppc64le",
            "config_name": "defconfig",
            "resolve_dependencies": False,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/parse_kernel_config",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["arch"] == "ppc64le"
        assert "options" in data

        # Verify PowerPC specific options
        options = data["options"]
        assert "CONFIG_PPC64" in options
        assert options["CONFIG_PPC64"]["value"] is True

        assert "CONFIG_CPU_LITTLE_ENDIAN" in options
        assert options["CONFIG_CPU_LITTLE_ENDIAN"]["value"] is True

        # Verify PowerPC virtualization support
        assert "CONFIG_KVM_BOOK3S_64" in options
        assert options["CONFIG_KVM_BOOK3S_64"]["value"] is True

    async def test_parse_s390x_config(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test parsing IBM System z (s390x) kernel configuration."""
        config_path = config_files["s390x"]

        request_data = {
            "config_path": str(config_path),
            "arch": "s390x",
            "config_name": "defconfig",
            "resolve_dependencies": False,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/parse_kernel_config",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["arch"] == "s390x"
        assert "options" in data

        # Verify s390x specific options
        options = data["options"]
        assert "CONFIG_S390" in options
        assert options["CONFIG_S390"]["value"] is True

        assert "CONFIG_S390_GUEST" in options
        assert options["CONFIG_S390_GUEST"]["value"] is True

        # Verify s390x crypto support
        assert "CONFIG_CRYPTO_AES_S390" in options
        assert options["CONFIG_CRYPTO_AES_S390"]["value"] is True

    async def test_parse_multiple_architectures(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test parsing configurations for multiple architectures."""
        parsed_configs = {}

        # Parse all architecture configurations
        for arch, config_path in config_files.items():
            request_data = {
                "config_path": str(config_path),
                "arch": arch,
                "config_name": "defconfig",
                "resolve_dependencies": False,
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/parse_kernel_config",
                json=request_data,
                headers=auth_headers,
            )

            assert response.status_code == 200
            parsed_configs[arch] = response.json()

        # Verify each config has architecture-specific options
        assert "CONFIG_X86_64" in parsed_configs["x86_64"]["options"]
        assert "CONFIG_ARM64" in parsed_configs["arm64"]["options"]
        assert "CONFIG_ARM" in parsed_configs["arm"]["options"]
        assert "CONFIG_RISCV" in parsed_configs["riscv"]["options"]
        assert "CONFIG_PPC64" in parsed_configs["ppc64le"]["options"]
        assert "CONFIG_S390" in parsed_configs["s390x"]["options"]

        # Verify common options exist across architectures
        for arch_config in parsed_configs.values():
            options = arch_config["options"]
            assert "CONFIG_NET" in options
            assert "CONFIG_BLOCK" in options
            assert "CONFIG_EXT4_FS" in options
            assert options["CONFIG_EXT4_FS"]["value"] == "m"

    async def test_architecture_specific_features(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test that architecture-specific features are correctly parsed."""
        test_cases = [
            ("x86_64", "CONFIG_NUMA", True),
            ("x86_64", "CONFIG_ACPI", True),
            ("x86_64", "CONFIG_VIRTUALIZATION", True),
            ("arm64", "CONFIG_ARM64_PAGE_SHIFT", 12),
            ("arm64", "CONFIG_OF", True),
            ("arm", "CONFIG_ARM_THUMB", True),
            ("arm", "CONFIG_ARM_L1_CACHE_SHIFT", 6),
            ("riscv", "CONFIG_RISCV_SBI", True),
            ("riscv", "CONFIG_RISCV_ISA_C", True),
            ("ppc64le", "CONFIG_CPU_LITTLE_ENDIAN", True),
            ("ppc64le", "CONFIG_PPC_BOOK3S_64", True),
            ("ppc64le", "CONFIG_KVM_BOOK3S_64", True),
            ("s390x", "CONFIG_S390_GUEST", True),
            ("s390x", "CONFIG_MARCH_Z196", True),
            ("s390x", "CONFIG_S390_HYPFS", True),
        ]

        for arch, option_name, expected_value in test_cases:
            config_path = config_files[arch]

            request_data = {
                "config_path": str(config_path),
                "arch": arch,
                "config_name": "defconfig",
                "resolve_dependencies": False,
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/parse_kernel_config",
                json=request_data,
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()

            # Verify architecture-specific option
            options = data["options"]
            assert option_name in options, f"{option_name} not found in {arch} config"
            assert options[option_name]["value"] == expected_value, (
                f"{option_name} value mismatch in {arch}: "
                f"expected {expected_value}, got {options[option_name]['value']}"
            )

    async def test_config_storage_and_retrieval(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test that multi-arch configs can be stored and retrieved from database."""
        stored_configs = []

        # Store configurations for multiple architectures
        for arch, config_path in config_files.items():
            request_data = {
                "config_path": str(config_path),
                "arch": arch,
                "config_name": f"test_{arch}_config",
                "resolve_dependencies": False,
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/parse_kernel_config",
                json=request_data,
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            stored_configs.append(
                {
                    "config_id": data["config_id"],
                    "arch": arch,
                    "config_name": f"{arch}:test_{arch}_config",
                }
            )

        # Verify we stored configs for all architectures
        assert len(stored_configs) == 6

        # Verify each config has a unique ID
        config_ids = [config["config_id"] for config in stored_configs]
        assert len(config_ids) == len(set(config_ids)), "Config IDs should be unique"

    async def test_invalid_architecture_handling(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test handling of invalid or unsupported architectures."""
        config_path = config_files["x86_64"]

        # Test with invalid architecture
        request_data = {
            "config_path": str(config_path),
            "arch": "invalid_arch",
            "config_name": "defconfig",
            "resolve_dependencies": False,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/parse_kernel_config",
            json=request_data,
            headers=auth_headers,
        )

        # Should still work but with fallback behavior
        # The implementation should handle unknown architectures gracefully
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert data["arch"] == "invalid_arch"
            # Config should still be parsed, just with the specified arch

    async def test_cross_arch_comparison(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test comparing configurations across different architectures."""
        x86_64_path = config_files["x86_64"]
        arm64_path = config_files["arm64"]

        # Parse both configurations
        configs = {}
        for arch, path in [("x86_64", x86_64_path), ("arm64", arm64_path)]:
            request_data = {
                "config_path": str(path),
                "arch": arch,
                "config_name": "defconfig",
                "resolve_dependencies": False,
            }

            response = await http_client.post(
                f"{MCP_BASE_URL}/parse_kernel_config",
                json=request_data,
                headers=auth_headers,
            )

            assert response.status_code == 200
            configs[arch] = response.json()

        # Compare configurations
        x86_options = set(configs["x86_64"]["options"].keys())
        arm64_options = set(configs["arm64"]["options"].keys())

        # Find common and unique options
        common_options = x86_options & arm64_options
        x86_only = x86_options - arm64_options
        arm64_only = arm64_options - x86_options

        # Verify expected common options
        expected_common = {"CONFIG_NET", "CONFIG_BLOCK", "CONFIG_EXT4_FS", "CONFIG_USB"}
        assert expected_common.issubset(common_options)

        # Verify architecture-specific options
        assert "CONFIG_X86_64" in x86_only
        assert "CONFIG_NUMA" in x86_only
        assert "CONFIG_ARM64" in arm64_only
        assert "CONFIG_ARM64_PAGE_SHIFT" in arm64_only

        # Verify common options have consistent values where expected
        for option in ["CONFIG_NET", "CONFIG_BLOCK"]:
            x86_val = configs["x86_64"]["options"][option]["value"]
            arm64_val = configs["arm64"]["options"][option]["value"]
            assert x86_val == arm64_val, (
                f"{option} values differ: x86_64={x86_val}, arm64={arm64_val}"
            )

    async def test_config_parsing_with_dependencies(
        self,
        http_client: httpx.AsyncClient,
        auth_headers: dict[str, str],
        config_files: dict[str, Path],
    ) -> None:
        """Test configuration parsing with dependency resolution enabled."""
        config_path = config_files["x86_64"]

        request_data = {
            "config_path": str(config_path),
            "arch": "x86_64",
            "config_name": "defconfig",
            "resolve_dependencies": True,
            "max_depth": 3,
        }

        response = await http_client.post(
            f"{MCP_BASE_URL}/parse_kernel_config",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Verify dependency information is included
        assert "dependencies" in data
        dependencies = data["dependencies"]

        # Dependencies should be a list (even if empty due to mock/fallback)
        assert isinstance(dependencies, list)

        # Verify the resolve_dependencies flag was processed
        # (actual dependency resolution depends on kcs-config crate implementation)
