# KCS Project Makefile
# Provides targets for development workflow automation

.PHONY: help setup clean lint format test check build docs docker
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON_VENV := $(VENV_BIN)/python
PIP_VENV := $(VENV_BIN)/pip
UV := uv

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)KCS Project Development Makefile$(NC)"
	@echo "$(BLUE)===================================$(NC)"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  $(YELLOW)make setup$(NC)     - Initial project setup"
	@echo "  $(YELLOW)make check$(NC)     - Run all quality checks"
	@echo "  $(YELLOW)make test$(NC)      - Run test suite"
	@echo "  $(YELLOW)make format$(NC)    - Format all code"

# Setup targets
setup: ## Setup development environment (venv, dependencies, hooks)
	@echo "$(BLUE)Setting up KCS development environment...$(NC)"
	@$(MAKE) venv
	@$(MAKE) install-deps
	@$(MAKE) install-hooks
	@$(MAKE) build-rust
	@echo "$(GREEN)✅ Setup complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Activate virtual environment: $(YELLOW)source $(VENV_DIR)/bin/activate$(NC)"
	@echo "  2. Run tests: $(YELLOW)make test$(NC)"
	@echo "  3. Check code quality: $(YELLOW)make check$(NC)"

venv: ## Create Python virtual environment
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(BLUE)Creating virtual environment...$(NC)"; \
		$(UV) venv $(VENV_DIR); \
	else \
		echo "$(GREEN)Virtual environment already exists$(NC)"; \
	fi

install-deps: venv ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@$(UV) pip install --upgrade pip
	@$(UV) pip install -e ".[dev,performance]"
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

install-hooks: venv ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	@$(VENV_BIN)/pre-commit install --hook-type pre-commit --hook-type commit-msg
	@echo "$(GREEN)✅ Pre-commit hooks installed$(NC)"

build-rust: ## Build Rust components
	@echo "$(BLUE)Building Rust components...$(NC)"
	@if [ -d "src/rust" ]; then \
		cd src/rust && cargo build --release --workspace; \
		echo "$(GREEN)✅ Rust components built$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No Rust source found, skipping$(NC)"; \
	fi

# Code quality targets
check: lint test ## Run all quality checks (lint + test)
	@echo "$(GREEN)✅ All quality checks passed!$(NC)"

lint: ## Run linting tools (ruff, mypy)
	@echo "$(BLUE)Running linting checks...$(NC)"
	@$(MAKE) lint-python
	@$(MAKE) lint-rust
	@$(MAKE) lint-yaml
	@$(MAKE) lint-sql
	@echo "$(GREEN)✅ Linting complete$(NC)"

lint-python: venv ## Run Python linting (ruff, mypy)
	@echo "$(BLUE)Linting Python code...$(NC)"
	@$(VENV_BIN)/ruff check src/python/ tests/ tools/ || (echo "$(RED)❌ Ruff found issues$(NC)" && exit 1)
	@$(VENV_BIN)/mypy src/python/ || (echo "$(RED)❌ MyPy found issues$(NC)" && exit 1)
	@echo "$(GREEN)✅ Python linting passed$(NC)"

lint-rust: ## Run Rust linting (clippy)
	@echo "$(BLUE)Linting Rust code...$(NC)"
	@if [ -d "src/rust" ]; then \
		cd src/rust && cargo clippy --all-targets --all-features -- -D warnings || \
		(echo "$(RED)❌ Clippy found issues$(NC)" && exit 1); \
		echo "$(GREEN)✅ Rust linting passed$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No Rust source found, skipping$(NC)"; \
	fi

lint-yaml: venv ## Check YAML files
	@echo "$(BLUE)Checking YAML files...$(NC)"
	@$(VENV_BIN)/pre-commit run check-yaml --all-files || (echo "$(RED)❌ YAML check failed$(NC)" && exit 1)
	@echo "$(GREEN)✅ YAML files valid$(NC)"

lint-sql: venv ## Lint SQL files
	@echo "$(BLUE)Linting SQL files...$(NC)"
	@if find . -name "*.sql" -type f | head -1 | grep -q .; then \
		$(VENV_BIN)/sqlfluff lint src/sql/ || (echo "$(RED)❌ SQL linting found issues$(NC)" && exit 1); \
		echo "$(GREEN)✅ SQL linting passed$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No SQL files found, skipping$(NC)"; \
	fi

format: ## Format all code (Python, Rust, SQL)
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(MAKE) format-python
	@$(MAKE) format-rust
	@$(MAKE) format-sql
	@echo "$(GREEN)✅ Code formatting complete$(NC)"

format-python: venv ## Format Python code (black, ruff)
	@echo "$(BLUE)Formatting Python code...$(NC)"
	@$(VENV_BIN)/black src/python/ tests/ tools/
	@$(VENV_BIN)/ruff check --fix src/python/ tests/ tools/
	@echo "$(GREEN)✅ Python code formatted$(NC)"

format-rust: ## Format Rust code (rustfmt)
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	@if [ -d "src/rust" ]; then \
		cd src/rust && cargo fmt --all; \
		echo "$(GREEN)✅ Rust code formatted$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No Rust source found, skipping$(NC)"; \
	fi

format-sql: venv ## Format SQL files
	@echo "$(BLUE)Formatting SQL files...$(NC)"
	@if find . -name "*.sql" -type f | head -1 | grep -q .; then \
		$(VENV_BIN)/sqlfluff fix src/sql/ --force; \
		echo "$(GREEN)✅ SQL files formatted$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No SQL files found, skipping$(NC)"; \
	fi

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running test suite...$(NC)"
	@$(MAKE) test-unit
	@$(MAKE) test-integration
	@echo "$(GREEN)✅ All tests passed!$(NC)"

test-unit: venv ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(VENV_BIN)/pytest tests/unit/ -v --cov=src/python --cov-report=term-missing
	@echo "$(GREEN)✅ Unit tests passed$(NC)"

test-integration: venv ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(VENV_BIN)/pytest tests/integration/ -v --tb=short
	@echo "$(GREEN)✅ Integration tests passed$(NC)"

test-contract: venv ## Run contract tests
	@echo "$(BLUE)Running contract tests...$(NC)"
	@$(VENV_BIN)/pytest tests/contract/ -v
	@echo "$(GREEN)✅ Contract tests passed$(NC)"

test-performance: venv ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	@$(VENV_BIN)/pytest tests/performance/ -v -m performance
	@echo "$(GREEN)✅ Performance tests passed$(NC)"

test-system: venv ## Run full system test
	@echo "$(BLUE)Running full system test...$(NC)"
	@$(PYTHON_VENV) tools/run_system_test.py
	@echo "$(GREEN)✅ System test passed$(NC)"

test-watch: venv ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode (Ctrl+C to stop)...$(NC)"
	@$(VENV_BIN)/pytest-watch tests/ --verbose

coverage: venv ## Generate test coverage report
	@echo "$(BLUE)Generating coverage report...$(NC)"
	@$(VENV_BIN)/pytest tests/ --cov=src/python --cov-report=html --cov-report=term
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

# Build targets
build: ## Build all components
	@echo "$(BLUE)Building KCS...$(NC)"
	@$(MAKE) build-rust
	@$(MAKE) build-python
	@echo "$(GREEN)✅ Build complete$(NC)"

build-python: venv ## Build Python package
	@echo "$(BLUE)Building Python package...$(NC)"
	@$(VENV_BIN)/python -m build
	@echo "$(GREEN)✅ Python package built$(NC)"

build-docs: venv ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@$(PYTHON_VENV) tools/generate_docs.py specs/001-kernel-context-server/contracts/mcp-api.yaml -o docs/api
	@echo "$(GREEN)✅ Documentation built in docs/api/$(NC)"

# Benchmarking targets
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	@if [ -d "src/rust" ]; then \
		cd src/rust && cargo bench; \
		echo "$(GREEN)✅ Rust benchmarks complete$(NC)"; \
	fi
	@$(PYTHON_VENV) tools/performance_optimization.py --analyze
	@echo "$(GREEN)✅ Benchmarks complete$(NC)"

benchmark-k6: venv ## Run k6 load tests
	@echo "$(BLUE)Running k6 load tests...$(NC)"
	@if command -v k6 >/dev/null 2>&1; then \
		k6 run tests/performance/mcp_load.js; \
		echo "$(GREEN)✅ k6 load tests complete$(NC)"; \
	else \
		echo "$(RED)❌ k6 not installed. Install with: brew install k6$(NC)"; \
		exit 1; \
	fi

# Security targets
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	@$(MAKE) security-deps
	@$(MAKE) security-secrets
	@echo "$(GREEN)✅ Security checks passed$(NC)"

security-deps: venv ## Check for vulnerable dependencies
	@echo "$(BLUE)Checking for vulnerable dependencies...$(NC)"
	@$(VENV_BIN)/safety check || (echo "$(RED)❌ Vulnerable dependencies found$(NC)" && exit 1)
	@echo "$(GREEN)✅ No vulnerable dependencies found$(NC)"

security-secrets: venv ## Check for secrets in code
	@echo "$(BLUE)Scanning for secrets...$(NC)"
	@$(VENV_BIN)/detect-secrets scan --all-files --baseline .secrets.baseline
	@echo "$(GREEN)✅ No secrets detected$(NC)"

# Git targets
pre-commit: venv ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	@$(VENV_BIN)/pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit hooks passed$(NC)"

git-hooks: install-hooks ## Alias for install-hooks

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t kcs:latest .
	@echo "$(GREEN)✅ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	@docker run -p 8080:8080 --env-file .env kcs:latest

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started$(NC)"

docker-compose-down: ## Stop services with docker-compose
	@echo "$(BLUE)Stopping services with docker-compose...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped$(NC)"

# Database targets
db-start: ## Start PostgreSQL database
	@echo "$(BLUE)Starting PostgreSQL database...$(NC)"
	@docker run -d --name kcs-postgres \
		-e POSTGRES_DB=kcs \
		-e POSTGRES_USER=kcs \
		-e POSTGRES_PASSWORD=kcs \
		-p 5432:5432 \
		pgvector/pgvector:pg15
	@echo "$(GREEN)✅ Database started$(NC)"

db-stop: ## Stop PostgreSQL database
	@echo "$(BLUE)Stopping PostgreSQL database...$(NC)"
	@docker stop kcs-postgres || true
	@docker rm kcs-postgres || true
	@echo "$(GREEN)✅ Database stopped$(NC)"

db-migrate: venv ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@$(PYTHON_VENV) tools/setup/migrate.sh
	@echo "$(GREEN)✅ Migrations complete$(NC)"

# Utility targets
clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@if [ -d "src/rust" ]; then cd src/rust && cargo clean; fi
	@echo "$(GREEN)✅ Clean complete$(NC)"

clean-all: clean ## Clean everything including venv
	@echo "$(BLUE)Cleaning everything...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✅ Deep clean complete$(NC)"

install: setup ## Alias for setup

reinstall: clean-all setup ## Clean and reinstall everything

update-deps: venv ## Update dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@$(UV) pip install --upgrade pip
	@$(UV) pip install --upgrade -e ".[dev,performance]"
	@$(VENV_BIN)/pre-commit autoupdate
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

# Analysis targets
analyze: ## Run all analysis tools
	@echo "$(BLUE)Running analysis tools...$(NC)"
	@$(MAKE) lint
	@$(MAKE) security
	@$(MAKE) benchmark
	@echo "$(GREEN)✅ Analysis complete$(NC)"

profile: venv ## Profile Python code performance
	@echo "$(BLUE)Profiling Python code...$(NC)"
	@$(PYTHON_VENV) -m cProfile -o profile.stats tools/run_system_test.py
	@$(PYTHON_VENV) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)✅ Profiling complete$(NC)"

# CI targets
ci: ## Run CI pipeline matching pre-commit hooks exactly (skips Rust checks like CI)
	@echo "$(BLUE)Running CI pipeline (matching pre-commit hooks)...$(NC)"
	@$(MAKE) ci-ruff
	@$(MAKE) ci-mypy
	@$(MAKE) ci-file-checks
	@$(MAKE) ci-markdown
	@$(MAKE) ci-config-validation
	@$(MAKE) ci-pytest
	@echo "$(GREEN)✅ CI pipeline passed (all pre-commit hooks)$(NC)"

ci-local: ## Run full CI pipeline including local-only Rust checks
	@echo "$(BLUE)Running full local CI pipeline...$(NC)"
	@$(MAKE) ci-ruff
	@$(MAKE) ci-mypy
	@$(MAKE) ci-rust
	@$(MAKE) ci-file-checks
	@$(MAKE) ci-markdown
	@$(MAKE) ci-config-validation
	@$(MAKE) ci-pytest
	@echo "$(GREEN)✅ Full local CI pipeline passed$(NC)"

# Mini-kernel testing
test-mini-kernel: venv ## Run tests with mini-kernel fixture
	@echo "$(BLUE)Running mini-kernel tests...$(NC)"
	@$(VENV_BIN)/pytest tests/integration/test_mini_kernel.py -v
	@echo "$(GREEN)✅ Mini-kernel tests passed$(NC)"

test-mini-kernel-fast: venv ## Run fast mini-kernel tests only
	@echo "$(BLUE)Running fast mini-kernel tests...$(NC)"
	@$(VENV_BIN)/pytest tests/integration/test_mini_kernel.py -v -m "not performance"
	@echo "$(GREEN)✅ Fast mini-kernel tests passed$(NC)"

ci-ruff: venv ## Run ruff linting and formatting (matches pre-commit)
	@echo "$(BLUE)Running ruff lint and fix...$(NC)"
	@$(VENV_BIN)/ruff check src/python/ tests/ --fix
	@echo "$(BLUE)Running ruff format...$(NC)"
	@$(VENV_BIN)/ruff format src/python/ tests/

ci-mypy: venv ## Run mypy type checking (matches pre-commit)
	@echo "$(BLUE)Running mypy type checking...$(NC)"
	@$(VENV_BIN)/mypy src/python/

ci-rust: ## Run Rust formatting and linting (matches pre-commit)
	@echo "$(BLUE)Running Rust checks...$(NC)"
	@if [ -d "src/rust" ]; then \
		echo "$(BLUE)Running cargo fmt...$(NC)"; \
		cd src/rust && cargo fmt --all --; \
		echo "$(BLUE)Running cargo clippy...$(NC)"; \
		cd src/rust && cargo clippy --all-targets --all-features -- -D warnings; \
		echo "$(BLUE)Running cargo check...$(NC)"; \
		cd src/rust && cargo check --all-targets; \
	else \
		echo "$(YELLOW)⚠️  No Rust source found, skipping$(NC)"; \
	fi

ci-file-checks: venv ## Run general file checks (matches pre-commit)
	@echo "$(BLUE)Running file checks...$(NC)"
	@$(VENV_BIN)/pre-commit run trailing-whitespace --all-files
	@$(VENV_BIN)/pre-commit run end-of-file-fixer --all-files
	@$(VENV_BIN)/pre-commit run check-yaml --all-files
	@$(VENV_BIN)/pre-commit run check-toml --all-files
	@$(VENV_BIN)/pre-commit run check-json --all-files
	@$(VENV_BIN)/pre-commit run check-merge-conflict --all-files
	@$(VENV_BIN)/pre-commit run check-added-large-files --all-files
	@$(VENV_BIN)/pre-commit run detect-private-key --all-files

ci-markdown: venv ## Run markdown linting (matches pre-commit)
	@echo "$(BLUE)Running markdownlint...$(NC)"
	@$(VENV_BIN)/pre-commit run markdownlint --all-files

ci-config-validation: ## Run configuration validation (matches pre-commit)
	@echo "$(BLUE)Running configuration validation...$(NC)"
	@if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then \
		echo "$(BLUE)Checking docker-compose config...$(NC)"; \
		docker compose config >/dev/null; \
	fi

ci-pytest: venv ## Run pytest with coverage (matches pre-push hook)
	@echo "$(BLUE)Running pytest with coverage...$(NC)"
	@$(VENV_BIN)/pytest tests/ -v --cov=src/python --cov-report=xml

ci-setup: ## Setup for CI environment
	@echo "$(BLUE)Setting up CI environment...$(NC)"
	@$(MAKE) venv
	@$(MAKE) install-deps
	@$(MAKE) build-rust
	@echo "$(GREEN)✅ CI setup complete$(NC)"

# Development helpers
dev: ## Start development server
	@echo "$(BLUE)Starting development server...$(NC)"
	@$(VENV_BIN)/uvicorn kcs_mcp.app:app --reload --host 0.0.0.0 --port 8080

dev-tools: ## Install additional development tools
	@echo "$(BLUE)Installing development tools...$(NC)"
	@$(UV) pip install ipython jupyter notebook ipdb
	@echo "$(GREEN)✅ Development tools installed$(NC)"

shell: venv ## Start interactive Python shell
	@$(VENV_BIN)/python

notebook: venv ## Start Jupyter notebook
	@$(VENV_BIN)/jupyter notebook

# Information targets
info: ## Show project information
	@echo "$(BLUE)KCS Project Information$(NC)"
	@echo "======================="
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Virtual env: $(VENV_DIR)"
	@echo "UV version: $(shell $(UV) --version 2>/dev/null || echo 'not installed')"
	@if [ -d "src/rust" ]; then echo "Rust version: $(shell rustc --version 2>/dev/null || echo 'not installed')"; fi
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'not a git repo')"
	@echo "Git status: $(shell git status --porcelain 2>/dev/null | wc -l | tr -d ' ') files changed"

version: ## Show version information
	@echo "$(BLUE)Version Information$(NC)"
	@echo "==================="
	@grep -E "^version" pyproject.toml || echo "Version not found in pyproject.toml"

# Quick aliases
l: lint           ## Quick alias for lint
f: format         ## Quick alias for format
t: test           ## Quick alias for test
c: check          ## Quick alias for check
s: setup          ## Quick alias for setup

# Make sure .venv is created with the right Python version
$(VENV_DIR)/pyvenv.cfg:
	$(UV) venv $(VENV_DIR)

# Ensure commands run in the virtual environment
$(VENV_BIN)/%: $(VENV_DIR)/pyvenv.cfg
	@true
