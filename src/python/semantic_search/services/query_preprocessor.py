"""
QueryPreprocessor for text normalization and enhancement.

Implements multi-stage preprocessing pipeline for semantic search queries
as specified in specs/008-semantic-search-engine/research.md:

1. Tokenization preserving code identifiers
2. Technical abbreviation expansion
3. Case normalization while preserving semantic boundaries
4. Query enrichment with domain-specific synonyms
"""

import re


class QueryPreprocessor:
    """Multi-stage query preprocessor for text normalization and enhancement."""

    def __init__(self) -> None:
        """Initialize query preprocessor with domain-specific knowledge."""
        # Technical abbreviation mapping for kernel/systems context
        self._abbreviations = {
            "mem": "memory",
            "alloc": "allocation allocate",
            "dealloc": "deallocate deallocation",
            "malloc": "memory allocation",
            "free": "memory free deallocation",
            "buf": "buffer",
            "ptr": "pointer",
            "addr": "address",
            "vm": "virtual memory",
            "fs": "filesystem",
            "proc": "process",
            "sched": "scheduler scheduling",
            "irq": "interrupt",
            "dma": "direct memory access",
            "pci": "peripheral component interconnect",
            "usb": "universal serial bus",
            "tcp": "transmission control protocol",
            "udp": "user datagram protocol",
            "ip": "internet protocol",
            "eth": "ethernet",
            "gpio": "general purpose input output",
            "i2c": "inter integrated circuit",
            "spi": "serial peripheral interface",
            "uart": "universal asynchronous receiver transmitter",
            "cpu": "central processing unit processor",
            "gpu": "graphics processing unit",
            "mmu": "memory management unit",
            "tlb": "translation lookaside buffer",
            "iommu": "input output memory management unit",
            "numa": "non uniform memory access",
            "smp": "symmetric multiprocessing",
            "rcu": "read copy update",
            "bh": "bottom half",
            "softirq": "software interrupt",
            "tasklet": "task",
            "workqueue": "work queue",
            "kthread": "kernel thread",
            "syscall": "system call",
            "vfs": "virtual file system",
            "bio": "block input output",
            "skb": "socket buffer",
            "netdev": "network device",
            "pdev": "platform device",
        }

        # Domain-specific synonyms for query enrichment
        self._synonyms = {
            "lock": ["mutex", "semaphore", "spinlock", "rwlock"],
            "security": ["vulnerability", "exploit", "cve", "overflow"],
            "error": ["bug", "fault", "panic", "oops", "warning"],
            "performance": ["optimization", "latency", "throughput", "bottleneck"],
            "driver": ["module", "device", "hardware"],
            "network": ["networking", "socket", "protocol", "packet"],
            "memory": ["heap", "stack", "cache", "swap", "page"],
            "file": ["inode", "dentry", "directory", "path"],
            "process": ["task", "thread", "pid", "scheduling"],
            "interrupt": ["irq", "handler", "bottom half", "softirq"],
        }

        # Code patterns to preserve (only actual code identifiers)
        self._code_patterns = [
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b",  # CamelCase (min 2 parts)
            r"\b[a-z]+(?:_[a-z]+)+\b",  # snake_case (min 2 parts)
            r"\b[A-Z]+(?:_[A-Z]+)+\b",  # CONSTANT_CASE (min 2 parts)
            r"\b0x[0-9a-fA-F]+\b",  # hex values
            r"\b\d+\.\d+\.\d+\b",  # version numbers
            r"\bCONFIG_[A-Z_]+\b",  # kernel config options
        ]

    def preprocess(self, query_text: str) -> str:
        """
        Preprocess query text using multi-stage pipeline.

        Implements preprocessing strategy from research.md:
        1. Tokenization preserving code identifiers
        2. Technical abbreviation expansion
        3. Case normalization with semantic preservation
        4. Query enrichment with domain synonyms

        Args:
            query_text: Original user query

        Returns:
            Enhanced and normalized query text
        """
        if not query_text or not query_text.strip():
            return ""

        # Check if this looks like a simple text query vs code-heavy query
        has_clear_code_identifiers = self._has_clear_code_identifiers(query_text)

        # Stage 1: Basic cleanup and normalization
        processed = self._basic_cleanup(query_text)

        if has_clear_code_identifiers:
            # Full sophisticated pipeline for code-aware queries
            # Stage 2: Preserve code identifiers before further processing
            code_tokens, processed = self._extract_code_tokens(processed)

            # Stage 3: Expand technical abbreviations
            processed = self._expand_abbreviations(processed)

            # Stage 4: Case normalization with semantic preservation
            processed = self._normalize_case(processed)

            # Stage 5: Query enrichment with synonyms
            processed = self._enrich_with_synonyms(processed)

            # Stage 6: Restore preserved code tokens
            processed = self._restore_code_tokens(processed, code_tokens)
        else:
            # Simple pipeline for regular text queries
            # Stage 3: Expand technical abbreviations
            processed = self._expand_abbreviations(processed)

            # Stage 4: Simple lowercasing for regular text
            processed = processed.lower()

            # Stage 5: Query enrichment with synonyms
            processed = self._enrich_with_synonyms(processed)

        # Stage 7: Final cleanup
        processed = self._final_cleanup(processed)

        return processed

    def _has_clear_code_identifiers(self, text: str) -> bool:
        """Check if the query contains clear code identifiers that should preserve case."""
        # Look for patterns that strongly suggest code content
        code_indicators = [
            r"\b[a-z]+_[a-z]+\b",  # snake_case with underscore
            r"\b[A-Z]+_[A-Z]+\b",  # CONSTANT_CASE with underscore
            r"\bCONFIG_[A-Z_]+\b",  # kernel configs
            r"\b0x[0-9a-fA-F]+\b",  # hex values
            r"\b\w*\(\)",  # function calls
            r"->[a-zA-Z_]",  # pointer dereference
            r"::[a-zA-Z_]",  # scope resolution
            r"\b\w+\.\w+\b",  # member access (but not version numbers)
        ]

        for pattern in code_indicators:
            if re.search(pattern, text):
                return True

        return False

    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleanup and normalization."""
        # Strip whitespace
        text = text.strip()

        # Normalize line breaks to spaces
        text = re.sub(r"\r?\n", " ", text)

        # Normalize multiple whitespace to single space
        text = re.sub(r"\s+", " ", text)

        # Remove excessive punctuation but preserve meaningful ones
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[.]{3,}", "...", text)

        return text

    def _extract_code_tokens(self, text: str) -> tuple[dict[str, str], str]:
        """Extract and preserve code identifiers."""
        code_tokens = {}
        token_counter = 0

        for pattern in self._code_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                token = match.group(0)
                placeholder = f"__CODE_TOKEN_{token_counter}__"
                code_tokens[placeholder] = token
                text = text.replace(token, placeholder, 1)
                token_counter += 1

        return code_tokens, text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand technical abbreviations to improve recall."""
        words = text.split()
        expanded_words = []

        for word in words:
            # Check if word is an abbreviation (case-insensitive)
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in self._abbreviations:
                # Add both original and expanded forms
                expansion = self._abbreviations[word_lower]
                expanded_words.append(f"{word} {expansion}")
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    def _normalize_case(self, text: str) -> str:
        """Normalize case while preserving semantic boundaries."""
        # Convert to lowercase but preserve certain patterns
        words = text.split()
        normalized_words = []

        for word in words:
            # Skip placeholder tokens
            if word.startswith("__CODE_TOKEN_"):
                normalized_words.append(word)
                continue

            # Check if word contains mixed case that should be preserved
            if self._has_semantic_case(word):
                # Keep original case for semantically meaningful patterns
                normalized_words.append(word)
            else:
                # Convert to lowercase
                normalized_words.append(word.lower())

        return " ".join(normalized_words)

    def _has_semantic_case(self, word: str) -> bool:
        """Check if word has semantically meaningful case patterns."""
        # Only preserve case for actual code identifiers, not general text

        # Check for CamelCase (at least 2 parts)
        if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]*)+$", word):
            return True

        # Check for CONSTANT_CASE with underscores (actual constants)
        if re.match(r"^[A-Z]+(?:_[A-Z]+)+$", word):
            return True

        # Check for mixed case technical terms with both cases
        if re.match(r"^[a-z]+[A-Z]", word):
            return True

        # Check for known code patterns like CONFIG_* or specific kernel constants
        if re.match(r"^CONFIG_[A-Z_]+$", word):
            return True

        # Single all-caps words are likely just emphasis, not code identifiers
        return False

    def _enrich_with_synonyms(self, text: str) -> str:
        """Enrich query with domain-specific synonyms."""
        words = text.split()
        enriched_words = []

        for word in words:
            # Skip placeholder tokens
            if word.startswith("__CODE_TOKEN_"):
                enriched_words.append(word)
                continue

            word_lower = word.lower().strip(".,!?;:")

            # Check if word has synonyms
            if word_lower in self._synonyms:
                synonyms = self._synonyms[word_lower]
                # Add original word plus key synonyms (limit to avoid query explosion)
                enriched_words.append(f"{word} {' '.join(synonyms[:2])}")
            else:
                enriched_words.append(word)

        return " ".join(enriched_words)

    def _restore_code_tokens(self, text: str, code_tokens: dict[str, str]) -> str:
        """Restore preserved code tokens."""
        for placeholder, original_token in code_tokens.items():
            text = text.replace(placeholder, original_token)
        return text

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup of processed text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove empty parentheses or brackets that might be left
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r"\{\s*\}", "", text)

        # Final whitespace cleanup
        text = re.sub(r"\s+", " ", text).strip()

        return text
