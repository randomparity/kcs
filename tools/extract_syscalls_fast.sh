#!/bin/bash
# Fast syscall extractor using grep and sed
# Processes large JSON files efficiently

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_json> <output_json>"
    exit 1
fi

INPUT_JSON="$1"
OUTPUT_JSON="$2"

echo "Fast syscall extraction from: $INPUT_JSON"

# Extract syscall patterns using grep and sed
{
    echo "["

    # Find all sys_ functions and SYSCALL_DEFINE patterns
    timeout 60 grep -o '"name":"sys_[^"]*"' "$INPUT_JSON" 2>/dev/null | \
        sed 's/"name":"sys_\([^"]*\)"/{"name":"sys_\1","entry_type":"Syscall","file_path":"extracted_from_symbols","line_number":0,"signature":"sys_\1","description":"System call function: \1"}/' | \
        head -1000 | \
        sed '$!s/$/,/'

    echo "]"
} > "$OUTPUT_JSON"

# Count results
SYSCALL_COUNT=$(grep -c '"name":' "$OUTPUT_JSON" 2>/dev/null || echo "0")

echo "Found $SYSCALL_COUNT syscall entry points"
echo "Results written to: $OUTPUT_JSON"
