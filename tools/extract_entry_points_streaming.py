#!/usr/bin/env python3
"""
Streaming entry point extractor for large JSON files.
Processes kernel symbols JSON in chunks to avoid memory issues.
Multi-threaded for better performance on large datasets.
Supports pattern detection and multiple entry point types.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

# Global progress tracking
progress_lock = Lock()
total_entry_points = 0
processed_files = 0

def extract_syscalls_from_chunk(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract syscall entry points from a single file's data."""
    entry_points = []

    file_path = file_data.get("path", "")
    symbols = file_data.get("symbols", [])

    # Patterns to match syscall function names (same logic as Rust extractor)
    sys_pattern = re.compile(r"^(?:__se_sys_|__do_sys_|sys_)(\w+)$")
    ksys_pattern = re.compile(r"^ksys_(\w+)$")

    for symbol in symbols:
        name = symbol.get("name", "")
        signature = symbol.get("signature", "")
        kind = symbol.get("kind", "")
        start_line = symbol.get("start_line", 0)
        metadata = symbol.get("metadata", {})

        if kind != "Function":
            continue

        # Check for syscall function name patterns
        sys_match = sys_pattern.match(name)
        if sys_match:
            syscall_name = sys_match.group(1)
            entry_points.append({
                "name": f"sys_{syscall_name}",
                "entry_type": "Syscall",
                "file_path": file_path,
                "line_number": start_line,
                "signature": signature,
                "description": f"System call: {syscall_name}",
                "metadata": metadata
            })
            continue

        # Check for ksys_ helper functions
        ksys_match = ksys_pattern.match(name)
        if ksys_match:
            syscall_name = ksys_match.group(1)
            entry_points.append({
                "name": f"ksys_{syscall_name}",
                "entry_type": "Syscall",
                "file_path": file_path,
                "line_number": start_line,
                "signature": signature,
                "description": f"Kernel syscall helper: {syscall_name}",
                "metadata": metadata
            })
            continue

        # Check for direct sys_ function names (but avoid double-counting)
        if name.startswith("sys_") and not name.startswith(("__se_sys_", "__do_sys_")):
            syscall_name = name[4:]  # Remove 'sys_' prefix
            entry_points.append({
                "name": name,
                "entry_type": "Syscall",
                "file_path": file_path,
                "line_number": start_line,
                "signature": signature,
                "description": f"Direct syscall: {syscall_name}",
                "metadata": metadata
            })

    return entry_points


def extract_ioctls_from_chunk(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ioctl entry points from a single file's data."""
    entry_points = []

    file_path = file_data.get("path", "")

    # Skip non-source files
    if not file_path.endswith(('.c', '.h')):
        return entry_points

    # Look for ioctl handlers in file operations structures
    # This is a simplified version - the Rust implementation is more comprehensive
    content = file_data.get("content", "")
    if content and "unlocked_ioctl" in content:
        # Extract line number if available
        for i, line in enumerate(content.split('\n'), 1):
            if "unlocked_ioctl" in line and "=" in line:
                # Try to extract the handler name
                match = re.search(r'\.unlocked_ioctl\s*=\s*(\w+)', line)
                if match:
                    handler_name = match.group(1)
                    entry_points.append({
                        "name": handler_name,
                        "entry_type": "Ioctl",
                        "file_path": file_path,
                        "line_number": i,
                        "signature": f"long {handler_name}(struct file *, unsigned int, unsigned long)",
                        "description": f"Ioctl handler: {handler_name}",
                        "metadata": {"handler_type": "unlocked_ioctl"}
                    })

    return entry_points


def extract_procfs_from_chunk(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract procfs entry points from a single file's data."""
    entry_points = []

    file_path = file_data.get("path", "")
    content = file_data.get("content", "")

    if not content:
        return entry_points

    # Pattern for proc_create and related functions
    proc_create_pattern = re.compile(
        r'proc_create(?:_data)?\s*\(\s*"([^"]+)"\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*(\w+)'
    )

    for i, line in enumerate(content.split('\n'), 1):
        match = proc_create_pattern.search(line)
        if match:
            proc_name = match.group(1)
            ops_name = match.group(2)
            entry_points.append({
                "name": f"/proc/{proc_name}",
                "entry_type": "ProcFs",
                "file_path": file_path,
                "line_number": i,
                "signature": f"proc_create(\"{proc_name}\", ..., {ops_name})",
                "description": f"Procfs entry: /proc/{proc_name}",
                "metadata": {"proc_ops": ops_name}
            })

    return entry_points


def extract_debugfs_from_chunk(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract debugfs entry points from a single file's data."""
    entry_points = []

    file_path = file_data.get("path", "")
    content = file_data.get("content", "")

    if not content:
        return entry_points

    # Pattern for debugfs_create functions
    debugfs_pattern = re.compile(
        r'debugfs_create_\w+\s*\(\s*"([^"]+)"'
    )

    for i, line in enumerate(content.split('\n'), 1):
        match = debugfs_pattern.search(line)
        if match:
            debug_name = match.group(1)
            entry_points.append({
                "name": f"/sys/kernel/debug/{debug_name}",
                "entry_type": "DebugFs",
                "file_path": file_path,
                "line_number": i,
                "signature": line.strip(),
                "description": f"Debugfs entry: {debug_name}",
                "metadata": {}
            })

    return entry_points


def extract_netlink_from_chunk(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract netlink handler entry points from a single file's data."""
    entry_points = []

    file_path = file_data.get("path", "")
    content = file_data.get("content", "")

    if not content:
        return entry_points

    # Pattern for netlink_kernel_create
    netlink_pattern = re.compile(
        r'netlink_kernel_create\s*\([^,]+,\s*(\w+)'
    )

    for i, line in enumerate(content.split('\n'), 1):
        match = netlink_pattern.search(line)
        if match:
            protocol = match.group(1)
            entry_points.append({
                "name": f"netlink_{protocol}",
                "entry_type": "Netlink",
                "file_path": file_path,
                "line_number": i,
                "signature": line.strip(),
                "description": f"Netlink handler for protocol: {protocol}",
                "metadata": {"protocol": protocol}
            })

    return entry_points


def extract_interrupts_from_chunk(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract interrupt handler entry points from a single file's data."""
    entry_points = []

    file_path = file_data.get("path", "")
    content = file_data.get("content", "")

    if not content:
        return entry_points

    # Patterns for request_irq and related functions
    irq_patterns = [
        re.compile(r'request_(?:threaded_)?irq\s*\([^,]+,\s*(\w+)'),
        re.compile(r'devm_request_(?:threaded_)?irq\s*\([^,]+,[^,]+,\s*(\w+)'),
    ]

    for i, line in enumerate(content.split('\n'), 1):
        for pattern in irq_patterns:
            match = pattern.search(line)
            if match:
                handler_name = match.group(1)
                entry_points.append({
                    "name": handler_name,
                    "entry_type": "Interrupt",
                    "file_path": file_path,
                    "line_number": i,
                    "signature": f"irqreturn_t {handler_name}(int, void *)",
                    "description": f"Interrupt handler: {handler_name}",
                    "metadata": {}
                })
                break

    return entry_points


def detect_kernel_patterns(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect kernel-specific patterns like EXPORT_SYMBOL, module_param, etc."""
    patterns = []

    file_path = file_data.get("path", "")
    content = file_data.get("content", "")

    if not content:
        return patterns

    # EXPORT_SYMBOL patterns
    export_patterns = [
        (re.compile(r'\bEXPORT_SYMBOL\s*\(\s*(\w+)\s*\)'), "ExportSymbol", "EXPORT_SYMBOL"),
        (re.compile(r'\bEXPORT_SYMBOL_GPL\s*\(\s*(\w+)\s*\)'), "ExportSymbolGPL", "EXPORT_SYMBOL_GPL"),
        (re.compile(r'\bEXPORT_SYMBOL_NS\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'), "ExportSymbolNS", "EXPORT_SYMBOL_NS"),
    ]

    # module_param patterns
    param_patterns = [
        (re.compile(r'\bmodule_param\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)'), "ModuleParam", "module_param"),
        (re.compile(r'\bmodule_param_array\s*\(\s*(\w+)'), "ModuleParamArray", "module_param_array"),
    ]

    # Boot parameter patterns
    boot_patterns = [
        (re.compile(r'\b__setup\s*\(\s*"([^"]+)"\s*,\s*(\w+)\s*\)'), "BootParam", "__setup"),
        (re.compile(r'\bearly_param\s*\(\s*"([^"]+)"\s*,\s*(\w+)\s*\)'), "EarlyParam", "early_param"),
    ]

    for i, line in enumerate(content.split('\n'), 1):
        # Check export symbols
        for pattern, pattern_type, export_type in export_patterns:
            match = pattern.search(line)
            if match:
                symbol_name = match.group(1)
                metadata = {"export_type": export_type}
                if "NS" in export_type and match.lastindex > 1:
                    metadata["namespace"] = match.group(2)

                patterns.append({
                    "pattern_type": pattern_type,
                    "name": symbol_name,
                    "file_path": file_path,
                    "line_number": i,
                    "raw_text": line.strip(),
                    "metadata": metadata
                })

        # Check module params
        for pattern, pattern_type, param_type in param_patterns:
            match = pattern.search(line)
            if match:
                param_name = match.group(1)
                metadata = {"param_type": param_type}
                if match.lastindex > 1:
                    metadata["type"] = match.group(2)
                if match.lastindex > 2:
                    metadata["perm"] = match.group(3)

                patterns.append({
                    "pattern_type": pattern_type,
                    "name": param_name,
                    "file_path": file_path,
                    "line_number": i,
                    "raw_text": line.strip(),
                    "metadata": metadata
                })

        # Check boot params
        for pattern, pattern_type, boot_type in boot_patterns:
            match = pattern.search(line)
            if match:
                param_name = match.group(1)
                handler_name = match.group(2)

                patterns.append({
                    "pattern_type": pattern_type,
                    "name": param_name,
                    "file_path": file_path,
                    "line_number": i,
                    "raw_text": line.strip(),
                    "metadata": {
                        "boot_type": boot_type,
                        "handler": handler_name
                    }
                })

    return patterns


def extract_entry_points_from_chunk(
    file_data: dict[str, Any],
    entry_types: set[str],
    enable_patterns: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract multiple types of entry points from a single file's data."""
    entry_points = []
    patterns = []

    # Extract requested entry point types
    if "syscalls" in entry_types:
        entry_points.extend(extract_syscalls_from_chunk(file_data))

    if "ioctls" in entry_types:
        entry_points.extend(extract_ioctls_from_chunk(file_data))

    if "procfs" in entry_types:
        entry_points.extend(extract_procfs_from_chunk(file_data))

    if "debugfs" in entry_types:
        entry_points.extend(extract_debugfs_from_chunk(file_data))

    if "netlink" in entry_types:
        entry_points.extend(extract_netlink_from_chunk(file_data))

    if "interrupts" in entry_types:
        entry_points.extend(extract_interrupts_from_chunk(file_data))

    # Detect kernel patterns if requested
    if enable_patterns:
        patterns = detect_kernel_patterns(file_data)

    return entry_points, patterns


def process_files_batch(
    files_batch: list[dict[str, Any]],
    entry_types: set[str],
    enable_patterns: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process a batch of files in a single thread."""
    global total_entry_points, processed_files

    all_entry_points = []
    all_patterns = []

    for file_data in files_batch:
        entry_points, patterns = extract_entry_points_from_chunk(file_data, entry_types, enable_patterns)
        all_entry_points.extend(entry_points)
        all_patterns.extend(patterns)

    # Update global counters safely
    with progress_lock:
        total_entry_points += len(all_entry_points)
        processed_files += len(files_batch)

    return all_entry_points, all_patterns


def process_large_json_streaming(
    input_file: str,
    output_file: str,
    entry_types: set[str],
    enable_patterns: bool = False,
    max_workers: int | None = None,
    patterns_output: str | None = None
):
    """Process large JSON file in streaming fashion with multi-threading.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file for entry points
        entry_types: Set of entry types to extract
        enable_patterns: Whether to detect kernel patterns
        max_workers: Number of worker threads
        patterns_output: Path to output JSON file for patterns (optional)
    """
    global total_entry_points, processed_files

    # Default to number of CPU cores
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid too much overhead

    print(f"Processing {input_file} ({os.path.getsize(input_file) / (1024*1024):.1f}MB) in streaming mode...")
    print(f"Extracting entry types: {', '.join(sorted(entry_types))}")
    if enable_patterns:
        print("Pattern detection: ENABLED")
    print(f"Using {max_workers} worker threads for parallel processing...")

    all_entry_points = []
    all_patterns = []

    # Get file size for progress calculation
    file_size = os.path.getsize(input_file)
    file_size_mb = file_size / (1024 * 1024)

    start_time = time.time()
    bytes_read = 0
    last_progress_time = start_time

    # Batch size for multi-threading (files per batch)
    batch_size = 50
    files_batch = []

    with open(input_file, encoding='utf-8') as f:
        # Skip opening bracket
        f.read(1)
        bytes_read += 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            while True:
                line = f.readline()
                if not line:
                    break

                bytes_read += len(line.encode('utf-8'))
                line = line.strip()

                # Skip empty lines and commas
                if not line or line == ',' or line == ']':
                    continue

                # Remove trailing comma if present
                if line.endswith(','):
                    line = line[:-1]

                try:
                    file_data = json.loads(line)
                    files_batch.append(file_data)

                    # Process batch when full
                    if len(files_batch) >= batch_size:
                        future = executor.submit(process_files_batch, files_batch[:], entry_types, enable_patterns)
                        futures.append(future)
                        files_batch = []

                        # Limit concurrent futures to avoid memory issues
                        if len(futures) >= max_workers * 2:
                            # Collect some completed futures
                            completed_futures = []
                            for future in as_completed(futures[:len(futures)//2], timeout=1):
                                entry_points, patterns = future.result()
                                all_entry_points.extend(entry_points)
                                all_patterns.extend(patterns)
                                completed_futures.append(future)

                            # Remove completed futures
                            for future in completed_futures:
                                futures.remove(future)

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    continue

                # Progress reporting every few seconds
                current_time = time.time()
                if current_time - last_progress_time >= 2.0:  # Report every 2 seconds
                    elapsed = current_time - start_time
                    progress_percent = (bytes_read / file_size) * 100
                    mb_per_sec = (bytes_read / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    eta_seconds = ((file_size - bytes_read) / (bytes_read / elapsed)) if bytes_read > 0 and elapsed > 0 else 0

                    print(f"Progress: {progress_percent:.1f}% ({bytes_read/(1024*1024):.1f}/{file_size_mb:.1f}MB) "
                          f"| Speed: {mb_per_sec:.1f}MB/s | ETA: {eta_seconds/60:.1f}min "
                          f"| Found: {total_entry_points} entry points | Files: {processed_files}")
                    last_progress_time = current_time

            # Process remaining batch
            if files_batch:
                future = executor.submit(process_files_batch, files_batch, entry_types, enable_patterns)
                futures.append(future)

            # Collect all remaining futures
            for future in as_completed(futures):
                entry_points, patterns = future.result()
                all_entry_points.extend(entry_points)
                all_patterns.extend(patterns)

    # Final progress
    elapsed = time.time() - start_time
    print(f"Processing complete! Found {len(all_entry_points)} entry points in {elapsed:.1f}s")
    if enable_patterns:
        print(f"Detected {len(all_patterns)} kernel patterns")
    print(f"Average speed: {file_size_mb/elapsed:.1f}MB/s")

    # Write entry points
    with open(output_file, 'w') as f:
        json.dump(all_entry_points, f, indent=2)
    print(f"Entry points written to: {output_file}")

    # Write patterns if requested
    if enable_patterns and patterns_output:
        with open(patterns_output, 'w') as f:
            json.dump(all_patterns, f, indent=2)
        print(f"Patterns written to: {patterns_output}")

    return len(all_entry_points)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract entry points and patterns from kernel symbols JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Entry types:
  syscalls    - System call functions (sys_*, ksys_*, etc.)
  ioctls      - Ioctl handlers (unlocked_ioctl, compat_ioctl)
  procfs      - /proc filesystem entries
  debugfs     - /sys/kernel/debug entries
  netlink     - Netlink message handlers
  interrupts  - Interrupt handlers (request_irq, etc.)
  all         - All of the above

Examples:
  # Extract only syscalls (default)
  %(prog)s input.json output.json

  # Extract all entry point types
  %(prog)s input.json output.json --entry-types all

  # Extract syscalls and ioctls with pattern detection
  %(prog)s input.json output.json --entry-types syscalls,ioctls --pattern-detection

  # Enable Clang enhancement (requires compile_commands.json)
  %(prog)s input.json output.json --enable-clang

  # Save patterns to separate file
  %(prog)s input.json output.json --pattern-detection --patterns-output patterns.json
        """
    )

    parser.add_argument("input_file", help="Input JSON file with parsed symbols")
    parser.add_argument("output_file", help="Output JSON file for entry points")

    parser.add_argument(
        "--entry-types",
        default="syscalls",
        help="Comma-separated list of entry types to extract (default: syscalls)"
    )

    parser.add_argument(
        "--pattern-detection",
        action="store_true",
        help="Enable kernel pattern detection (EXPORT_SYMBOL, module_param, etc.)"
    )

    parser.add_argument(
        "--enable-clang",
        action="store_true",
        help="Enable Clang for enhanced symbol analysis (requires compile_commands.json)"
    )

    parser.add_argument(
        "--patterns-output",
        help="Output file for detected patterns (only with --pattern-detection)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads (default: auto-detect)"
    )

    args = parser.parse_args()

    # Parse entry types
    entry_types = set()
    for entry_type in args.entry_types.split(","):
        entry_type = entry_type.strip().lower()
        if entry_type == "all":
            entry_types = {"syscalls", "ioctls", "procfs", "debugfs", "netlink", "interrupts"}
            break
        elif entry_type in {"syscalls", "ioctls", "procfs", "debugfs", "netlink", "interrupts"}:
            entry_types.add(entry_type)
        else:
            print(f"Error: Unknown entry type: {entry_type}")
            sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)

    # Note: Clang enhancement would be done in a pre-processing step
    if args.enable_clang:
        print("Note: Clang enhancement should be applied during initial parsing")
        print("      This tool works with already-parsed JSON data")

    try:
        count = process_large_json_streaming(
            args.input_file,
            args.output_file,
            entry_types,
            args.pattern_detection,
            args.workers,
            args.patterns_output
        )
        print(f"Successfully extracted {count} entry points")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
