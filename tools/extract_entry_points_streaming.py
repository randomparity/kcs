#!/usr/bin/env python3
"""
Streaming entry point extractor for large JSON files.
Processes kernel symbols JSON in chunks to avoid memory issues.
Multi-threaded for better performance on large datasets.
"""

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
                "description": f"System call: {syscall_name}"
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
                "description": f"Kernel syscall helper: {syscall_name}"
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
                "description": f"Direct syscall: {syscall_name}"
            })

    return entry_points


def process_files_batch(files_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process a batch of files in a single thread."""
    global total_entry_points, processed_files

    all_entry_points = []
    for file_data in files_batch:
        entry_points = extract_syscalls_from_chunk(file_data)
        all_entry_points.extend(entry_points)

    # Update global counters safely
    with progress_lock:
        total_entry_points += len(all_entry_points)
        processed_files += len(files_batch)

    return all_entry_points


def process_large_json_streaming(input_file: str, output_file: str, max_workers: int | None = None):
    """Process large JSON file in streaming fashion with multi-threading."""
    global total_entry_points, processed_files

    # Default to number of CPU cores
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid too much overhead

    print(f"Processing {input_file} ({os.path.getsize(input_file) / (1024*1024):.1f}MB) in streaming mode...")
    print(f"Using {max_workers} worker threads for parallel processing...")

    all_entry_points = []

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
                        future = executor.submit(process_files_batch, files_batch[:])
                        futures.append(future)
                        files_batch = []

                        # Limit concurrent futures to avoid memory issues
                        if len(futures) >= max_workers * 2:
                            # Collect some completed futures
                            completed_futures = []
                            for future in as_completed(futures[:len(futures)//2], timeout=1):
                                entry_points = future.result()
                                all_entry_points.extend(entry_points)
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
                future = executor.submit(process_files_batch, files_batch)
                futures.append(future)

            # Collect all remaining futures
            for future in as_completed(futures):
                entry_points = future.result()
                all_entry_points.extend(entry_points)

    # Final progress
    elapsed = time.time() - start_time
    print(f"Processing complete! Found {len(all_entry_points)} entry points in {elapsed:.1f}s")
    print(f"Average speed: {file_size_mb/elapsed:.1f}MB/s")

    # Write results
    with open(output_file, 'w') as f:
        json.dump(all_entry_points, f, indent=2)

    print(f"Results written to: {output_file}")
    return len(all_entry_points)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_entry_points_streaming.py <input_json> <output_json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    try:
        count = process_large_json_streaming(input_file, output_file)
        print(f"Successfully extracted {count} entry points")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
