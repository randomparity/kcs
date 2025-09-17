# Kernel Pattern Detection and Entry Point Extraction

## Overview

The Kernel Context Server (KCS) provides comprehensive pattern detection and entry point
extraction capabilities for Linux kernel source code analysis. This feature enables deep
understanding of kernel interfaces, exported symbols, module parameters, and various entry
points that define how the kernel interacts with userspace, hardware, and other kernel components.

## Table of Contents

- [Entry Point Types](#entry-point-types)
- [Pattern Detection](#pattern-detection)
- [Clang Integration](#clang-integration)
- [Usage Guide](#usage-guide)
- [Performance Optimization](#performance-optimization)
- [API Reference](#api-reference)

## Entry Point Types

KCS detects and analyzes multiple types of kernel entry points, each representing different
interaction mechanisms within the Linux kernel.

### 1. System Calls (Syscalls)

System calls are the fundamental interface between userspace and kernel space.

**Detection Patterns:**

- `SYSCALL_DEFINE*` macros
- `sys_*` function prefixes
- `__se_sys_*` and `__do_sys_*` internal implementations
- `ksys_*` kernel-internal syscall helpers

**Example:**

```c
SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)
{
    return ksys_read(fd, buf, count);
}
```

**Extracted Metadata:**

- Syscall name and number
- Parameter types and names
- Return type
- File location and line number

### 2. IOCTL Handlers

IOCTLs provide device-specific control operations.

**Detection Patterns:**

- `unlocked_ioctl` in `file_operations` structures
- `compat_ioctl` for 32-bit compatibility
- `_IO`, `_IOR`, `_IOW`, `_IOWR` macro definitions
- Magic numbers and command codes

**Example:**

```c
static const struct file_operations device_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = device_ioctl,
    .compat_ioctl = device_compat_ioctl,
};

#define DEVICE_IOC_MAGIC 'D'
#define DEVICE_IOC_GET_VERSION _IOR(DEVICE_IOC_MAGIC, 1, int)
```

**Extracted Metadata:**

- IOCTL handler function name
- Magic number
- Command codes and their meanings
- Associated file operations structure

### 3. ProcFS Entries

The `/proc` filesystem provides runtime kernel information.

**Detection Patterns:**

- `proc_create` and variants
- `proc_create_data`
- `create_proc_entry` (legacy)
- `proc_ops` and `file_operations` structures

**Example:**

```c
static const struct proc_ops my_proc_ops = {
    .proc_open = my_proc_open,
    .proc_read = seq_read,
    .proc_write = my_proc_write,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

proc_create("my_entry", 0644, NULL, &my_proc_ops);
```

**Extracted Metadata:**

- Proc entry path (e.g., `/proc/my_entry`)
- Permission modes
- Handler operations
- Parent directory

### 4. DebugFS Entries

DebugFS provides kernel debugging interfaces.

**Detection Patterns:**

- `debugfs_create_*` family of functions
- `debugfs_create_file`
- `debugfs_create_dir`
- Various typed creators (`debugfs_create_u32`, etc.)

**Example:**

```c
struct dentry *debug_dir;
debug_dir = debugfs_create_dir("my_driver", NULL);
debugfs_create_file("status", 0444, debug_dir, NULL, &status_fops);
debugfs_create_u32("counter", 0644, debug_dir, &debug_counter);
```

**Extracted Metadata:**

- DebugFS path
- Entry type (file, directory, typed value)
- Associated operations
- Data types for typed entries

### 5. Netlink Handlers

Netlink provides socket-based kernel-userspace communication.

**Detection Patterns:**

- `netlink_kernel_create`
- `genl_register_family`
- Netlink message handlers
- Protocol family definitions

**Example:**

```c
static struct netlink_kernel_cfg cfg = {
    .input = my_netlink_handler,
    .groups = MY_NETLINK_GROUPS,
};

nl_sock = netlink_kernel_create(&init_net, MY_NETLINK_PROTOCOL, &cfg);
```

**Extracted Metadata:**

- Protocol number
- Message handlers
- Multicast groups
- Family information

### 6. Interrupt Handlers

Interrupt handlers process hardware and software interrupts.

**Detection Patterns:**

- `request_irq` and variants
- `request_threaded_irq`
- `devm_request_irq` (managed resources)
- `free_irq`

**Example:**

```c
static irqreturn_t my_irq_handler(int irq, void *dev_id)
{
    /* Handle interrupt */
    return IRQ_HANDLED;
}

ret = request_irq(irq_num, my_irq_handler, IRQF_SHARED, "my_device", dev);
```

**Extracted Metadata:**

- IRQ number or resource
- Handler function
- Interrupt flags (shared, threaded, etc.)
- Device identification

### 7. Module Entry/Exit Points

Module initialization and cleanup functions.

**Detection Patterns:**

- `module_init` and `module_exit`
- `__init` and `__exit` attributes
- `MODULE_*` macros for metadata

**Example:**

```c
static int __init my_module_init(void)
{
    return 0;
}

static void __exit my_module_exit(void)
{
    /* Cleanup */
}

module_init(my_module_init);
module_exit(my_module_exit);
MODULE_LICENSE("GPL");
```

## Pattern Detection

Beyond entry points, KCS detects various kernel-specific patterns that define module behavior and interfaces.

### EXPORT_SYMBOL Variants

Kernel symbols can be exported for use by other modules.

**Types Detected:**

- `EXPORT_SYMBOL` - Standard export
- `EXPORT_SYMBOL_GPL` - GPL-only export
- `EXPORT_SYMBOL_NS` - Namespaced export
- `EXPORT_SYMBOL_NS_GPL` - Namespaced GPL export

**Example:**

```c
int my_kernel_function(void) { /* ... */ }
EXPORT_SYMBOL_GPL(my_kernel_function);

void my_ns_function(void) { /* ... */ }
EXPORT_SYMBOL_NS(my_ns_function, MY_NAMESPACE);
```

### Module Parameters

Runtime configurable module parameters.

**Detection Patterns:**

- `module_param` - Single parameter
- `module_param_array` - Array parameter
- `module_param_named` - Parameter with different variable name
- `MODULE_PARM_DESC` - Parameter descriptions

**Example:**

```c
static int debug_level = 0;
module_param(debug_level, int, 0644);
MODULE_PARM_DESC(debug_level, "Debug verbosity level (0-3)");

static int irqs[MAX_DEVICES];
static int num_irqs;
module_param_array(irqs, int, &num_irqs, 0444);
MODULE_PARM_DESC(irqs, "IRQ numbers for devices");
```

### Boot Parameters

Kernel command-line parameters processed at boot time.

**Detection Patterns:**

- `__setup` - Standard boot parameter
- `early_param` - Early boot parameter
- `core_param` - Core kernel parameter

**Example:**

```c
static int __init setup_my_feature(char *str)
{
    my_feature_enabled = simple_strtol(str, NULL, 0);
    return 1;
}
__setup("my_feature=", setup_my_feature);

static int __init early_my_param(char *str)
{
    /* Process early in boot */
    return 0;
}
early_param("my_early", early_my_param);
```

## Clang Integration

KCS optionally integrates with Clang for enhanced semantic analysis, providing richer type
information and documentation extraction.

### Setup Guide

#### Prerequisites

1. **Install Clang and development libraries:**

   ```bash
   # Ubuntu/Debian
   sudo apt-get install clang libclang-dev

   # Fedora/RHEL
   sudo dnf install clang clang-devel

   # Arch Linux
   sudo pacman -S clang
   ```

2. **Generate compilation database:**

   ```bash
   # In kernel source directory
   make clean
   make defconfig  # Or your target config
   # Generates compile_commands.json
   bear -- make -j$(nproc)
   ```

   Alternatively, use kernel's built-in support (kernel 5.x+):

   ```bash
   make compile_commands.json
   ```

#### Enabling Clang Enhancement

1. **Via CLI tool:**

   ```bash
   tools/extract_entry_points_streaming.py \
       input.json output.json \
       --enable-clang
   ```

2. **Via Python API:**

   ```python
   from kcs_python_bridge import enhance_symbols_with_clang

   symbols = enhance_symbols_with_clang(
       file_path="/path/to/kernel/file.c",
       compile_commands_path="/path/to/compile_commands.json"
   )
   ```

3. **Via MCP server configuration:**

   ```bash
   export KCS_ENABLE_CLANG=true
   export KCS_COMPILE_COMMANDS=/path/to/compile_commands.json
   kcs-mcp --host 0.0.0.0 --port 8080
   ```

### Benefits of Clang Integration

1. **Accurate Type Information:**
   - Complete function signatures
   - Parameter types and names
   - Return types
   - Type qualifiers (const, volatile, etc.)

2. **Documentation Extraction:**
   - Kernel-doc comments
   - Function descriptions
   - Parameter documentation
   - Return value documentation

3. **Semantic Analysis:**
   - Macro expansion
   - Include file resolution
   - Symbol cross-referencing
   - Attribute detection (`__init`, `__exit`, etc.)

## Usage Guide

### Command-Line Interface

The primary CLI tool for entry point extraction:

```bash
# Extract all entry points
tools/extract_entry_points_streaming.py \
    parsed_symbols.json \
    entry_points.json \
    --entry-types all

# Extract specific types with pattern detection
tools/extract_entry_points_streaming.py \
    parsed_symbols.json \
    entry_points.json \
    --entry-types syscalls,ioctls,procfs \
    --pattern-detection \
    --patterns-output patterns.json

# With parallel processing
tools/extract_entry_points_streaming.py \
    parsed_symbols.json \
    entry_points.json \
    --entry-types all \
    --workers 8
```

### Python API

```python
import kcs_python_bridge as kpb

# Extract entry points
entry_points = kpb.extract_entry_points(
    kernel_dir="/path/to/kernel",
    include_syscalls=True,
    include_ioctls=True,
    include_procfs=True,
    include_debugfs=True,
    include_netlink=True,
    include_interrupts=True,
    include_modules=True
)

# Detect patterns
patterns = kpb.detect_patterns(
    file_path="/path/to/kernel/file.c",
    content=file_content
)

# Enhanced parsing with Clang
symbols = kpb.enhance_symbols_with_clang(
    file_path="/path/to/kernel/file.c",
    compile_commands_path="/path/to/compile_commands.json"
)
```

### Database Storage

Entry points and patterns are stored with JSONB metadata:

```sql
-- Query exported GPL symbols
SELECT name, file_path, metadata->>'export_type' as export_type
FROM symbol
WHERE metadata->>'export_type' = 'EXPORT_SYMBOL_GPL';

-- Find all procfs entries
SELECT name, file_path, metadata->>'proc_ops' as ops
FROM entry_point
WHERE entry_type = 'ProcFs';

-- Get module parameters with descriptions
SELECT name, metadata->>'description' as description
FROM kernel_pattern
WHERE pattern_type = 'ModuleParam';
```

## Performance Optimization

### Target Performance Metrics

- **Single file processing:** < 100ms
- **Pattern detection:** < 50ms per file
- **Full subsystem analysis:** < 30 seconds
- **Complete kernel indexing:** < 20 minutes

### Optimization Strategies

#### 1. Parallel Processing

Use multiple workers for large-scale processing:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(file_path):
    with open(file_path) as f:
        content = f.read()
    return kpb.detect_patterns(file_path, content)

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_file, f) for f in files]
    for future in as_completed(futures):
        patterns = future.result()
```

#### 2. Streaming Processing

For large JSON files, use streaming to avoid memory issues:

```python
# The CLI tool automatically streams large files
# Processes in chunks of 50 files by default
```

#### 3. Selective Extraction

Extract only needed entry point types:

```bash
# Fast: Only syscalls
--entry-types syscalls

# Slower: All types
--entry-types all
```

#### 4. Caching Strategies

1. **Translation Unit Cache:** Clang integration caches parsed translation units
2. **Database Indexing:** Ensure proper indexes on metadata JSONB fields
3. **File-level Caching:** Skip unchanged files based on SHA hash

#### 5. Database Optimization

```sql
-- Ensure indexes exist for common queries
CREATE INDEX idx_symbol_export_type
ON symbol ((metadata->>'export_type'))
WHERE metadata IS NOT NULL;

CREATE INDEX idx_entry_point_type
ON entry_point (entry_type);

-- Use connection pooling (2-10 connections)
-- Configure in MCP server
```

### Benchmarking

Run performance tests:

```bash
# Run performance validation
pytest tests/performance/test_pattern_performance.py -v

# Benchmark with real kernel
python tests/performance/test_pattern_performance.py
```

Expected results:

- Simple files: < 10ms mean, < 20ms max
- Medium files: < 50ms mean, < 100ms max
- Complex files: < 100ms mean, < 200ms max
- Parallel speedup: > 1.3x with 2+ workers

## API Reference

### Entry Point Structure

```json
{
  "name": "sys_read",
  "entry_type": "Syscall",
  "file_path": "fs/read_write.c",
  "line_number": 620,
  "signature": "SYSCALL_DEFINE3(read, unsigned int, fd, ...)",
  "description": "System call: read",
  "metadata": {
    "syscall_number": 0,
    "parameters": ["fd", "buf", "count"],
    "return_type": "ssize_t"
  }
}
```

### Pattern Structure

```json
{
  "pattern_type": "ExportSymbol",
  "name": "vfs_read",
  "file_path": "fs/read_write.c",
  "line_number": 500,
  "raw_text": "EXPORT_SYMBOL_GPL(vfs_read);",
  "metadata": {
    "export_type": "EXPORT_SYMBOL_GPL",
    "namespace": null
  }
}
```

### MCP Endpoints

The pattern detection features are exposed through MCP endpoints:

- `POST /mcp/tools/extract_entry_points` - Extract entry points
- `POST /mcp/tools/detect_patterns` - Detect kernel patterns
- `POST /mcp/tools/enhance_symbols` - Enhance with Clang

See [API documentation](../api/) for detailed endpoint specifications.

## Troubleshooting

### Common Issues

1. **Clang not found:**
   - Ensure libclang is installed
   - Check `LIBCLANG_PATH` environment variable
   - Verify Python binding: `python -c "import clang"`

2. **Missing compile_commands.json:**
   - Generate with `bear` or kernel's built-in support
   - Ensure paths are absolute in the file
   - Check file permissions

3. **Slow performance:**
   - Increase worker count for parallel processing
   - Use selective entry type extraction
   - Check database indexes
   - Monitor with `tests/performance/`

4. **Memory issues with large files:**
   - Use streaming mode (automatic for CLI)
   - Process in smaller batches
   - Increase system memory limits

### Debug Mode

Enable detailed logging:

```bash
export KCS_LOG_LEVEL=DEBUG
export RUST_LOG=kcs_parser=debug,kcs_extractor=debug
```

## Future Enhancements

Planned improvements for pattern detection:

1. **Additional Entry Points:**
   - Tracepoints and trace events
   - Kprobes and kretprobes
   - eBPF attachment points
   - Security hooks (LSM)

2. **Enhanced Pattern Detection:**
   - Design pattern recognition
   - Lock usage patterns
   - Memory allocation patterns
   - Error handling patterns

3. **Semantic Analysis:**
   - Data flow analysis
   - Control flow graphs
   - Taint analysis for security
   - Cross-reference resolution

4. **Performance:**
   - Incremental parsing
   - Distributed processing
   - GPU acceleration for pattern matching
   - Persistent caching layer

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on adding new pattern types or
entry point detectors.

## License

The KCS pattern detection features are licensed under the same terms as the main project.
See [LICENSE](../../LICENSE) for details.
