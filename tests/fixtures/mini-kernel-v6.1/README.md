# Mini Kernel Test Fixture v6.1

This is a simplified Linux kernel structure for testing KCS (Kernel Context Server) functionality.

## Purpose

- **CI Testing**: Provides a small, fast kernel-like structure for automated tests
- **Development**: Enables KCS development without requiring a full 5GB+ kernel repository
- **Validation**: Contains representative kernel patterns that KCS should analyze correctly

## Structure

```
mini-kernel-v6.1/
├── Makefile                    # Version info and build targets
├── Kconfig                     # Configuration options for testing config parsing
├── fs/
│   └── read_write.c           # VFS operations (sys_read, sys_write, vfs_read, vfs_write)
├── include/linux/
│   ├── fs.h                   # File system structures
│   ├── kernel.h               # Basic kernel definitions
│   ├── syscalls.h             # System call macros
│   └── types.h                # Basic type definitions
├── arch/x86/entry/syscalls/
│   └── syscall_64.tbl         # System call table
├── kernel/
│   └── sys.c                  # Additional system calls (open, close, openat)
└── drivers/char/
    └── mem.c                  # Character device operations (/dev/null, /dev/zero)
```

## Test Coverage

This fixture enables testing of:

### Symbol Analysis

- ✅ **Functions**: `vfs_read`, `vfs_write`, `sys_read`, `sys_write`, `null_read`, etc.
- ✅ **Structures**: `file`, `file_operations`
- ✅ **Macros**: `SYSCALL_DEFINE3`, `__user`
- ✅ **System calls**: Entry points and implementations

### Entry Point Detection

- ✅ **Syscalls**: `__NR_read`, `__NR_write`, `__NR_open`, `__NR_close`, `__NR_openat`
- ✅ **File operations**: `.read`, `.write` function pointers
- ✅ **Device interfaces**: Character device operations

### Call Graph Building

- ✅ **Direct calls**: `sys_read` → `vfs_read` → `__vfs_read`
- ✅ **Function pointers**: `file->f_op->read`
- ✅ **Helper functions**: `fget`, `fput`

### Configuration Analysis

- ✅ **Kconfig options**: `DEBUG_KERNEL`, `VFS`, `PROC_FS`, `64BIT`, `X86_64`, `MODULES`
- ✅ **Dependencies**: `PROC_FS` depends on `VFS`
- ✅ **Default values**: Various default configurations

### Impact Analysis

- ✅ **Config changes**: Modifications to Kconfig entries
- ✅ **Function signature changes**: Parameter modifications
- ✅ **Symbol additions/removals**: New or deleted functions

## Usage

### Basic Testing

```bash
# Set environment variable for tests
export KCS_KERNEL_PATH="./tests/fixtures/mini-kernel-v6.1"

# Run indexing test
tools/index_kernel.sh "$KCS_KERNEL_PATH"

# Run specific tests
python -m pytest tests/integration/test_mini_kernel.py -v
```

### CI Integration

```bash
# Fast CI testing without full kernel
make test-mini-kernel

# Verify all core functionality
python -m pytest tests/contract/ --kernel-fixture=mini-kernel-v6.1
```

## File Statistics

- **Total files**: 10 source files
- **Lines of code**: ~340 lines
- **Symbols**: ~20 functions, 3 structures, 5 macros
- **Entry points**: 5 system calls, 4 file operations
- **Size**: <50KB (vs 5GB+ for full kernel)

## Expected KCS Behavior

When KCS analyzes this fixture, it should:

1. **Parse all symbols** in under 1 second
2. **Detect all entry points** correctly
3. **Build call graph** with proper relationships
4. **Handle configuration** options and dependencies
5. **Generate citations** with correct file:line references

## Maintenance

This fixture should be updated when:

- KCS adds support for new kernel patterns
- Test coverage needs to be expanded
- New edge cases are discovered

Keep it minimal but representative of real kernel code patterns.
