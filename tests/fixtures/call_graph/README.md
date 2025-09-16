# Call Graph Test Fixtures

This directory contains C source files for testing call graph extraction functionality.

## Test Files

### simple_calls.c

- **Purpose**: Basic direct function calls
- **Expected**: `main_function` calls `helper_function` twice (lines 7, 8)
- **Call Types**: All Direct

### function_pointers.c

- **Purpose**: Function pointer and indirect calls
- **Expected**:
  - `execute_operation` has indirect call via `op` parameter (line 5)
  - `main` has direct calls to `execute_operation` (lines 9, 10)
- **Call Types**: Direct and Indirect

### macro_calls.c

- **Purpose**: Macro expansion calls
- **Expected**: `test_macro` calls `CALL_HELPER` macro (line 8)
- **Call Types**: Macro

### problematic.c

- **Purpose**: Error handling and graceful degradation
- **Expected**:
  - Parser should not crash on syntax errors
  - Valid calls should still be extracted
  - Invalid syntax should be reported in errors
- **Call Types**: Mixed, with parse errors

### kernel_style.c

- **Purpose**: Realistic kernel code patterns
- **Expected**:
  - `ext4_create` calls multiple functions directly
  - `ext4_file_open` has indirect call through function pointer
  - `test_error_handling` calls macro
- **Call Types**: Direct, Indirect, Macro

## Usage

These fixtures are used by:

- Contract tests (validate parser API contracts)
- Integration tests (end-to-end parsing scenarios)
- Performance tests (benchmarking with known datasets)
- Manual testing during development

Each file includes comments indicating expected call relationships for validation.
