; Tree-sitter query patterns for C function call detection
; Used by kcs-parser for call graph extraction
; Author: KCS Team
; Date: 2025-01-20
; Specification: 007-call-graph-extraction/research.md

; =============================================================================
; Direct Function Calls
; =============================================================================

; Basic function calls with identifier names
(call_expression
  function: (identifier) @function-name
  arguments: (argument_list) @args) @call-site

; =============================================================================
; Function Pointer Calls
; =============================================================================

; Indirect calls through function pointers
(call_expression
  function: (pointer_expression
    argument: (identifier) @pointer-name)
  arguments: (argument_list) @args) @call-site

; Function pointer dereference calls
(call_expression
  function: (parenthesized_expression
    (pointer_expression
      argument: (identifier) @pointer-name))
  arguments: (argument_list) @args) @call-site

; =============================================================================
; Member Function Calls (Callbacks)
; =============================================================================

; Struct member function calls (callback patterns)
(call_expression
  function: (field_expression
    argument: (identifier) @struct-name
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; Arrow operator function calls (ptr->func())
(call_expression
  function: (field_expression
    argument: (pointer_expression
      argument: (identifier) @struct-pointer)
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; Nested member access (obj.member.func())
(call_expression
  function: (field_expression
    argument: (field_expression
      argument: (identifier) @outer-struct
      field: (field_identifier) @inner-member)
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; =============================================================================
; Macro Function Calls
; =============================================================================

; Macro calls (uppercase identifiers)
(call_expression
  function: (identifier) @macro-name
  arguments: (argument_list) @args) @call-site
  (#match? @macro-name "^[A-Z_][A-Z0-9_]*$")

; =============================================================================
; Kernel-Specific Patterns
; =============================================================================

; EXPORT_SYMBOL calls
(call_expression
  function: (identifier) @export-macro
  arguments: (argument_list
    (identifier) @exported-function)) @export-site
  (#eq? @export-macro "EXPORT_SYMBOL")

; EXPORT_SYMBOL_GPL calls
(call_expression
  function: (identifier) @export-macro
  arguments: (argument_list
    (identifier) @exported-function)) @export-site
  (#eq? @export-macro "EXPORT_SYMBOL_GPL")

; module_init and module_exit
(call_expression
  function: (identifier) @module-macro
  arguments: (argument_list
    (identifier) @init-function)) @module-site
  (#match? @module-macro "^module_(init|exit)$")

; Syscall definitions
(call_expression
  function: (identifier) @syscall-macro
  arguments: (argument_list) @args) @syscall-site
  (#match? @syscall-macro "^SYSCALL_DEFINE[0-6]$")

; =============================================================================
; Function Pointer Assignments
; =============================================================================

; Function pointer assignments in struct initializers
(init_declarator
  declarator: (identifier) @struct-name
  value: (initializer_list
    (initializer_pair
      (field_designator
        field: (field_identifier) @field-name)
      value: (identifier) @function-name))) @assignment-site

; Direct function pointer assignments
(assignment_expression
  left: (field_expression
    argument: (identifier) @struct-name
    field: (field_identifier) @field-name)
  right: (identifier) @function-name) @assignment-site

; Function pointer assignments through arrow operator
(assignment_expression
  left: (field_expression
    argument: (pointer_expression
      argument: (identifier) @struct-pointer)
    field: (field_identifier) @field-name)
  right: (identifier) @function-name) @assignment-site

; =============================================================================
; Array Function Pointer Calls
; =============================================================================

; Function calls through array indexing
(call_expression
  function: (subscript_expression
    argument: (identifier) @array-name
    index: (_) @index)
  arguments: (argument_list) @args) @call-site

; =============================================================================
; Conditional and Guarded Calls
; =============================================================================

; Function calls within if statements
(if_statement
  condition: (_)
  consequence: (compound_statement
    (expression_statement
      (call_expression
        function: (identifier) @function-name
        arguments: (argument_list) @args) @call-site)))

; Function calls within preprocessor conditionals
(preproc_if
  condition: (_)
  (expression_statement
    (call_expression
      function: (identifier) @function-name
      arguments: (argument_list) @args) @call-site))

; =============================================================================
; Complex Call Patterns
; =============================================================================

; Chained function calls (func1()->func2())
(call_expression
  function: (field_expression
    argument: (call_expression) @inner-call
    field: (field_identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; Function calls with cast expressions
(call_expression
  function: (cast_expression
    type: (_)
    value: (identifier) @function-name)
  arguments: (argument_list) @args) @call-site

; =============================================================================
; Assembly and Inline Calls
; =============================================================================

; Inline assembly with function calls
(gnu_asm_expression
  assembly_code: (string_literal) @asm-code) @asm-site

; =============================================================================
; Error Handling Patterns
; =============================================================================

; IS_ERR, PTR_ERR patterns (common in kernel)
(call_expression
  function: (identifier) @error-macro
  arguments: (argument_list
    (identifier) @checked-pointer)) @error-check
  (#match? @error-macro "^(IS_ERR|PTR_ERR|ERR_PTR)$")

; =============================================================================
; Memory Management Calls
; =============================================================================

; kmalloc, kfree, vmalloc patterns
(call_expression
  function: (identifier) @memory-function
  arguments: (argument_list) @args) @memory-call
  (#match? @memory-function "^(k|v)?(m|re)?alloc|k?free|get_free_pages|__get_free_pages$")

; =============================================================================
; Locking and Synchronization
; =============================================================================

; Spinlock, mutex, and other locking primitives
(call_expression
  function: (identifier) @lock-function
  arguments: (argument_list) @args) @lock-call
  (#match? @lock-function "^(spin_|mutex_|raw_spin_|read_|write_)?(lock|unlock|trylock)(_irq|_irqsave|_bh)?$")
