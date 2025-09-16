// Problematic C code with syntax errors and edge cases
// Tests error handling and graceful degradation

// Incomplete function declaration
int broken_function(

// Function with macro call that should still be detected
#define CALL_HELPER(x) helper_function(x)

int helper_function(int x) { return x; }

int test_macro(void) {
    CALL_HELPER(42);  // Should be detected as macro call
    return 0;
}

// Function with syntax error
int another_broken() {
    missing_semicolon()  // Missing semicolon
    return 0;
}

// Valid function that should still be parsed
int valid_function(void) {
    return helper_function(123);
}
