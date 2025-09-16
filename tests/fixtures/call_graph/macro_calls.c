// Macro expansion calls test case
// Expected: test_macro calls CALL_HELPER macro (line 8)
// Expected: macro should be detected as Macro call type

#define CALL_HELPER(x) helper_function(x)

int helper_function(int x) { return x + 10; }

int test_macro(void) {
    CALL_HELPER(42);  // Should be detected as macro call
    return 0;
}
