// Simple function calls test case
// Expected: main_function calls helper_function twice (lines 6 and 7)

int helper_function(int x) {
    return x * 2;
}

int main_function(int a, int b) {
    int result = helper_function(a);
    helper_function(b);
    return result;
}
