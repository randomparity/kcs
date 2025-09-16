// Function pointer calls test case
// Expected: execute_operation has indirect call to op parameter (line 5)
// Expected: main has direct calls to execute_operation (lines 9 and 10)

int operation_a(int x) { return x + 1; }
int operation_b(int x) { return x * 2; }

int execute_operation(int value, int (*op)(int)) {
    return op(value);  // Indirect call
}

int main(void) {
    execute_operation(5, operation_a);
    execute_operation(10, &operation_b);
    return 0;
}
