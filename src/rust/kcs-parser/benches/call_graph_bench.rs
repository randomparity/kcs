use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kcs_parser::{ExtendedParserConfig, Parser};
use std::time::Duration;
use tempfile::tempdir;

// Sample C code with various call patterns for benchmarking
const SIMPLE_CALLS: &str = r#"
#include <linux/kernel.h>

static int helper1(int x) {
    return x + 1;
}

static int helper2(int x) {
    return helper1(x) * 2;
}

int main_function(int value) {
    int result = helper2(value);
    printk("Result: %d\n", result);
    return result;
}
"#;

const COMPLEX_CALLS: &str = r#"
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>

struct file_operations example_fops = {
    .open = example_open,
    .read = example_read,
    .write = example_write,
    .release = example_release,
};

static int example_open(struct inode *inode, struct file *file) {
    int ret = generic_file_open(inode, file);
    if (ret)
        return ret;

    mutex_lock(&example_mutex);
    example_count++;
    mutex_unlock(&example_mutex);

    return 0;
}

static ssize_t example_read(struct file *file, char __user *buf,
                           size_t count, loff_t *ppos) {
    ssize_t ret;

    ret = simple_read_from_buffer(buf, count, ppos,
                                 example_buffer, example_size);
    if (ret > 0) {
        example_stats_update(ret);
        trace_example_read(file, count, ret);
    }

    return ret;
}

static ssize_t example_write(struct file *file, const char __user *buf,
                            size_t count, loff_t *ppos) {
    int ret;

    if (count > MAX_WRITE_SIZE)
        return -EINVAL;

    ret = copy_from_user(example_buffer, buf, count);
    if (ret)
        return -EFAULT;

    example_process_data(example_buffer, count);
    example_stats_update(count);

    return count;
}

SYSCALL_DEFINE2(example_syscall, int, fd, const char __user *, data) {
    struct file *file = fget(fd);
    if (!file)
        return -EBADF;

    int ret = example_validate_data(data);
    if (ret) {
        fput(file);
        return ret;
    }

    ret = example_process_syscall(file, data);
    fput(file);

    return ret;
}
"#;

const FUNCTION_POINTERS: &str = r#"
#include <linux/kernel.h>

typedef int (*callback_fn)(int);

static int callback1(int x) {
    return x * 2;
}

static int callback2(int x) {
    return x + 10;
}

static callback_fn callbacks[] = {
    callback1,
    callback2,
    NULL
};

int process_with_callback(int value, int cb_index) {
    if (cb_index >= 0 && callbacks[cb_index]) {
        return callbacks[cb_index](value);
    }
    return value;
}

struct ops {
    int (*init)(void);
    int (*process)(int);
    void (*cleanup)(void);
};

static int ops_init(void) {
    return 0;
}

static int ops_process(int value) {
    return process_with_callback(value, 0);
}

static void ops_cleanup(void) {
    // cleanup
}

static struct ops example_ops = {
    .init = ops_init,
    .process = ops_process,
    .cleanup = ops_cleanup,
};

int use_ops(int value) {
    int ret = example_ops.init();
    if (ret)
        return ret;

    ret = example_ops.process(value);
    example_ops.cleanup();

    return ret;
}
"#;

fn bench_call_extraction_simple(c: &mut Criterion) {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("simple.c");
    std::fs::write(&file_path, SIMPLE_CALLS).unwrap();

    let mut group = c.benchmark_group("call_extraction_simple");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark with call extraction enabled
    group.bench_function("with_calls", |b| {
        b.iter(|| {
            let config = ExtendedParserConfig {
                include_call_graphs: true,
                ..Default::default()
            };
            let mut parser = Parser::new(config).unwrap();
            let result = parser.parse_file(black_box(file_path.to_str().unwrap()));
            black_box(result)
        })
    });

    // Benchmark without call extraction for comparison
    group.bench_function("without_calls", |b| {
        b.iter(|| {
            let config = ExtendedParserConfig {
                include_call_graphs: false,
                ..Default::default()
            };
            let mut parser = Parser::new(config).unwrap();
            let result = parser.parse_file(black_box(file_path.to_str().unwrap()));
            black_box(result)
        })
    });

    group.finish();
}

fn bench_call_extraction_complex(c: &mut Criterion) {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("complex.c");
    std::fs::write(&file_path, COMPLEX_CALLS).unwrap();

    let mut group = c.benchmark_group("call_extraction_complex");
    group.measurement_time(Duration::from_secs(15));

    // Benchmark complex code with many function calls
    group.bench_function("complex_with_calls", |b| {
        b.iter(|| {
            let config = ExtendedParserConfig {
                include_call_graphs: true,
                ..Default::default()
            };
            let mut parser = Parser::new(config).unwrap();
            let result = parser.parse_file(black_box(file_path.to_str().unwrap()));
            black_box(result)
        })
    });

    group.bench_function("complex_without_calls", |b| {
        b.iter(|| {
            let config = ExtendedParserConfig {
                include_call_graphs: false,
                ..Default::default()
            };
            let mut parser = Parser::new(config).unwrap();
            let result = parser.parse_file(black_box(file_path.to_str().unwrap()));
            black_box(result)
        })
    });

    group.finish();
}

fn bench_function_pointer_calls(c: &mut Criterion) {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("function_pointers.c");
    std::fs::write(&file_path, FUNCTION_POINTERS).unwrap();

    let mut group = c.benchmark_group("function_pointer_calls");
    group.measurement_time(Duration::from_secs(10));

    // Test function pointer call detection
    group.bench_function("function_pointers", |b| {
        b.iter(|| {
            let config = ExtendedParserConfig {
                include_call_graphs: true,
                ..Default::default()
            };
            let mut parser = Parser::new(config).unwrap();
            let result = parser.parse_file(black_box(file_path.to_str().unwrap()));
            black_box(result)
        })
    });

    group.finish();
}

// Removed direct CallExtractor benchmark - requires more complex tree-sitter setup

fn bench_call_graph_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_graph_scaling");
    group.measurement_time(Duration::from_secs(20));

    // Test scaling with different numbers of functions
    for num_functions in [10, 50, 100, 200].iter() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join(format!("scaling_{}.c", num_functions));

        // Generate C code with many functions calling each other
        let mut c_code = String::from("#include <linux/kernel.h>\n\n");

        // Generate helper functions
        for i in 0..*num_functions {
            c_code.push_str(&format!("static int func_{}(int x) {{\n", i));
            if i > 0 {
                c_code.push_str(&format!("    return func_{}(x + {});\n", i - 1, i));
            } else {
                c_code.push_str("    return x;\n");
            }
            c_code.push_str("}\n\n");
        }

        // Generate main function that calls all functions
        c_code.push_str("int main_function(int value) {\n");
        c_code.push_str("    int result = 0;\n");
        for i in 0..*num_functions {
            c_code.push_str(&format!("    result += func_{}(value + {});\n", i, i));
        }
        c_code.push_str("    return result;\n");
        c_code.push_str("}\n");

        std::fs::write(&file_path, c_code).unwrap();

        group.throughput(Throughput::Elements(*num_functions as u64));
        group.bench_with_input(
            BenchmarkId::new("functions", num_functions),
            num_functions,
            |b, _num_functions| {
                b.iter(|| {
                    let config = ExtendedParserConfig {
                        include_call_graphs: true,
                        ..Default::default()
                    };
                    let mut parser = Parser::new(config).unwrap();
                    let result = parser.parse_file(black_box(file_path.to_str().unwrap()));
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_call_extraction_simple,
    bench_call_extraction_complex,
    bench_function_pointer_calls,
    bench_call_graph_scaling
);
criterion_main!(benches);
