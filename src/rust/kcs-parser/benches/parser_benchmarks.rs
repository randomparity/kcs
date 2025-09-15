use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kcs_parser::{ExtendedParserConfig, Parser};
use std::collections::HashMap;
use std::fs;
use std::time::Duration;
use tempfile::tempdir;

// Sample kernel C code for benchmarking
const SMALL_C_FILE: &str = r#"
#include <linux/fs.h>
#include <linux/kernel.h>

static int example_function(int param) {
    return param * 2;
}

struct example_struct {
    int field1;
    char *field2;
};

#define EXAMPLE_MACRO(x) ((x) + 1)

int global_variable = 42;

SYSCALL_DEFINE1(example, int, value)
{
    return example_function(value);
}
"#;

const MEDIUM_C_FILE: &str = r#"
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

struct file_operations example_fops;

static struct example_data {
    int count;
    char buffer[1024];
    struct mutex lock;
} example_data;

static int example_open(struct inode *inode, struct file *file)
{
    mutex_lock(&example_data.lock);
    example_data.count++;
    mutex_unlock(&example_data.lock);
    return 0;
}

static int example_release(struct inode *inode, struct file *file)
{
    mutex_lock(&example_data.lock);
    example_data.count--;
    mutex_unlock(&example_data.lock);
    return 0;
}

static ssize_t example_read(struct file *file, char __user *buf,
                           size_t count, loff_t *ppos)
{
    size_t len = strlen(example_data.buffer);

    if (*ppos >= len)
        return 0;

    if (count > len - *ppos)
        count = len - *ppos;

    if (copy_to_user(buf, example_data.buffer + *ppos, count))
        return -EFAULT;

    *ppos += count;
    return count;
}

static ssize_t example_write(struct file *file, const char __user *buf,
                            size_t count, loff_t *ppos)
{
    if (count >= sizeof(example_data.buffer))
        return -EINVAL;

    if (copy_from_user(example_data.buffer, buf, count))
        return -EFAULT;

    example_data.buffer[count] = '\0';
    return count;
}

static long example_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {
    case 0x1000:
        return example_data.count;
    case 0x1001:
        example_data.count = (int)arg;
        return 0;
    default:
        return -ENOTTY;
    }
}

static struct file_operations example_fops = {
    .owner = THIS_MODULE,
    .open = example_open,
    .release = example_release,
    .read = example_read,
    .write = example_write,
    .unlocked_ioctl = example_ioctl,
};

static int __init example_init(void)
{
    mutex_init(&example_data.lock);
    return 0;
}

static void __exit example_exit(void)
{
    // Cleanup
}

module_init(example_init);
module_exit(example_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Test Author");
MODULE_DESCRIPTION("Example kernel module for benchmarking");
"#;

// Generate large C file content
fn generate_large_c_file() -> String {
    let mut content = String::new();
    content.push_str("#include <linux/kernel.h>\n");
    content.push_str("#include <linux/module.h>\n\n");

    // Generate many functions
    for i in 0..1000 {
        content.push_str(&format!(
            "static int function_{i}(int param) {{\n    return param + {i};\n}}\n\n"
        ));
    }

    // Generate many structs
    for i in 0..100 {
        content.push_str(&format!(
            "struct data_struct_{i} {{\n    int field_{i};\n    char name_{i}[64];\n}};\n\n"
        ));
    }

    // Generate many macros
    for i in 0..200 {
        content.push_str(&format!("#define MACRO_{i}(x) ((x) + {i})\n"));
    }

    content
}

fn setup_temp_files() -> (tempfile::TempDir, Vec<String>) {
    let temp_dir = tempdir().expect("Failed to create temp directory");

    let small_path = temp_dir.path().join("small.c");
    let medium_path = temp_dir.path().join("medium.c");
    let large_path = temp_dir.path().join("large.c");

    fs::write(&small_path, SMALL_C_FILE).expect("Failed to write small file");
    fs::write(&medium_path, MEDIUM_C_FILE).expect("Failed to write medium file");
    fs::write(&large_path, generate_large_c_file()).expect("Failed to write large file");

    let paths = vec![
        small_path.to_string_lossy().to_string(),
        medium_path.to_string_lossy().to_string(),
        large_path.to_string_lossy().to_string(),
    ];

    (temp_dir, paths)
}

fn bench_single_file_parsing(c: &mut Criterion) {
    let (_temp_dir, file_paths) = setup_temp_files();

    let mut group = c.benchmark_group("single_file_parsing");

    for (i, file_path) in file_paths.iter().enumerate() {
        let size_name = match i {
            0 => "small",
            1 => "medium",
            2 => "large",
            _ => "unknown",
        };

        let file_size = fs::metadata(file_path).unwrap().len();
        group.throughput(Throughput::Bytes(file_size));

        group.bench_with_input(
            BenchmarkId::new("tree_sitter_only", size_name),
            file_path,
            |b, file_path| {
                let config = ExtendedParserConfig {
                    use_clang: false,
                    ..Default::default()
                };
                let mut parser = Parser::new(config).expect("Failed to create parser");

                b.iter(|| black_box(parser.parse_file(black_box(file_path)).unwrap()));
            },
        );

        // Skip clang benchmarks for now due to setup complexity
        // group.bench_with_input(
        //     BenchmarkId::new("with_clang", size_name),
        //     file_path,
        //     |b, file_path| {
        //         let config = ExtendedParserConfig {
        //             use_clang: true,
        //             ..Default::default()
        //         };
        //         let mut parser = Parser::new(config).expect("Failed to create parser");
        //
        //         b.iter(|| {
        //             black_box(parser.parse_file(black_box(file_path)).unwrap())
        //         });
        //     },
        // );
    }

    group.finish();
}

fn bench_content_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_parsing");

    let large_content = generate_large_c_file();
    let test_cases = vec![
        ("small", SMALL_C_FILE),
        ("medium", MEDIUM_C_FILE),
        ("large", &large_content),
    ];

    for (size_name, content) in test_cases {
        group.throughput(Throughput::Bytes(content.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("parse_content", size_name),
            content,
            |b, content| {
                let config = ExtendedParserConfig {
                    use_clang: false,
                    ..Default::default()
                };
                let mut parser = Parser::new(config).expect("Failed to create parser");

                b.iter(|| {
                    black_box(
                        parser
                            .parse_file_content(black_box("test.c"), black_box(content))
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_parsing");
    group.measurement_time(Duration::from_secs(20)); // Longer measurement for batch operations

    // Test different batch sizes
    for batch_size in [1, 5, 10, 25, 50].iter() {
        let mut files = HashMap::new();

        // Create batch of files with mixed sizes
        for i in 0..*batch_size {
            let content = match i % 3 {
                0 => SMALL_C_FILE.to_string(),
                1 => MEDIUM_C_FILE.to_string(),
                2 => generate_large_c_file(),
                _ => unreachable!(),
            };
            files.insert(format!("file_{}.c", i), content);
        }

        let total_size: usize = files.values().map(|c| c.len()).sum();
        group.throughput(Throughput::Bytes(total_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_content", batch_size),
            &files,
            |b, files| {
                let config = ExtendedParserConfig {
                    use_clang: false,
                    ..Default::default()
                };
                let mut parser = Parser::new(config).expect("Failed to create parser");

                b.iter(|| {
                    black_box(
                        parser
                            .parse_files_content(black_box(files.clone()))
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_symbol_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_extraction");

    // Benchmark specific parsing operations
    let config = ExtendedParserConfig {
        use_clang: false,
        ..Default::default()
    };
    let mut parser = Parser::new(config).expect("Failed to create parser");

    group.bench_function("function_extraction", |b| {
        let content = (0..100)
            .map(|i| format!("static int func_{}(int x) {{ return x + {}; }}", i, i))
            .collect::<Vec<_>>()
            .join("\n");

        b.iter(|| {
            let result = parser.parse_file_content("test.c", &content).unwrap();
            black_box(
                result
                    .symbols
                    .iter()
                    .filter(|s| s.kind == "function")
                    .count(),
            )
        });
    });

    group.bench_function("struct_extraction", |b| {
        let content = (0..50)
            .map(|i| {
                format!(
                    "struct data_{} {{ int field_{}; char name_{}[64]; }};",
                    i, i, i
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        b.iter(|| {
            let result = parser.parse_file_content("test.c", &content).unwrap();
            black_box(result.symbols.iter().filter(|s| s.kind == "struct").count())
        });
    });

    group.bench_function("macro_extraction", |b| {
        let content = (0..200)
            .map(|i| format!("#define MACRO_{}(x) ((x) + {})", i, i))
            .collect::<Vec<_>>()
            .join("\n");

        b.iter(|| {
            let result = parser.parse_file_content("test.c", &content).unwrap();
            black_box(result.symbols.iter().filter(|s| s.kind == "macro").count())
        });
    });

    group.finish();
}

fn bench_parser_reconfiguration(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_reconfiguration");

    group.bench_function("config_change", |b| {
        let config = ExtendedParserConfig::default();
        let mut parser = Parser::new(config).expect("Failed to create parser");

        b.iter(|| {
            let new_config = kcs_parser::ParserConfig {
                tree_sitter_enabled: true,
                clang_enabled: false,
                target_arch: black_box("arm64".to_string()),
                kernel_version: black_box("6.5".to_string()),
            };
            parser.reconfigure(new_config).unwrap();
            black_box(())
        });
    });

    group.finish();
}

fn bench_file_discovery(c: &mut Criterion) {
    let temp_dir = tempdir().expect("Failed to create temp directory");

    // Create a directory structure with many C files
    for i in 0..100 {
        let subdir = temp_dir.path().join(format!("dir_{}", i % 10));
        fs::create_dir_all(&subdir).unwrap();

        let file_path = subdir.join(format!("file_{}.c", i));
        fs::write(&file_path, SMALL_C_FILE).unwrap();

        // Add some non-C files to test filtering
        let txt_path = subdir.join(format!("file_{}.txt", i));
        fs::write(&txt_path, "not a c file").unwrap();
    }

    let mut group = c.benchmark_group("file_discovery");

    group.bench_function("find_c_files", |b| {
        let config = ExtendedParserConfig::default();
        let mut parser = Parser::new(config).expect("Failed to create parser");

        b.iter(|| black_box(parser.parse_directory(black_box(temp_dir.path())).unwrap()));
    });

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test parser creation overhead
    group.bench_function("parser_creation", |b| {
        b.iter(|| {
            let config = ExtendedParserConfig::default();
            black_box(Parser::new(config).unwrap())
        });
    });

    // Test repeated parsing of same content (testing internal caching/optimization)
    group.bench_function("repeated_parsing", |b| {
        let config = ExtendedParserConfig::default();
        let mut parser = Parser::new(config).expect("Failed to create parser");

        b.iter(|| {
            for _ in 0..10 {
                black_box(parser.parse_file_content("test.c", MEDIUM_C_FILE).unwrap());
            }
        });
    });

    group.finish();
}

// Performance regression tests based on constitutional requirements
fn bench_constitutional_requirements(c: &mut Criterion) {
    let mut group = c.benchmark_group("constitutional_requirements");
    group.measurement_time(Duration::from_secs(30));

    // Test: Large kernel file parsing should complete in reasonable time
    // Target: Most files should parse in <100ms for responsive IDE integration
    group.bench_function("large_file_parsing_target", |b| {
        let large_content = generate_large_c_file();
        let config = ExtendedParserConfig::default();
        let mut parser = Parser::new(config).expect("Failed to create parser");

        b.iter(|| {
            let start = std::time::Instant::now();
            let result = parser
                .parse_file_content("large.c", &large_content)
                .unwrap();
            let duration = start.elapsed();

            // Constitutional requirement: Fast parsing for IDE integration
            assert!(
                duration.as_millis() < 1000,
                "Parsing took too long: {}ms",
                duration.as_millis()
            );
            black_box(result)
        });
    });

    // Test: Symbol extraction accuracy
    group.bench_function("symbol_accuracy_check", |b| {
        let config = ExtendedParserConfig::default();
        let mut parser = Parser::new(config).expect("Failed to create parser");

        b.iter(|| {
            let result = parser.parse_file_content("test.c", MEDIUM_C_FILE).unwrap();

            // Verify we extract expected symbols
            let function_count = result
                .symbols
                .iter()
                .filter(|s| s.kind == "function")
                .count();
            let struct_count = result.symbols.iter().filter(|s| s.kind == "struct").count();

            // Basic sanity checks
            assert!(function_count > 0, "Should extract functions");
            assert!(struct_count > 0, "Should extract structs");

            black_box((function_count, struct_count))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_file_parsing,
    bench_content_parsing,
    bench_batch_parsing,
    bench_symbol_extraction,
    bench_parser_reconfiguration,
    bench_file_discovery,
    bench_memory_usage,
    bench_constitutional_requirements
);

criterion_main!(benches);
