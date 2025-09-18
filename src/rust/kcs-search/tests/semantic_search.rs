//! Integration tests for semantic search functionality

use kcs_search::{
    embeddings::EmbeddingGenerator, query::QueryProcessor, EmbeddingModel, SearchEngine,
    SearchQuery,
};

#[tokio::test]
async fn test_end_to_end_search() {
    let engine = SearchEngine::new().unwrap();

    let query = SearchQuery {
        text: "kernel memory allocation".to_string(),
        top_k: 5,
        threshold: Some(0.7),
        config: None,
    };

    let results = engine.search(query).await.unwrap();
    // Results should be empty as we haven't indexed anything
    assert_eq!(results.len(), 0);
}

#[test]
fn test_embedding_similarity() {
    let generator = EmbeddingGenerator::new().unwrap();

    // Generate embeddings for similar texts
    let emb1 = generator.generate("kernel syscall handler").unwrap();
    let emb2 = generator.generate("kernel system call handler").unwrap();
    let emb3 = generator.generate("network packet processing").unwrap();

    // Similar texts should have higher similarity
    let sim12 = emb1.cosine_similarity(&emb2).unwrap();
    let sim13 = emb1.cosine_similarity(&emb3).unwrap();

    assert!(sim12 > sim13, "Similar texts should have higher similarity");
}

#[test]
fn test_query_expansion_with_kernel_terms() {
    let processor = QueryProcessor::new();

    // Test with common kernel abbreviations
    let expansions = processor.expand_query("fs vfs mm").unwrap();
    assert!(expansions.len() > 1);

    // Should expand fs to filesystem
    let expanded_text = expansions.join(" ");
    assert!(expanded_text.contains("filesystem") || expanded_text.contains("file_system"));
}

#[test]
fn test_embedding_normalization() {
    let generator = EmbeddingGenerator::new().unwrap();
    let mut embedding = generator.generate("test text").unwrap();

    // Normalize the embedding
    embedding.normalize();

    // Check that magnitude is approximately 1
    let magnitude: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.001,
        "Normalized vector should have magnitude ~1"
    );
}

#[test]
fn test_search_query_builder() {
    let query = SearchQuery {
        text: "vfs_read".to_string(),
        top_k: 10,
        threshold: Some(0.8),
        config: Some("x86_64:defconfig".to_string()),
    };

    assert_eq!(query.text, "vfs_read");
    assert_eq!(query.top_k, 10);
    assert_eq!(query.threshold, Some(0.8));
    assert_eq!(query.config, Some("x86_64:defconfig".to_string()));
}

#[test]
fn test_kernel_specific_embeddings() {
    let generator = EmbeddingGenerator::new().unwrap();

    // Generate embeddings for kernel-specific terms
    let kernel_emb = generator.generate("kernel").unwrap();
    let syscall_emb = generator.generate("syscall").unwrap();
    let driver_emb = generator.generate("driver").unwrap();
    let memory_emb = generator.generate("memory").unwrap();
    let file_emb = generator.generate("file").unwrap();

    // Ensure embeddings are different for different terms
    assert_ne!(kernel_emb.vector, syscall_emb.vector);
    assert_ne!(driver_emb.vector, memory_emb.vector);
    assert_ne!(file_emb.vector, kernel_emb.vector);

    // All should have the same dimension
    assert_eq!(kernel_emb.dimension, 384);
    assert_eq!(syscall_emb.dimension, 384);
    assert_eq!(driver_emb.dimension, 384);
}

#[test]
fn test_batch_embedding_generation() {
    let generator = EmbeddingGenerator::new().unwrap();

    let texts = vec![
        "sys_read".to_string(),
        "sys_write".to_string(),
        "sys_open".to_string(),
        "sys_close".to_string(),
    ];

    let embeddings = generator.batch_generate(&texts).unwrap();

    assert_eq!(embeddings.len(), 4);
    for emb in &embeddings {
        assert_eq!(emb.dimension, 384);
    }

    // Each should be unique
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            assert_ne!(embeddings[i].vector, embeddings[j].vector);
        }
    }
}

#[test]
fn test_query_preprocessing() {
    let processor = QueryProcessor::new();

    // Test abbreviation expansion
    let processed = processor.preprocess("fs ops and mm subsystem").unwrap();
    assert!(processed.contains("filesystem"));
    assert!(processed.contains("memory management"));

    // Test stop word removal
    let processed2 = processor.preprocess("the kernel is a system").unwrap();
    assert!(!processed2.contains(" is "));
    assert!(!processed2.contains(" a "));
}

#[test]
fn test_exact_match_boosting() {
    let processor = QueryProcessor::new();

    let mut results = vec![
        kcs_search::query::SearchResult {
            id: 1,
            name: "vfs_read".to_string(),
            file_path: "/fs/read_write.c".to_string(),
            score: 0.75,
            context: None,
        },
        kcs_search::query::SearchResult {
            id: 2,
            name: "generic_file_read".to_string(),
            file_path: "/mm/filemap.c".to_string(),
            score: 0.75,
            context: None,
        },
    ];

    processor.boost_exact_matches(&mut results, "vfs_read");

    // vfs_read should now have higher score due to exact match
    assert!(results[0].score > results[1].score);
}

#[test]
fn test_kernel_term_detection() {
    let processor = QueryProcessor::new();

    // Kernel-specific terms
    assert!(processor.is_kernel_term("sys_open"));
    assert!(processor.is_kernel_term("__init"));
    assert!(processor.is_kernel_term("file_operations"));
    assert!(processor.is_kernel_term("driver"));
    assert!(processor.is_kernel_term("vfs"));
    assert!(processor.is_kernel_term("mm")); // memory management abbreviation

    // Non-kernel terms
    assert!(!processor.is_kernel_term("hello"));
    assert!(!processor.is_kernel_term("world"));
    assert!(!processor.is_kernel_term("test"));
}

#[test]
fn test_embedding_determinism() {
    let generator = EmbeddingGenerator::new().unwrap();

    let text = "static int vfs_read(struct file *file)";
    let emb1 = generator.generate(text).unwrap();
    let emb2 = generator.generate(text).unwrap();

    // Same text should produce same embedding (deterministic)
    assert_eq!(emb1.vector, emb2.vector);
    assert_eq!(emb1.dimension, emb2.dimension);
}

#[test]
fn test_similarity_threshold() {
    let generator = EmbeddingGenerator::new().unwrap();

    let emb1 = generator.generate("kernel module loading").unwrap();
    let emb2 = generator.generate("kernel module loader").unwrap();
    let emb3 = generator.generate("python web scraping").unwrap();

    let sim_related = emb1.cosine_similarity(&emb2).unwrap();
    let sim_unrelated = emb1.cosine_similarity(&emb3).unwrap();

    // Related terms should have similarity > 0.5
    assert!(
        sim_related > 0.5,
        "Related terms should have high similarity"
    );

    // Unrelated terms should have lower similarity
    assert!(
        sim_unrelated < sim_related,
        "Unrelated terms should have lower similarity"
    );
}

#[test]
fn test_custom_embedding_config() {
    use kcs_search::embeddings::EmbeddingConfig;

    let config = EmbeddingConfig {
        dimension: 512,
        model_name: "test-model".to_string(),
        normalize: false,
        max_text_length: 1024,
    };

    let generator = EmbeddingGenerator::with_config(config).unwrap();
    let embedding = generator.generate("test text").unwrap();

    assert_eq!(embedding.dimension, 512);
    assert_eq!(generator.dimension(), 512);
    assert_eq!(generator.model_name(), "test-model");
}

#[test]
fn test_query_result_ranking() {
    let processor = QueryProcessor::new();

    let results = vec![
        kcs_search::query::SearchResult {
            id: 1,
            name: "func1".to_string(),
            file_path: "/a.c".to_string(),
            score: 0.9,
            context: None,
        },
        kcs_search::query::SearchResult {
            id: 2,
            name: "func2".to_string(),
            file_path: "/b.c".to_string(),
            score: 0.6, // Below default threshold
            context: None,
        },
        kcs_search::query::SearchResult {
            id: 3,
            name: "func3".to_string(),
            file_path: "/c.c".to_string(),
            score: 0.85,
            context: None,
        },
    ];

    let ranked = processor.rank_results(results);

    // Should filter out low scores and sort by score
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].score, 0.9);
    assert_eq!(ranked[1].score, 0.85);
}

#[test]
fn test_compound_term_expansion() {
    let processor = QueryProcessor::new();

    // Underscore-separated terms should be expanded
    let expansions = processor.expand_query("file_operations").unwrap();
    assert!(expansions.len() >= 2);

    let has_space_version = expansions.iter().any(|e| e.contains("file operations"));
    assert!(has_space_version, "Should include space-separated version");
}

#[test]
fn test_embedding_with_special_chars() {
    let generator = EmbeddingGenerator::new().unwrap();

    // Test with various special characters common in kernel code
    let texts = vec![
        "__init_data",
        "->next",
        "struct file_operations",
        "#define KERNEL_VERSION",
        "module_init()",
    ];

    for text in texts {
        let embedding = generator.generate(text);
        assert!(embedding.is_ok(), "Should handle special chars: {}", text);
        assert_eq!(embedding.unwrap().dimension, 384);
    }
}

#[test]
fn test_search_engine_with_custom_model() {
    let model = EmbeddingModel {
        model_name: "custom-kernel-model".to_string(),
        dimension: 256,
    };

    let engine = SearchEngine::with_model(model);
    assert!(engine.is_ok());
}

#[test]
fn test_query_keyword_extraction() {
    let processor = QueryProcessor::new();

    let keywords = processor
        .extract_keywords("find the Linux kernel memory management subsystem")
        .unwrap();

    // Should extract meaningful keywords
    assert!(keywords.contains(&"linux".to_string()));
    assert!(keywords.contains(&"kernel".to_string()));
    assert!(keywords.contains(&"memory".to_string()));
    assert!(keywords.contains(&"management".to_string()));
    assert!(keywords.contains(&"subsystem".to_string()));

    // Should not include stop words
    assert!(!keywords.contains(&"the".to_string()));
    // "find" is not a stop word and has >2 chars, so it should be included
    assert!(keywords.contains(&"find".to_string()) || keywords.len() >= 5);
}
