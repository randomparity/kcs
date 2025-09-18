//! Integration tests for kcs-search

use kcs_search::{EmbeddingModel, SearchEngine, SearchQuery};

#[tokio::test]
async fn test_search_engine_creation() {
    let engine = SearchEngine::new();
    assert!(engine.is_ok());
}

#[tokio::test]
async fn test_search_with_empty_query() {
    let engine = SearchEngine::new().unwrap();
    let query = SearchQuery {
        text: String::new(),
        top_k: 10,
        threshold: None,
        config: None,
    };

    let results = engine.search(query).await;
    assert!(results.is_ok());
    assert_eq!(results.unwrap().len(), 0);
}

#[tokio::test]
async fn test_custom_embedding_model() {
    let model = EmbeddingModel {
        model_name: "custom-model".to_string(),
        dimension: 512,
    };

    let engine = SearchEngine::with_model(model);
    assert!(engine.is_ok());
}

#[tokio::test]
async fn test_index_symbol() {
    let engine = SearchEngine::new().unwrap();
    let result = engine.index_symbol(1, "test symbol text").await;
    assert!(result.is_ok());
}
