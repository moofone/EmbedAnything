use embed_anything::embeddings::embed::EmbedderBuilder;
use embed_anything::embed_query;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Helper to load a model and assert expected dims
    async fn run_model(model_id: &str, expected: usize) -> anyhow::Result<()> {
        let model = EmbedderBuilder::new()
            .model_architecture("jina")
            .model_id(Some(model_id))
            .dtype(None)
            .from_pretrained_hf()?;

        let sentences = vec![
            "find functions related to poml",
            "validate schema for PRD yaml",
        ];

        let out = embed_query(&sentences, &model, None).await?;
        let dims = out[0].embedding.to_dense()?.len();
        println!(
            "model={} dims={} first3={:?}",
            model_id,
            dims,
            &out[0].embedding.to_dense()?[..3]
        );
        assert_eq!(dims, expected, "expected {}-d embeddings for {}", expected, model_id);
        Ok(())
    }

    // 1) Jina v2 Base Code -> 768 dims
    run_model("jinaai/jina-embeddings-v2-base-code", 768).await?;
    // 2) Jina v2 Small EN -> 512 dims (legacy layout fallback path)
    run_model("jinaai/jina-embeddings-v2-small-en", 512).await?;

    Ok(())
}
