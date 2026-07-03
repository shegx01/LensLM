//! Dev/QA Embeddings Inspector (M4): read-only command that surfaces a source's chunks
//! and its notebook's active embedding-index stats. Gated behind `debug_assertions`
//! so it never appears on the release command surface.
#[cfg(debug_assertions)]
use lens_core::{EmbeddingStats, InspectorChunk, LensEngine, LensError};
#[cfg(debug_assertions)]
use serde::Serialize;

#[cfg(debug_assertions)]
#[derive(Serialize)]
pub struct InspectorResponse {
    pub chunks: Vec<InspectorChunk>,
    pub stats: Vec<EmbeddingStats>,
}

#[cfg(debug_assertions)]
#[tracing::instrument(skip_all)]
#[tauri::command]
pub async fn list_source_chunks(
    source_id: String,
    notebook_id: String,
    engine: tauri::State<'_, LensEngine>,
) -> Result<InspectorResponse, LensError> {
    let chunks = engine.list_source_chunks(&source_id).await?;
    let stats = engine.get_embedding_stats(&notebook_id).await?;
    Ok(InspectorResponse { chunks, stats })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lens_core::NotebookId;
    use tauri::Manager;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_list_source_chunks_command() {
        let app = tauri::test::mock_app();
        app.manage(LensEngine::for_test().await);
        let engine = app.state::<LensEngine>();

        let pool = engine.pool().await;
        let nb = engine
            .create_notebook("Notebook", None, None)
            .await
            .unwrap();
        let src = engine
            .add_source(
                &NotebookId::from(nb.id.to_string()),
                "doc.md",
                "/abs/doc.md",
            )
            .await
            .unwrap()
            .source;

        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO chunks \
                 (id, source_id, parent_id, kind, level, section_path, text, \
                  token_start, token_end, page, char_start, char_end, block_type, \
                  source_anchor, embedding_text, created_at) \
             VALUES (?, ?, NULL, 'parent', 0, 'Intro', 'parent text', 0, NULL, NULL, \
                  0, 40, 'heading', NULL, 'Intro: parent text', ?)",
        )
        .bind(Uuid::now_v7().to_string())
        .bind(&src.id)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO embedding_index \
                 (id, notebook_id, model, dim, prefix_convention, lance_table_name, status, created_at) \
             VALUES (?, ?, 'nomic-embed-text', 768, 'nomic', 'lance_nomic_768', 'active', ?)",
        )
        .bind(Uuid::now_v7().to_string())
        .bind(nb.id.to_string())
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

        let resp = list_source_chunks(src.id.clone(), nb.id.to_string(), engine)
            .await
            .unwrap();

        assert_eq!(resp.chunks.len(), 1, "the seeded chunk is returned");
        assert_eq!(resp.chunks[0].text, "parent text");
        assert_eq!(
            resp.stats.len(),
            1,
            "the active embedding index is returned"
        );
        assert_eq!(resp.stats[0].model, "nomic-embed-text");
        assert_eq!(resp.stats[0].dim, 768);
    }
}
