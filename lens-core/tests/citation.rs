//! Integration tests for the #23a citation subsystem (public `lens-core` surface).
//!
//! `extract_citations`/`hydrate_locators` are pure and offline (hand-built
//! [`ContextUnit`]s / row maps). Only `load_chunk_locators` touches a DB — a temp
//! SQLite instance with migrations applied, still fully offline (no network, no
//! `LENS_RUN_MODEL_TESTS`).

use std::collections::HashMap;

use lens_core::{
    CITATION_PROMPT_INSTRUCTION, ChunkLocatorRow, Citation, LensEngine, Locator, extract_citations,
    hydrate_locators, load_chunk_locators,
};
use lens_core::{ContextUnit, HitSource, Provenance};

/// Builds a synthetic `ContextUnit` at a chosen `order_index` (deliberately
/// decoupled from Vec position so the positional-mapping invariant can be tested).
fn unit(source_id: &str, chunk_id: &str, locator: Option<&str>, order_index: usize) -> ContextUnit {
    ContextUnit {
        text: format!("text for {chunk_id}"),
        source_id: source_id.to_string(),
        chunk_id: chunk_id.to_string(),
        parent_id: None,
        locator: locator.map(str::to_string),
        order_index,
        provenance: Provenance {
            source: HitSource::Dense,
            graph_confidence: None,
        },
    }
}

// --- Step 1: payload serde shape -------------------------------------------

#[test]
fn citation_serde_round_trip() {
    let citation = Citation {
        source_id: "src-1".to_string(),
        ordinal: 1,
        locators: vec![Locator {
            chunk_id: "c1".to_string(),
            anchor: Some("Intro".to_string()),
            section_path: Some("Chapter 1 > Intro".to_string()),
            page: Some(3),
            char_start: Some(10),
            char_end: Some(42),
        }],
    };
    let json = serde_json::to_string(&citation).expect("serialize");
    let back: Citation = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(citation, back);
}

/// Locks the wire shape #23b binds to. Asserts `None` fields serialize as explicit
/// `null` (no `skip_serializing_if`) — one hydrated locator, one unhydrated.
#[test]
fn citation_serialized_shape() {
    let citation = Citation {
        source_id: "src-1".to_string(),
        ordinal: 1,
        locators: vec![
            Locator {
                chunk_id: "c1".to_string(),
                anchor: Some("Intro".to_string()),
                section_path: Some("Chapter 1 > Intro".to_string()),
                page: Some(3),
                char_start: Some(10),
                char_end: Some(42),
            },
            Locator {
                chunk_id: "c2".to_string(),
                anchor: Some("body#p2".to_string()),
                section_path: None,
                page: None,
                char_start: None,
                char_end: None,
            },
        ],
    };
    insta::assert_json_snapshot!(citation, @r#"
    {
      "source_id": "src-1",
      "ordinal": 1,
      "locators": [
        {
          "chunk_id": "c1",
          "anchor": "Intro",
          "section_path": "Chapter 1 > Intro",
          "page": 3,
          "char_start": 10,
          "char_end": 42
        },
        {
          "chunk_id": "c2",
          "anchor": "body#p2",
          "section_path": null,
          "page": null,
          "char_start": null,
          "char_end": null
        }
      ]
    }
    "#);
}

// --- Step 2: prompt-instruction template ------------------------------------

#[test]
fn prompt_instruction_is_instructional() {
    assert!(!CITATION_PROMPT_INSTRUCTION.is_empty());
    assert!(CITATION_PROMPT_INSTRUCTION.contains("[1]"));
    assert!(CITATION_PROMPT_INSTRUCTION.contains("inline"));
    // The template is a contract, not a unit list — it must forbid the mis-grammar.
    assert!(CITATION_PROMPT_INSTRUCTION.contains("[1,2]"));
}

// --- Step 4: extraction, mapping, grouping, ordering ------------------------

#[test]
fn maps_valid_markers_to_source_and_locator() {
    let units = [
        unit("src-a", "c1", Some("Sec A"), 0),
        unit("src-b", "c2", Some("Sec B"), 1),
    ];
    let citations = extract_citations("Foo [1] bar [2].", &units);
    assert_eq!(citations.len(), 2);
    assert_eq!(citations[0].source_id, "src-a");
    assert_eq!(citations[0].locators[0].chunk_id, "c1");
    assert_eq!(citations[0].locators[0].anchor.as_deref(), Some("Sec A"));
    // Extract leaves hydrated fields None.
    assert_eq!(citations[0].locators[0].page, None);
    assert_eq!(citations[0].locators[0].section_path, None);
    assert_eq!(citations[1].source_id, "src-b");
}

#[test]
fn out_of_range_and_malformed_dropped_no_panic() {
    let units = [unit("src-a", "c1", None, 0)];
    // [0] out-of-range low, [99] out-of-range high, [abc] malformed; [1] valid.
    let citations = extract_citations("[0] [99] [abc] [1]", &units);
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0].source_id, "src-a");
    assert_eq!(citations[0].ordinal, 1);
}

#[test]
fn ordinal_is_first_appearance_over_survivors() {
    // [3][99][1] with 3 units → [99] dropped, order = unit3 then unit1, ordinals 1,2.
    let units = [
        unit("src-a", "c1", None, 0),
        unit("src-b", "c2", None, 1),
        unit("src-c", "c3", None, 2),
    ];
    let citations = extract_citations("[3][99][1]", &units);
    assert_eq!(citations.len(), 2);
    assert_eq!(citations[0].source_id, "src-c");
    assert_eq!(citations[0].ordinal, 1);
    assert_eq!(citations[1].source_id, "src-a");
    assert_eq!(citations[1].ordinal, 2);
}

#[test]
fn duplicate_markers_collapse_and_dedup_locators() {
    // Same source, two distinct chunks → one Citation, two Locators.
    let units = [unit("src-a", "c1", None, 0), unit("src-a", "c2", None, 1)];
    let citations = extract_citations("[1][2]", &units);
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0].locators.len(), 2);

    // Same source, same chunk cited twice → one Citation, one Locator.
    let citations = extract_citations("[1][1]", &units);
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0].locators.len(), 1);
}

#[test]
fn empty_and_all_malformed_yield_no_citations() {
    let units = [unit("src-a", "c1", None, 0)];
    assert!(extract_citations("no markers here", &units).is_empty());
    assert!(extract_citations("[abc] [1,2] [ 1 ]", &units).is_empty());
    assert!(extract_citations("", &units).is_empty());
}

/// C1 — positional trust: `order_index` is deliberately scrambled relative to Vec
/// position. `[1]` must resolve to `units[0]` (position), NOT the unit whose
/// `order_index == 0`. This FAILS a `.find(|u| u.order_index == n-1)` implementation.
#[test]
fn resolves_by_slice_position_not_order_index() {
    let units = [
        unit("src-pos0", "c-pos0", None, 7),
        unit("src-pos1", "c-pos1", None, 2),
    ];
    let citations = extract_citations("[1]", &units);
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0].source_id, "src-pos0");
    assert_eq!(citations[0].locators[0].chunk_id, "c-pos0");
}

// --- Step 5a: pure hydration ------------------------------------------------

#[test]
fn hydrate_fills_present_rows_and_leaves_missing_none() {
    let mut citations = vec![Citation {
        source_id: "src-a".to_string(),
        ordinal: 1,
        locators: vec![
            Locator {
                chunk_id: "c1".to_string(),
                anchor: Some("a1".to_string()),
                section_path: None,
                page: None,
                char_start: None,
                char_end: None,
            },
            Locator {
                chunk_id: "c-missing".to_string(),
                anchor: Some("a2".to_string()),
                section_path: None,
                page: None,
                char_start: None,
                char_end: None,
            },
        ],
    }];
    let mut rows = HashMap::new();
    rows.insert(
        "c1".to_string(),
        ChunkLocatorRow {
            section_path: "Chapter 1".to_string(),
            page: Some(5),
            char_start: Some(1),
            char_end: Some(9),
        },
    );
    hydrate_locators(&mut citations, &rows);

    let hydrated = &citations[0].locators[0];
    assert_eq!(hydrated.section_path.as_deref(), Some("Chapter 1"));
    assert_eq!(hydrated.page, Some(5));
    assert_eq!(hydrated.char_start, Some(1));
    assert_eq!(hydrated.char_end, Some(9));
    // anchor is never touched by hydration.
    assert_eq!(hydrated.anchor.as_deref(), Some("a1"));

    let missing = &citations[0].locators[1];
    assert_eq!(missing.section_path, None);
    assert_eq!(missing.page, None);
}

// --- Step 5b: owned async DB filler (only DB-touching test) -----------------

#[tokio::test]
async fn load_chunk_locators_reads_real_rows() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;

    let nb = engine
        .create_notebook("nb", None, None)
        .await
        .expect("create notebook")
        .id
        .to_string();
    let source_id = uuid::Uuid::now_v7().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO sources (id, notebook_id, kind, title, status, locator, selected, \
         content_hash, enrichment_status, created_at) \
         VALUES (?, ?, 'text', 'seed', 'indexed', '/tmp/seed.txt', 1, 'h', NULL, ?)",
    )
    .bind(&source_id)
    .bind(&nb)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert source");

    // Chunk with full page/char metadata.
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, \
          token_start, token_end, page, char_start, char_end, block_type, created_at) \
         VALUES ('c1', ?, NULL, 'child', 1, 'Chapter 1 > Intro', 'hello', \
                 0, 1, 7, 10, 42, 'paragraph', ?)",
    )
    .bind(&source_id)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert chunk c1");

    // Chunk with NULL page/char (nullable columns).
    sqlx::query(
        "INSERT INTO chunks \
         (id, source_id, parent_id, kind, level, section_path, text, created_at) \
         VALUES ('c2', ?, NULL, 'child', 1, 'Chapter 2', 'world', ?)",
    )
    .bind(&source_id)
    .bind(&now)
    .execute(&pool)
    .await
    .expect("insert chunk c2");

    let ids = vec!["c1".to_string(), "c2".to_string(), "c-absent".to_string()];
    let rows = load_chunk_locators(&pool, &ids).await.expect("load");

    let c1 = rows.get("c1").expect("c1 present");
    assert_eq!(c1.section_path, "Chapter 1 > Intro");
    assert_eq!(c1.page, Some(7));
    assert_eq!(c1.char_start, Some(10));
    assert_eq!(c1.char_end, Some(42));

    let c2 = rows.get("c2").expect("c2 present");
    assert_eq!(c2.section_path, "Chapter 2");
    assert_eq!(c2.page, None);
    assert_eq!(c2.char_start, None);
    assert_eq!(c2.char_end, None);

    // An absent id is simply missing from the map.
    assert!(!rows.contains_key("c-absent"));
}

#[tokio::test]
async fn load_chunk_locators_empty_input_is_empty_map() {
    let engine = LensEngine::for_test().await;
    let pool = engine.pool().await;
    let rows = load_chunk_locators(&pool, &[]).await.expect("load");
    assert!(rows.is_empty());
}
