//! SYNC-CHECK (#248 AC-1.8(b)): quiesce coverage is enumerable, not structural, so
//! this parses the source to fail when the enumeration changes — and pins what the
//! runtime tests cannot, that the guarded call is still inside the guarded block.

use std::path::{Path, PathBuf};

const SHARED: &str = ".quiesce().read()";
const WRITE: &str = ".quiesce.write()";
const GUARDED_CALL: &str = "upsert_entity_vectors";

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read src dir").flatten() {
        let p = entry.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// Every `file:line` in `lens-core/src` containing `needle`.
fn sites(needle: &str) -> Vec<(PathBuf, usize, String)> {
    let mut files = Vec::new();
    rust_files(&src_root(), &mut files);
    files.sort();
    let mut hits = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).expect("read source");
        for (i, line) in text.lines().enumerate() {
            if line.contains(needle) {
                hits.push((f.clone(), i + 1, line.to_string()));
            }
        }
    }
    hits
}

#[test]
fn shared_quiesce_guard_is_taken_at_exactly_one_site() {
    let hits = sites(SHARED);
    let rendered: Vec<String> = hits
        .iter()
        .map(|(f, l, _)| format!("{}:{l}", f.display()))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "the shared quiesce guard must be taken at exactly one site; found {rendered:?}. \
         A new acquisition is a coverage change that needs review, not a silent edit."
    );
    let (file, _, _) = &hits[0];
    assert!(
        file.ends_with("resolution/worker.rs"),
        "the shared guard belongs at the entity-vector write, not {}",
        file.display()
    );
}

#[test]
fn exclusive_quiesce_guard_is_taken_at_exactly_one_site() {
    let hits = sites(WRITE);
    let rendered: Vec<String> = hits
        .iter()
        .map(|(f, l, _)| format!("{}:{l}", f.display()))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "the exclusive quiesce guard must be taken at exactly one site; found {rendered:?}"
    );
    assert!(hits[0].0.ends_with("lib.rs"));
}

/// The runtime tests hold the guard and observe a relocation blocking, but they
/// cannot see WHICH statements the guard covers. This can: it fails if the Lance
/// write leaves the guarded block.
#[test]
fn the_guarded_block_still_contains_the_lance_write() {
    let hits = sites(SHARED);
    let (file, line_no, _) = hits.first().expect("shared guard site");
    let text = std::fs::read_to_string(file).expect("read worker");
    let lines: Vec<&str> = text.lines().collect();

    // Depth going negative closes the block the guard was declared in; everything
    // before that is the guarded region.
    let mut depth: i32 = 0;
    let mut region = String::new();
    for line in &lines[*line_no..] {
        let opens = line.matches('{').count() as i32;
        let closes = line.matches('}').count() as i32;
        if depth + opens - closes < 0 {
            break;
        }
        depth += opens - closes;
        region.push_str(line);
        region.push('\n');
    }

    assert!(
        region.contains(GUARDED_CALL),
        "`{GUARDED_CALL}` must stay INSIDE the quiesce-guarded block. Guard is at \
         {}:{line_no}, and the block it opens does not contain the call — the write \
         it exists to protect is running unguarded. Guarded region was:\n{region}",
        file.display()
    );
}
