//! Data-directory relocation and model-cache offload (#238).
//!
//! Relocation is copy → verify → rewrite → (caller writes pointer) → restart, and
//! never mutates the live pool's location: the running engine keeps the old dir
//! until restart, when [`resolve_data_dir`] re-points it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::error::LensError;

/// Pointer file written into the OS anchor dir (the fixed `app_data_dir`, which
/// never moves) so a relocated data dir is found at next launch. Its absence means
/// data lives in the anchor itself (the pre-relocation default).
pub const LOCATION_FILE: &str = "location.json";
const LOCATION_PENDING: &str = "location.json.pending";

// The three enumerations below are DERIVED from `crate::layout::LAYOUT`; adding a
// subsystem there updates copy, cleanup and accounting together. `copy_tree` stays
// a DENY-list driven by `copy_skip()` — an allow-list over these would silently
// stop copying anything the descriptor does not name.

/// Pointer describing where the active data dir lives, plus an optional old dir
/// awaiting best-effort cleanup on the next successful boot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataLocation {
    pub data_dir: String,
    #[serde(default)]
    pub cleanup: Option<String>,
}

/// Resolves the active data dir from the anchor's pointer file, falling back to the
/// anchor when the pointer is absent, malformed, or names a directory that no longer
/// exists. Never errors — a bad pointer must not brick startup.
pub fn resolve_data_dir(anchor: &Path) -> PathBuf {
    match read_location(anchor) {
        Some(loc) => {
            let dir = PathBuf::from(&loc.data_dir);
            if !loc.data_dir.is_empty() && dir.is_dir() {
                dir
            } else {
                tracing::warn!(
                    pointer = %loc.data_dir,
                    "relocation pointer names a missing dir; using anchor"
                );
                anchor.to_path_buf()
            }
        }
        None => anchor.to_path_buf(),
    }
}

/// Reads and parses the anchor's pointer file. `None` when absent or unreadable;
/// a malformed pointer logs and returns `None` (treated as no relocation).
pub(crate) fn read_location(anchor: &Path) -> Option<DataLocation> {
    let path = anchor.join(LOCATION_FILE);
    let contents = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str(&contents) {
        Ok(loc) => Some(loc),
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "malformed relocation pointer");
            None
        }
    }
}

/// Atomically writes the pointer file into `anchor` (write-temp-then-rename so a
/// crash never leaves a half-written pointer that would strand the data dir).
pub fn write_location(anchor: &Path, loc: &DataLocation) -> Result<(), LensError> {
    std::fs::create_dir_all(anchor)
        .map_err(|e| LensError::Io(format!("failed to create anchor dir: {e}")))?;
    let json = serde_json::to_string_pretty(loc)?;
    let pending = anchor.join(LOCATION_PENDING);
    std::fs::write(&pending, json).map_err(|e| {
        tracing::error!(error = %e, "failed to write pending relocation pointer");
        LensError::Io("failed to write relocation pointer".into())
    })?;
    std::fs::rename(&pending, anchor.join(LOCATION_FILE)).map_err(|e| {
        tracing::error!(error = %e, "failed to promote relocation pointer");
        LensError::Io("failed to write relocation pointer".into())
    })
}

/// Best-effort cleanup of a superseded old data dir, run once on the next boot after
/// the engine opens cleanly on the new dir. Removes the known data entries plus any
/// caller-supplied `extra` regenerable dirs (never the directory itself or a pointer
/// file), then clears `cleanup` from the pointer so it runs at most once.
///
/// Data-safety guard (issue #238, C1): if the old dir's DB is NEWER than the active
/// dir's snapshot, the user wrote to the old dir after the relocation copy (e.g. a
/// crash or a stray write before restart). Deleting it would lose that work, so we
/// refuse and keep the old dir intact. A cleanup naming the active dir or anchor is
/// likewise refused.
pub async fn run_boot_cleanup(
    anchor: &Path,
    active_data_dir: &Path,
    extra: &[&str],
    pool: &SqlitePool,
) {
    let Some(mut loc) = read_location(anchor) else {
        return;
    };
    let Some(cleanup) = loc.cleanup.clone() else {
        return;
    };
    let old = PathBuf::from(&cleanup);
    // NOT `old == anchor`: the first relocation always leaves the anchor as the old
    // dir, and refusing there stranded the previous corpus invisibly. `data_entries()`
    // omits the pointer files, so the corpus goes and the pointer stays.
    if old == active_data_dir || cleanup.is_empty() {
        tracing::warn!(dir = %cleanup, "refusing to clean the active data dir");
        loc.cleanup = None;
        let _ = write_location(anchor, &loc);
        return;
    }
    if old_dir_is_newer(&old, active_data_dir) {
        erase_config_files(&old);
        tracing::warn!(
            reason = "old_dir_newer",
            dir = %cleanup,
            "old data dir has newer writes than the relocated snapshot; keeping it \
             intact and leaving the cleanup marker for a later manual reclaim"
        );
        return;
    }

    // AC-4.6: verify the dir we are about to depend on exclusively, at the only
    // moment both still exist. On refusal keep the marker so the old dir survives
    // for manual reclaim, and name the check that failed.
    if let Err(reason) = verify_before_delete(pool, &old, active_data_dir).await {
        erase_config_files(&old);
        tracing::warn!(
            reason = %reason,
            dir = %cleanup,
            "refusing to delete the old data dir; keeping it and the cleanup marker"
        );
        return;
    }
    for entry in crate::layout::data_entries().iter().chain(extra) {
        let path = old.join(entry);
        if let Err(e) = remove_if_exists(&path) {
            tracing::warn!(path = %path.display(), error = %e, "old-dir cleanup entry failed");
        }
    }
    loc.cleanup = None;
    if let Err(e) = write_location(anchor, &loc) {
        tracing::warn!(error = %e, "failed to clear cleanup marker after boot cleanup");
    }
}

/// AC-4.10: erases `config.json*` from `dir` — the old dir even when cleanup is
/// refused, and the target during a rollback —
/// it holds a plaintext cloud API key and is settings, not corpus, so erasing closes
/// an otherwise indefinite exposure without touching what the refusal protects.
fn erase_config_files(dir: &Path) {
    // By PREFIX: `config.json.bak` and the `.corrupt-*.bak` siblings carry the same
    // `api_key`, so erasing only the exact name leaves the secret behind.
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.starts_with("config.json"))
            && let Err(e) = remove_if_exists(&entry.path())
        {
            tracing::warn!(error = %e, "failed to erase an old config during a refused cleanup");
        }
    }
}

/// The AC-4.1/4.2 checks, re-run against the dir that is about to become the only
/// copy. Returns the name of the failing check so the refusal says which one.
async fn verify_before_delete(
    pool: &SqlitePool,
    old: &Path,
    active: &Path,
) -> Result<(), &'static str> {
    if verify_active_vector_tables(pool, active).await.is_err() {
        return Err("active_vector_table_unreadable");
    }
    let (Ok(old_prefix), Ok(active_prefix)) = (prefix_of(old), prefix_of(active)) else {
        return Err("path_not_utf8");
    };
    if verify_ready_audio(pool, &old_prefix, &active_prefix)
        .await
        .is_err()
    {
        return Err("ready_audio_missing_from_copy");
    }
    Ok(())
}

/// True when `old/lens.db` has a strictly later mtime than `active/lens.db` — the
/// signal that the old dir received writes after the relocation snapshot. On any
/// stat ambiguity returns `true` (refuse to delete) — the safe default for a
/// data-loss guard.
fn old_dir_is_newer(old: &Path, active: &Path) -> bool {
    let mtime = |dir: &Path| std::fs::metadata(dir.join("lens.db")).and_then(|m| m.modified());
    match (mtime(old), mtime(active)) {
        (Ok(old_t), Ok(active_t)) => old_t > active_t,
        // Old DB missing → nothing worth keeping. Active DB missing/unreadable →
        // ambiguous, so keep the old dir rather than risk deleting live data.
        (Err(_), _) => false,
        (Ok(_), Err(_)) => true,
    }
}

fn remove_if_exists(path: &Path) -> Result<(), LensError> {
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(LensError::Io(format!("stat failed: {e}"))),
    };
    let res = if meta.file_type().is_dir() && !meta.file_type().is_symlink() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
    res.map_err(|e| LensError::Io(format!("remove failed: {e}")))
}

/// AC-4.1: opens the copied `active` vector tables. Deliberately NOT every table —
/// `building`/`stale` are GC-reclaimed and not load-bearing, so a mid-write one
/// captured by the copy must not abort an otherwise valid multi-minute move.
async fn verify_active_vector_tables(new_pool: &SqlitePool, to: &Path) -> Result<(), LensError> {
    let names: Vec<String> =
        sqlx::query_scalar("SELECT lance_table_name FROM embedding_index WHERE status = 'active'")
            .fetch_all(new_pool)
            .await?;
    if names.is_empty() {
        return Ok(());
    }
    let store = crate::vector_store::LanceVectorStore::new(to, new_pool.clone());
    store.verify_tables_open(&names).await
}

/// AC-4.2: `ready` rows only, failing only when the file exists in the source and is
/// missing from the copy. `ready` legitimately outlives its file, so a pre-dangling
/// row warns rather than blocking — a blanket check would refuse those users.
async fn verify_ready_audio(
    new_pool: &SqlitePool,
    old_prefix: &str,
    new_prefix: &str,
) -> Result<(), LensError> {
    let paths: Vec<String> =
        sqlx::query_scalar("SELECT path FROM audio_overviews WHERE status = 'ready'")
            .fetch_all(new_pool)
            .await?;
    for copied in paths {
        // `rewrite_paths` already ran, so `copied` is new-prefixed; map back to find
        // what the source held.
        let Some(rel) = copied.strip_prefix(new_prefix) else {
            continue;
        };
        let source = PathBuf::from(format!("{old_prefix}{rel}"));
        if !source.exists() {
            tracing::warn!(
                reason = "audio_pre_dangling",
                "a ready audio overview had no file before the move; not blocking"
            );
            continue;
        }
        if !PathBuf::from(&copied).exists() {
            tracing::error!(
                reason = "audio_copy_missing",
                "a ready audio overview did not copy"
            );
            return Err(LensError::Io(
                "an audio overview did not copy; the move was cancelled".into(),
            ));
        }
    }
    Ok(())
}

/// #248 AC-4.8 test seam: a path deleted after `read_dir` has listed it, so the
/// vanishing-entry window is deterministic instead of a race.
#[cfg(feature = "test-util")]
pub static VANISH_DURING_COPY: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// The old data dir a refused cleanup is holding onto, with the bytes it occupies.
/// `None` when nothing is retained. Read-only: reclaiming it stays manual, because
/// deletion is the thing that needed verifying in the first place.
pub fn retained_cleanup(anchor: &Path) -> Option<(String, u64)> {
    let cleanup = read_location(anchor)?.cleanup?;
    if cleanup.is_empty() {
        return None;
    }
    let path = PathBuf::from(&cleanup);
    if !path.exists() {
        return None;
    }
    let bytes = crate::storage::path_size_bytes(&path).unwrap_or(0);
    Some((cleanup, bytes))
}

/// Empties `dir` in place. Used on a relocation failure: `validate_target` accepts
/// an existing empty folder, so preserving the directory keeps a retry unblocked
/// while still leaving no residue.
fn clear_dir_contents(dir: &Path) -> Result<(), LensError> {
    // Secrets first, then continue past a failure: `?` here aborted the sweep on the
    // first un-removable entry, so whether the plaintext config survived depended on
    // readdir order.
    erase_config_files(dir);
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "could not read the target while rolling back");
            return Ok(());
        }
    };
    let mut failed = 0usize;
    for entry in entries.flatten() {
        if let Err(e) = remove_if_exists(&entry.path()) {
            failed += 1;
            tracing::warn!(error = %e, "failed to remove an entry while rolling back");
        }
    }
    if failed > 0 {
        tracing::warn!(failed, "rollback left entries behind in the target");
    }
    Ok(())
}

/// Recursively copies `from` into `to`, skipping symlinks (never followed — matches
/// the storage-scan discipline) and any top-level names in `skip`. `to` is created.
fn copy_tree(from: &Path, to: &Path, skip: &[&str]) -> Result<(), LensError> {
    std::fs::create_dir_all(to)
        .map_err(|e| LensError::Io(format!("failed to create copy target: {e}")))?;
    let entries = std::fs::read_dir(from)
        .map_err(|e| LensError::Io(format!("failed to read source dir: {e}")))?;
    for entry in entries {
        let entry = entry.map_err(|e| LensError::Io(format!("dir entry failed: {e}")))?;
        let name = entry.file_name();
        if name.to_str().is_some_and(|n| skip.contains(&n)) {
            continue;
        }
        // Deleting HERE — after `read_dir` yielded the entry, before the stat — is the
        // only way to reproduce the vanishing-entry window: `read_dir` is lazy, so
        // removing the file earlier just means it is never listed.
        #[cfg(feature = "test-util")]
        if let Ok(mut slot) = VANISH_DURING_COPY.lock()
            && slot.as_deref() == Some(entry.path().as_path())
        {
            let _ = std::fs::remove_file(entry.path());
            slot.take();
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!(
                    reason = "copy_entry_vanished",
                    "an entry disappeared mid-copy"
                );
                continue;
            }
            Err(e) => return Err(LensError::Io(format!("stat failed: {e}"))),
        };
        let src = entry.path();
        let dst = to.join(&name);
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            copy_tree(&src, &dst, &[])?;
        } else {
            match std::fs::copy(&src, &dst) {
                Ok(_) => {}
                // Only when the SOURCE went away: `fs::copy` also reports NotFound for
                // a missing destination, and skipping that drops a needed file while
                // reporting success — after which cleanup deletes the original.
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && !src.exists() => {
                    tracing::warn!(
                        reason = "copy_entry_vanished",
                        "an entry disappeared mid-copy"
                    );
                    continue;
                }
                Err(e) => return Err(LensError::Io(format!("copy failed: {e}"))),
            }
        }
    }
    Ok(())
}

/// Rejects a target that is the source itself, nested inside it, or (for a target
/// that already exists) non-empty. A cross-device target is fine — we always copy.
fn validate_target(from: &Path, to: &Path) -> Result<(), LensError> {
    // AC-5.3: `starts_with` is LEXICAL, so a symlinked parent can present a nested
    // target as unrelated — and nesting the target inside the source makes the copy
    // walk into its own output.
    let (from_c, to_c) = (canonical_or_ancestor(from), canonical_or_ancestor(to));
    if to_c == from_c {
        return Err(LensError::Validation(
            "the new location is the same as the current one".into(),
        ));
    }
    if to_c.starts_with(&from_c) {
        return Err(LensError::Validation(
            "the new location cannot be inside the current data folder".into(),
        ));
    }
    let nonempty = std::fs::read_dir(to).is_ok_and(|mut e| e.next().is_some());
    if nonempty {
        return Err(LensError::Validation(
            "the new location must be an empty folder".into(),
        ));
    }
    Ok(())
}

/// Resolves symlinks and `..` in `path`. A target that does not exist yet resolves
/// its nearest existing ancestor and re-appends the tail, so a not-yet-created
/// folder is still compared on its real location.
fn canonical_or_ancestor(path: &Path) -> PathBuf {
    if let Ok(c) = path.canonicalize() {
        return c;
    }
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = path;
    while let Some(parent) = cur.parent() {
        if let Some(name) = cur.file_name() {
            tail.push(name.to_os_string());
        }
        if let Ok(c) = parent.canonicalize() {
            let mut out = c;
            for seg in tail.iter().rev() {
                out = out.join(seg);
            }
            return out;
        }
        cur = parent;
    }
    path.to_path_buf()
}

/// Snapshots the live DB into `dest` with `VACUUM INTO` — transactionally consistent
/// even while the pool is open, and it flushes the WAL as part of the vacuum.
async fn snapshot_db(pool: &SqlitePool, dest: &Path) -> Result<(), LensError> {
    let dest = dest
        .to_str()
        .ok_or_else(|| LensError::Validation("the new location path is not valid UTF-8".into()))?;
    // VACUUM INTO takes a string literal, not a bind param; escape single quotes.
    let escaped = dest.replace('\'', "''");
    sqlx::query(&format!("VACUUM INTO '{escaped}'"))
        .execute(pool)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "VACUUM INTO snapshot failed");
            LensError::Io("failed to snapshot the database".into())
        })?;
    Ok(())
}

async fn source_count(pool: &SqlitePool) -> Result<i64, LensError> {
    // `?` routes any sqlx error through the opaque `From<sqlx::Error>` so no DB
    // internals cross the IPC boundary.
    Ok(sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM sources")
        .fetch_one(pool)
        .await?)
}

/// Prefix-rewrites the two DB columns that store absolute paths under the data dir
/// so they point into `new_prefix`. URL `sources.locator`s (and any path not under
/// `old_prefix`) are left untouched because they never start with `old_prefix`.
async fn rewrite_paths(
    pool: &SqlitePool,
    old_prefix: &str,
    new_prefix: &str,
) -> Result<(), LensError> {
    let sources = sqlx::query_as::<_, (String, String)>("SELECT id, locator FROM sources")
        .fetch_all(pool)
        .await?;
    for (id, locator) in sources {
        if let Some(rest) = locator.strip_prefix(old_prefix) {
            let updated = format!("{new_prefix}{rest}");
            sqlx::query("UPDATE sources SET locator = ? WHERE id = ?")
                .bind(&updated)
                .bind(&id)
                .execute(pool)
                .await?;
        }
    }

    let overviews =
        sqlx::query_as::<_, (String, String)>("SELECT notebook_id, path FROM audio_overviews")
            .fetch_all(pool)
            .await?;
    for (notebook_id, path) in overviews {
        if let Some(rest) = path.strip_prefix(old_prefix) {
            let updated = format!("{new_prefix}{rest}");
            sqlx::query("UPDATE audio_overviews SET path = ? WHERE notebook_id = ?")
                .bind(&updated)
                .bind(&notebook_id)
                .execute(pool)
                .await?;
        }
    }
    Ok(())
}

/// A path as a prefix string with any trailing separator stripped, so joining
/// `strip_prefix` remainders (which start with a separator) never yields `//`.
fn prefix_of(path: &Path) -> Result<String, LensError> {
    let s = path
        .to_str()
        .ok_or_else(|| LensError::Validation("a data path is not valid UTF-8".into()))?;
    Ok(s.strip_suffix(std::path::MAIN_SEPARATOR)
        .unwrap_or(s)
        .to_string())
}

/// Copies the data dir `from` → `to`, verifies the snapshot, and rewrites the
/// absolute-path DB columns in the copy. `extra_skip` names regenerable top-level
/// dirs the caller wants re-provisioned at the destination rather than copied (e.g.
/// the Qwen sidecar venv). On ANY failure the target's CONTENTS are removed so a
/// retry is not blocked and no private residue is left behind — the directory itself
/// is preserved, since deleting a folder the user chose is its own surprise. On
/// success the caller writes the anchor pointer (with `cleanup = from`) and prompts a
/// restart; the live pool is never touched. Runs behind the engine `ingest_lock`.
pub async fn relocate_data_dir(
    pool: &SqlitePool,
    from: &Path,
    to: &Path,
    extra_skip: &[&str],
) -> Result<(), LensError> {
    validate_target(from, to)?;
    let old_prefix = prefix_of(from)?;
    let new_prefix = prefix_of(to)?;
    let expected_sources = source_count(pool).await?;

    std::fs::create_dir_all(to)
        .map_err(|e| LensError::Io(format!("failed to create the new location: {e}")))?;

    // Any failure after the target dir exists must leave nothing OF OURS behind: a
    // partial corpus plus a copied config.json holding a plaintext api_key would
    // strand in a user folder and block retry.
    let result = relocate_into(
        pool,
        from,
        to,
        &old_prefix,
        &new_prefix,
        expected_sources,
        extra_skip,
    )
    .await;
    if result.is_err() {
        let _ = clear_dir_contents(to);
    }
    result
}

async fn relocate_into(
    pool: &SqlitePool,
    from: &Path,
    to: &Path,
    old_prefix: &str,
    new_prefix: &str,
    expected_sources: i64,
    extra_skip: &[&str],
) -> Result<(), LensError> {
    snapshot_db(pool, &to.join("lens.db")).await?;
    // Owned, because the multi-GB walk moves to the blocking pool rather than
    // pinning a runtime worker while the exclusive quiesce guard is held.
    let skip: Vec<String> = crate::layout::copy_skip()
        .iter()
        .chain(extra_skip.iter())
        .map(|s| (*s).to_string())
        .collect();
    let (copy_from, copy_to) = (from.to_path_buf(), to.to_path_buf());
    tokio::task::spawn_blocking(move || {
        let skip_refs: Vec<&str> = skip.iter().map(String::as_str).collect();
        copy_tree(&copy_from, &copy_to, &skip_refs)
    })
    .await
    .map_err(|e| LensError::Io(format!("data-dir copy task panicked: {e}")))??;

    // Verify + rewrite against the copied DB via its own pool; close it on every
    // path so no stray connection holds the new file open.
    let new_pool = crate::db::open_pool(to).await?;
    let verified = async {
        if source_count(&new_pool).await? != expected_sources {
            return Err(LensError::Io(
                "the copied database did not verify; the move was cancelled".into(),
            ));
        }
        rewrite_paths(&new_pool, old_prefix, new_prefix).await?;
        verify_active_vector_tables(&new_pool, to).await?;
        verify_ready_audio(&new_pool, old_prefix, new_prefix).await
    }
    .await;
    new_pool.close().await;
    verified
}

/// Copies the model cache (`models/` + `hf-cache/`) from `from_root` to `to_root`
/// WITHOUT deleting the originals, returning the bytes copied. The caller persists
/// `config.paths.cache_dir` and only then calls [`remove_cache`] on the old root, so
/// a persist failure never strands the cache moved-but-unreferenced. Runs behind the
/// engine `ingest_lock`.
pub fn copy_cache(from_root: &Path, to_root: &Path) -> Result<u64, LensError> {
    if to_root == from_root {
        return Err(LensError::Validation(
            "the new cache location is the same as the current one".into(),
        ));
    }
    if to_root.starts_with(from_root) || from_root.starts_with(to_root) {
        return Err(LensError::Validation(
            "the cache location and data folder cannot be nested".into(),
        ));
    }
    let mut copied = 0u64;
    for entry in crate::layout::cache_entries() {
        let src = from_root.join(entry);
        if !src.exists() {
            continue;
        }
        let dst = to_root.join(entry);
        if dst.exists() {
            return Err(LensError::Validation(
                "the new cache location already contains a model cache".into(),
            ));
        }
        copy_tree(&src, &dst, &[])?;
        copied = copied.saturating_add(crate::storage::path_size_bytes(&dst)?);
    }
    Ok(copied)
}

/// Removes the model-cache dirs under `root` (the old cache root after a successful
/// offload, or the offload target when resetting). Best-effort per entry.
pub fn remove_cache(root: &Path) -> Result<(), LensError> {
    for entry in crate::layout::cache_entries() {
        remove_if_exists(&root.join(entry))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_falls_back_to_anchor_without_pointer() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(resolve_data_dir(dir.path()), dir.path());
    }

    #[test]
    fn resolve_reads_valid_pointer() {
        let anchor = tempfile::tempdir().expect("anchor");
        let target = tempfile::tempdir().expect("target");
        write_location(
            anchor.path(),
            &DataLocation {
                data_dir: target.path().display().to_string(),
                cleanup: None,
            },
        )
        .expect("write");
        assert_eq!(resolve_data_dir(anchor.path()), target.path());
    }

    #[test]
    fn resolve_falls_back_when_target_missing() {
        let anchor = tempfile::tempdir().expect("anchor");
        write_location(
            anchor.path(),
            &DataLocation {
                data_dir: "/nonexistent/lens-data".into(),
                cleanup: None,
            },
        )
        .expect("write");
        assert_eq!(resolve_data_dir(anchor.path()), anchor.path());
    }

    #[test]
    fn validate_rejects_nested_and_same_target() {
        let from = tempfile::tempdir().expect("from");
        let nested = from.path().join("sub");
        std::fs::create_dir_all(&nested).expect("mkdir");
        assert!(validate_target(from.path(), from.path()).is_err());
        assert!(validate_target(from.path(), &nested).is_err());
    }

    /// A `..` path resolving back inside the data dir is rejected. NOT a proof of
    /// canonicalization — `from/sub/../sub` is lexically nested too, so this passes
    /// either way; the symlink case below is the discriminating one.
    #[test]
    fn validate_rejects_a_dotdot_path_that_lands_inside_the_source() {
        let from = tempfile::tempdir().expect("from");
        let nested = from.path().join("sub");
        std::fs::create_dir_all(&nested).expect("mkdir");
        let sneaky = nested.join("..").join("sub");
        assert!(
            validate_target(from.path(), &sneaky).is_err(),
            "a `..` path resolving inside the data dir must be rejected"
        );
    }

    /// AC-5.3: a symlinked parent hides nesting from a lexical check entirely.
    #[cfg(unix)]
    #[test]
    fn validate_rejects_a_symlink_pointing_into_the_source() {
        let from = tempfile::tempdir().expect("from");
        let elsewhere = tempfile::tempdir().expect("elsewhere");
        let inside = from.path().join("inside");
        std::fs::create_dir_all(&inside).expect("mkdir");
        let link = elsewhere.path().join("link");
        std::os::unix::fs::symlink(&inside, &link).expect("symlink");
        assert!(
            validate_target(from.path(), &link).is_err(),
            "a symlink resolving into the data dir must be rejected"
        );
    }

    /// Canonicalization must not break the normal case: the picker hands back a
    /// folder that does not exist yet.
    #[test]
    fn validate_accepts_a_target_that_does_not_exist_yet() {
        let from = tempfile::tempdir().expect("from");
        let parent = tempfile::tempdir().expect("parent");
        let target = parent.path().join("not-created-yet");
        assert!(
            validate_target(from.path(), &target).is_ok(),
            "a not-yet-created target must still be accepted"
        );
    }

    #[test]
    fn validate_rejects_nonempty_target() {
        let from = tempfile::tempdir().expect("from");
        let to = tempfile::tempdir().expect("to");
        std::fs::write(to.path().join("x"), b"1").expect("seed");
        assert!(validate_target(from.path(), to.path()).is_err());
    }

    /// Empty migrated pool. These tests exercise the pointer/mtime logic, and both
    /// AC-4.6 checks no-op on an empty registry.
    async fn probe_pool() -> SqlitePool {
        let pool = crate::db::open_in_memory_pool().await.expect("pool");
        crate::db::run_migrations(&pool).await.expect("migrate");
        pool
    }

    /// Writes `db` seed and stamps its mtime to `UNIX_EPOCH + secs` so the C1
    /// old-vs-active freshness guard is deterministic (no sleeps).
    fn seed_db_at(dir: &Path, secs: u64) {
        std::fs::write(dir.join("lens.db"), b"db").expect("seed db");
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join("lens.db"))
            .expect("open db");
        f.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .expect("set mtime");
    }

    #[tokio::test]
    async fn boot_cleanup_removes_old_data_but_keeps_pointer_and_dir() {
        let anchor = tempfile::tempdir().expect("anchor");
        let old = tempfile::tempdir().expect("old");
        let active = tempfile::tempdir().expect("active");
        seed_db_at(old.path(), 100);
        seed_db_at(active.path(), 200); // active snapshot newer → old safe to delete
        std::fs::create_dir_all(old.path().join("sources")).expect("seed");
        write_location(
            anchor.path(),
            &DataLocation {
                data_dir: active.path().display().to_string(),
                cleanup: Some(old.path().display().to_string()),
            },
        )
        .expect("write");

        run_boot_cleanup(anchor.path(), active.path(), &[], &probe_pool().await).await;

        assert!(!old.path().join("lens.db").exists());
        assert!(!old.path().join("sources").exists());
        assert!(old.path().exists(), "old dir itself preserved");
        let loc = read_location(anchor.path()).expect("pointer survives");
        assert_eq!(loc.cleanup, None, "cleanup marker cleared");
    }

    #[tokio::test]
    async fn boot_cleanup_removes_extra_regenerable_dirs() {
        let anchor = tempfile::tempdir().expect("anchor");
        let old = tempfile::tempdir().expect("old");
        let active = tempfile::tempdir().expect("active");
        seed_db_at(old.path(), 100);
        seed_db_at(active.path(), 200);
        std::fs::create_dir_all(old.path().join("qwen-venv")).expect("venv");
        write_location(
            anchor.path(),
            &DataLocation {
                data_dir: active.path().display().to_string(),
                cleanup: Some(old.path().display().to_string()),
            },
        )
        .expect("write");

        run_boot_cleanup(
            anchor.path(),
            active.path(),
            &["qwen-venv"],
            &probe_pool().await,
        )
        .await;
        assert!(!old.path().join("qwen-venv").exists(), "extra dir cleaned");
    }

    #[tokio::test]
    async fn boot_cleanup_refuses_when_old_dir_newer() {
        let anchor = tempfile::tempdir().expect("anchor");
        let old = tempfile::tempdir().expect("old");
        let active = tempfile::tempdir().expect("active");
        // Old dir has newer writes than the relocated snapshot → data-loss guard.
        seed_db_at(old.path(), 200);
        seed_db_at(active.path(), 100);
        write_location(
            anchor.path(),
            &DataLocation {
                data_dir: active.path().display().to_string(),
                cleanup: Some(old.path().display().to_string()),
            },
        )
        .expect("write");

        run_boot_cleanup(anchor.path(), active.path(), &[], &probe_pool().await).await;
        assert!(old.path().join("lens.db").exists(), "newer old data kept");
        let loc = read_location(anchor.path()).expect("pointer survives");
        assert_eq!(
            loc.cleanup,
            Some(old.path().display().to_string()),
            "cleanup marker retained for manual reclaim"
        );
    }

    #[tokio::test]
    async fn boot_cleanup_refuses_active_dir() {
        let anchor = tempfile::tempdir().expect("anchor");
        let active = tempfile::tempdir().expect("active");
        std::fs::write(active.path().join("lens.db"), b"db").expect("seed");
        write_location(
            anchor.path(),
            &DataLocation {
                data_dir: active.path().display().to_string(),
                cleanup: Some(active.path().display().to_string()),
            },
        )
        .expect("write");

        run_boot_cleanup(anchor.path(), active.path(), &[], &probe_pool().await).await;
        assert!(
            active.path().join("lens.db").exists(),
            "active data untouched"
        );
    }

    #[tokio::test]
    async fn relocate_copies_and_rewrites_absolute_paths() {
        let from = tempfile::tempdir().expect("from");
        let to_parent = tempfile::tempdir().expect("to_parent");
        let to = to_parent.path().join("moved");

        let pool = crate::db::open_pool(from.path()).await.expect("pool");
        crate::db::run_migrations(&pool).await.expect("migrate");

        // A notebook + one file source (absolute locator) + one URL source.
        sqlx::query(
            "INSERT INTO notebooks (id, title, created_at, updated_at) VALUES ('nb1','N','t','t')",
        )
        .execute(&pool)
        .await
        .expect("nb");
        let file_locator = from.path().join("sources").join("a.pdf");
        sqlx::query("INSERT INTO sources (id, notebook_id, kind, status, title, locator, created_at) VALUES ('s1','nb1','file','ready','A',?,'t')")
            .bind(file_locator.display().to_string())
            .execute(&pool)
            .await
            .expect("file source");
        sqlx::query("INSERT INTO sources (id, notebook_id, kind, status, title, locator, created_at) VALUES ('s2','nb1','url','ready','U','https://example.com/x','t')")
            .execute(&pool)
            .await
            .expect("url source");
        sqlx::query("INSERT INTO audio_overviews (notebook_id, path, generated_at, status, source_set_hash) VALUES ('nb1', ?, 't', 'ready', 'h')")
            .bind(from.path().join("notebooks").join("nb1").join("overview.wav").display().to_string())
            .execute(&pool)
            .await
            .expect("audio");
        std::fs::create_dir_all(from.path().join("sources")).expect("sources dir");
        std::fs::write(from.path().join("sources").join("a.pdf"), b"pdf").expect("file");

        relocate_data_dir(&pool, from.path(), &to, &[])
            .await
            .expect("relocate");

        let moved = crate::db::open_pool(&to).await.expect("new pool");
        let file: String = sqlx::query_scalar("SELECT locator FROM sources WHERE id='s1'")
            .fetch_one(&moved)
            .await
            .expect("s1");
        assert_eq!(file, to.join("sources").join("a.pdf").display().to_string());
        let url: String = sqlx::query_scalar("SELECT locator FROM sources WHERE id='s2'")
            .fetch_one(&moved)
            .await
            .expect("s2");
        assert_eq!(url, "https://example.com/x", "URL locator untouched");
        let audio: String =
            sqlx::query_scalar("SELECT path FROM audio_overviews WHERE notebook_id='nb1'")
                .fetch_one(&moved)
                .await
                .expect("audio");
        assert_eq!(
            audio,
            to.join("notebooks")
                .join("nb1")
                .join("overview.wav")
                .display()
                .to_string()
        );
        assert!(
            to.join("sources").join("a.pdf").exists(),
            "source file copied"
        );
        moved.close().await;
    }

    #[test]
    fn copy_cache_then_remove_moves_models_and_hf_cache() {
        let data = tempfile::tempdir().expect("data");
        let cache_parent = tempfile::tempdir().expect("cache_parent");
        let cache = cache_parent.path().join("offloaded");
        std::fs::create_dir_all(data.path().join("models").join("whisper")).expect("m");
        std::fs::write(
            data.path().join("models").join("whisper").join("g.bin"),
            [0u8; 100],
        )
        .expect("w");
        std::fs::create_dir_all(data.path().join("hf-cache")).expect("hf");
        std::fs::write(data.path().join("hf-cache").join("s.bin"), [0u8; 50]).expect("s");

        // copy_cache leaves the originals intact (caller persists config first).
        let copied = copy_cache(data.path(), &cache).expect("copy");
        assert_eq!(copied, 150);
        assert!(
            data.path().join("models").exists(),
            "originals kept until persist"
        );
        assert!(cache.join("models").join("whisper").join("g.bin").exists());
        assert!(cache.join("hf-cache").join("s.bin").exists());

        // remove_cache clears the old root after the config write.
        remove_cache(data.path()).expect("remove");
        assert!(!data.path().join("models").exists(), "old models removed");
        assert!(
            !data.path().join("hf-cache").exists(),
            "old hf-cache removed"
        );
    }

    #[test]
    fn copy_cache_refuses_when_target_already_has_cache() {
        let data = tempfile::tempdir().expect("data");
        let cache_parent = tempfile::tempdir().expect("cache_parent");
        let cache = cache_parent.path().join("offloaded");
        std::fs::create_dir_all(data.path().join("models")).expect("m");
        std::fs::create_dir_all(cache.join("models")).expect("pre-existing");
        assert!(copy_cache(data.path(), &cache).is_err());
    }
}
