//! Shared prompt infrastructure: the nonce fence (wraps untrusted source text so
//! injected text cannot forge a boundary) and [`PromptStore`], which serves each
//! prompt's editable creative body from a bundled default or a `{data_dir}/prompts/`
//! override. The security envelope (fence guard + JSON-schema contract) is composed
//! by code AROUND the template, never inside it, so an edit cannot remove the guard
//! or break the schema.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// A logical prompt template. Each variant maps to a bundled default (compiled in)
/// and a relative path under `{data_dir}/prompts/` where a user override may live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptName {
    /// Editable creative body of the audio-overview dialogue system prompt.
    DialogueSystem,
}

impl PromptName {
    /// Path (relative to the prompts root) of the override file for this template.
    pub(crate) fn relpath(self) -> &'static str {
        match self {
            PromptName::DialogueSystem => "dialogue/script.system.md",
        }
    }

    /// The compiled-in default body. Always valid — the app is usable with no
    /// override files present.
    pub(crate) fn embedded_default(self) -> &'static str {
        match self {
            PromptName::DialogueSystem => {
                include_str!("../prompts/dialogue/script.system.md")
            }
        }
    }
}

/// Resolves prompt templates: a user override under `{data_dir}/prompts/` when
/// present and readable, else the compiled-in default. Cheap to construct; a
/// template is read at most once per LLM call (never a hot path).
#[derive(Debug, Clone)]
pub struct PromptStore {
    /// `{data_dir}/prompts` when that directory exists, else `None` (defaults only).
    override_dir: Option<PathBuf>,
}

impl PromptStore {
    /// A store with no override directory — always serves compiled-in defaults.
    /// Test-only today; drop the gate when a production caller needs a
    /// default-only store.
    #[cfg(test)]
    pub(crate) fn embedded() -> Self {
        Self { override_dir: None }
    }

    /// A store rooted at `{data_dir}/prompts`. The directory is probed once here;
    /// if absent, this behaves exactly like [`PromptStore::embedded`].
    pub fn for_data_dir(data_dir: &Path) -> Self {
        let dir = data_dir.join("prompts");
        Self {
            override_dir: dir.is_dir().then_some(dir),
        }
    }

    /// Loads a template body: the override file if it exists and reads cleanly, else
    /// the compiled-in default. A read error on an override falls back to the
    /// default (a malformed override never breaks generation).
    pub(crate) fn load(&self, name: PromptName) -> Cow<'static, str> {
        if let Some(dir) = &self.override_dir {
            let path = dir.join(name.relpath());
            if let Ok(body) = std::fs::read_to_string(&path) {
                return Cow::Owned(body);
            }
        }
        Cow::Borrowed(name.embedded_default())
    }

    /// Loads `name` and substitutes each `{{key}}` placeholder with its value.
    /// Double braces are used so the single braces in embedded JSON schema examples
    /// are never touched. Unknown placeholders are left verbatim; unused vars are
    /// ignored.
    pub(crate) fn render(&self, name: PromptName, vars: &[(&str, &str)]) -> String {
        let mut body = self.load(name).into_owned();
        for (key, value) in vars {
            body = body.replace(&format!("{{{{{key}}}}}"), value);
        }
        body
    }
}

/// A fresh per-request fence nonce (12 hex chars). Untrusted source text is authored
/// at ingest — before this exists — so it can never pre-forge the marker.
pub(crate) fn fence_nonce() -> String {
    uuid::Uuid::now_v7().simple().to_string()[..12].to_string()
}

/// Wraps one excerpt's `inner` text in the `<<SRC:nonce>> … <<END:nonce>>` fence.
/// The single source of truth for the marker format, so the prompt builders cannot
/// drift apart.
pub(crate) fn fence_excerpt(nonce: &str, inner: &str) -> String {
    format!("<<SRC:{nonce}>>\n{inner}\n<<END:{nonce}>>\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_is_12_hex_chars() {
        let n = fence_nonce();
        assert_eq!(n.len(), 12);
        assert!(n.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[test]
    fn excerpt_wraps_inner_between_markers() {
        assert_eq!(
            fence_excerpt("abc", "body"),
            "<<SRC:abc>>\nbody\n<<END:abc>>\n"
        );
    }

    #[test]
    fn embedded_default_is_nonempty_and_has_placeholders() {
        let store = PromptStore::embedded();
        let body = store.load(PromptName::DialogueSystem);
        assert!(body.contains("{{turns}}"), "template keeps the turns slot");
        assert!(
            body.contains("{{emotions}}"),
            "template keeps the emotions slot"
        );
    }

    #[test]
    fn render_substitutes_double_brace_placeholders() {
        let store = PromptStore::embedded();
        let out = store.render(
            PromptName::DialogueSystem,
            &[("turns", "25"), ("emotions", "neutral, laugh")],
        );
        assert!(out.contains("about 25 turns"));
        assert!(out.contains("one of: neutral, laugh"));
        assert!(!out.contains("{{turns}}"));
        assert!(!out.contains("{{emotions}}"));
    }

    #[test]
    fn render_leaves_single_json_braces_untouched() {
        // A template containing a JSON example must survive rendering: only `{{k}}`
        // is a placeholder, never a bare `{...}`.
        let dir = tempfile::tempdir().unwrap();
        let prompts = dir.path().join("prompts").join("dialogue");
        std::fs::create_dir_all(&prompts).unwrap();
        std::fs::write(
            prompts.join("script.system.md"),
            "shape {\"speaker\":\"host\"} and {{turns}} turns",
        )
        .unwrap();
        let store = PromptStore::for_data_dir(dir.path());
        let out = store.render(PromptName::DialogueSystem, &[("turns", "8")]);
        assert_eq!(out, "shape {\"speaker\":\"host\"} and 8 turns");
    }

    #[test]
    fn override_file_wins_over_embedded_default() {
        let dir = tempfile::tempdir().unwrap();
        let prompts = dir.path().join("prompts").join("dialogue");
        std::fs::create_dir_all(&prompts).unwrap();
        std::fs::write(prompts.join("script.system.md"), "custom body {{turns}}").unwrap();
        let store = PromptStore::for_data_dir(dir.path());
        assert_eq!(
            store.load(PromptName::DialogueSystem),
            "custom body {{turns}}"
        );
    }

    #[test]
    fn missing_override_dir_falls_back_to_embedded() {
        let dir = tempfile::tempdir().unwrap();
        // No `prompts/` subdir created → override_dir resolves to None.
        let store = PromptStore::for_data_dir(dir.path());
        assert_eq!(
            store.load(PromptName::DialogueSystem),
            PromptName::DialogueSystem.embedded_default()
        );
    }

    #[test]
    fn missing_override_file_falls_back_even_when_dir_exists() {
        let dir = tempfile::tempdir().unwrap();
        // The prompts dir exists but this template's file does not.
        std::fs::create_dir_all(dir.path().join("prompts")).unwrap();
        let store = PromptStore::for_data_dir(dir.path());
        assert_eq!(
            store.load(PromptName::DialogueSystem),
            PromptName::DialogueSystem.embedded_default()
        );
    }
}
