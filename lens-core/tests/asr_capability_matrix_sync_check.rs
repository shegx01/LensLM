//! SYNC-CHECK (issue #136): the TS `ASR_CAPABILITY_MATRIX`
//! (`src/lib/asr/catalog.ts`) claims, per engine, whether `language` and
//! `translate` are honoured/ignored/rerouted. This greps the actual engine
//! sources so drift (e.g. cloud wiring up translate, or the Apple reroute
//! being dropped) fails the gate instead of a UI notice silently lying.

use std::fs;
use std::path::PathBuf;

fn read_source(rel_path: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel_path);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

#[test]
fn apple_native_translate_reroutes_to_local_whisper() {
    let src = read_source("src/lib.rs");
    assert!(
        src.contains("config.translate && backend == asr::AsrBackend::AppleNative"),
        "apple_native translate reroute condition not found in lib.rs — \
         update ASR_CAPABILITY_MATRIX.apple_native.translate in catalog.ts if this changed"
    );
    assert!(
        src.contains("backend = asr::AsrBackend::LocalWhisper;"),
        "apple_native reroute target (LocalWhisper) not found in lib.rs"
    );
}

#[test]
fn local_whisper_honours_both_language_and_translate() {
    let src = read_source("src/asr/whisper.rs");
    assert!(
        src.contains("params.set_language(language)"),
        "whisper.rs no longer reads language — update ASR_CAPABILITY_MATRIX.local_whisper.language"
    );
    assert!(
        src.contains("params.set_translate(translate)"),
        "whisper.rs no longer reads translate — update ASR_CAPABILITY_MATRIX.local_whisper.translate"
    );
}

#[test]
fn cloud_adapters_honour_language_but_ignore_translate() {
    for rel in [
        "src/asr/cloud/deepgram.rs",
        "src/asr/cloud/openai_compat.rs",
    ] {
        let src = read_source(rel);
        assert!(
            src.contains("config.language"),
            "{rel} no longer reads config.language — update ASR_CAPABILITY_MATRIX.cloud.language"
        );
        assert!(
            !src.contains("config.translate"),
            "{rel} now reads config.translate — update ASR_CAPABILITY_MATRIX.cloud.translate \
             (currently documented as ignored)"
        );
    }
}
