//! AC-3.4: `lens-core` is the headless engine — it must never gain a `tauri`
//! dependency, directly or transitively. A `Cargo.lock` grep would match `tauri`
//! regardless of who depends on it, so this walks the resolved graph.

#[test]
fn lens_core_does_not_depend_on_tauri() {
    let out = std::process::Command::new(env!("CARGO"))
        .args(["tree", "-p", "lens-core", "-e", "normal"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree");
    assert!(
        out.status.success(),
        "cargo tree failed, so this check would pass vacuously: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let tree = String::from_utf8_lossy(&out.stdout);
    assert!(
        tree.contains("lens-core"),
        "cargo tree produced no lens-core node — parse failure, not a clean tree"
    );
    let offenders: Vec<&str> = tree
        .lines()
        .filter(|l| l.contains("tauri"))
        .map(str::trim)
        .collect();
    assert!(
        offenders.is_empty(),
        "lens-core must stay headless; tauri reached it via: {offenders:?}"
    );
}
