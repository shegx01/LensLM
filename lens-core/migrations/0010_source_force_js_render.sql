-- #78: per-source "SPA / render this page" opt-in. When 1, ingest ALWAYS
-- routes the URL source through the offscreen-webview JS-render path instead of
-- relying on static-extraction auto-detection. Persisted so re-ingest and
-- crash-recovery honor it. SQLite integer boolean, mirroring `selected INTEGER`.
ALTER TABLE sources ADD COLUMN force_js_render INTEGER NOT NULL DEFAULT 0;
