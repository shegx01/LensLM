-- Persist the generated dialogue script alongside the overview (#29 redesign) so the
-- Studio "Transcript" tab survives a reload. `script` is the JSON-serialized
-- DialogueScript (a `{"turns":[...]}` object); NULL on legacy rows and `failed` rows.
ALTER TABLE audio_overviews ADD COLUMN script TEXT;
