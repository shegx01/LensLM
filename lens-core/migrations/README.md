# Migrations

These `.sql` files are embedded into the binary at build time via
`sqlx::migrate!("./migrations")` and applied idempotently on engine init.
sqlx records each applied file (name + checksum) in `_sqlx_migrations` and
wraps each file in a transaction (one file = one atomic unit).

## Rules

1. **Never edit an applied migration.** Changing the bytes of a file that has
   already run changes its checksum and `sqlx::migrate!` will refuse to start
   (checksum mismatch). To change the schema, add a **new** numbered file.
2. **Numbering:** `NNNN_description.sql`, monotonically increasing
   (`0001_init.sql`, `0002_...`).
3. **Idempotent DDL:** use `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT
EXISTS` as defense-in-depth.
4. **Own your tables:** each milestone adds the tables it introduces
   (e.g. `tts_voice` in M2, `audio_overview` in M7) as its own additive
   migration rather than pre-creating them here.
