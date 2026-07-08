-- M13 #154: relation predicate vocabulary + aliases. `entity_edges.relation` is
-- TEXT validated at write time by the Rust `Relation` enum (no SQL FK/CHECK),
-- mirroring the `entity_nodes.kind`/`EntityKind` pattern. Query-time alias
-- resolution only — never an in-place UPDATE of `relation`.

CREATE TABLE IF NOT EXISTS relation_types (
    id         TEXT PRIMARY KEY NOT NULL,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relation_type_aliases (
    id         TEXT PRIMARY KEY NOT NULL,
    alias      TEXT NOT NULL UNIQUE,
    canonical  TEXT NOT NULL REFERENCES relation_types (name),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rtype_aliases_canonical
    ON relation_type_aliases (canonical);

-- Seed: co_occurs (required so existing co-occurrence edges stay valid) + 25
-- semantic predicates. `rt-`/`rta-` id prefixes are a deliberate deviation from
-- UUIDv7 for stable reference/seed data (these are not user-created rows).
INSERT OR IGNORE INTO relation_types (id, name, created_at) VALUES
    ('rt-co_occurs',        'co_occurs',        '2026-01-01T00:00:00Z'),
    ('rt-founded',          'founded',          '2026-01-01T00:00:00Z'),
    ('rt-created',          'created',          '2026-01-01T00:00:00Z'),
    ('rt-part_of',          'part_of',          '2026-01-01T00:00:00Z'),
    ('rt-member_of',        'member_of',        '2026-01-01T00:00:00Z'),
    ('rt-located_in',       'located_in',       '2026-01-01T00:00:00Z'),
    ('rt-caused_by',        'caused_by',        '2026-01-01T00:00:00Z'),
    ('rt-led_to',           'led_to',           '2026-01-01T00:00:00Z'),
    ('rt-preceded',         'preceded',         '2026-01-01T00:00:00Z'),
    ('rt-succeeded',        'succeeded',        '2026-01-01T00:00:00Z'),
    ('rt-influenced',       'influenced',       '2026-01-01T00:00:00Z'),
    ('rt-opposed',          'opposed',          '2026-01-01T00:00:00Z'),
    ('rt-collaborated_with','collaborated_with','2026-01-01T00:00:00Z'),
    ('rt-employed_by',      'employed_by',      '2026-01-01T00:00:00Z'),
    ('rt-authored',         'authored',         '2026-01-01T00:00:00Z'),
    ('rt-published_in',     'published_in',     '2026-01-01T00:00:00Z'),
    ('rt-derived_from',     'derived_from',     '2026-01-01T00:00:00Z'),
    ('rt-related_to',       'related_to',       '2026-01-01T00:00:00Z'),
    ('rt-similar_to',       'similar_to',       '2026-01-01T00:00:00Z'),
    ('rt-contrasts_with',   'contrasts_with',   '2026-01-01T00:00:00Z'),
    ('rt-depends_on',       'depends_on',       '2026-01-01T00:00:00Z'),
    ('rt-contains',         'contains',         '2026-01-01T00:00:00Z'),
    ('rt-controls',         'controls',         '2026-01-01T00:00:00Z'),
    ('rt-funded_by',        'funded_by',        '2026-01-01T00:00:00Z'),
    ('rt-acquired',         'acquired',         '2026-01-01T00:00:00Z');

INSERT OR IGNORE INTO relation_type_aliases (id, alias, canonical, created_at) VALUES
    ('rta-works_at',      'works_at',      'employed_by', '2026-01-01T00:00:00Z'),
    ('rta-wrote',         'wrote',         'authored',    '2026-01-01T00:00:00Z'),
    ('rta-based_in',      'based_in',      'located_in',  '2026-01-01T00:00:00Z'),
    ('rta-subsidiary_of', 'subsidiary_of', 'part_of',     '2026-01-01T00:00:00Z'),
    ('rta-resulted_in',   'resulted_in',   'led_to',      '2026-01-01T00:00:00Z');
