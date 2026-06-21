# Proxima Nova → woff2 conversion (one-shot)

The bundled web fonts in `static/fonts/*.woff2` were generated **once** from the
user-supplied commercial Proxima Nova OTF/TTF sources. This is **not** a CI step —
the source files are licensed assets kept outside the repo, and the produced
`.woff2` artifacts are committed directly.

## What ships

Only four real weights are bundled — no faux/synthesized intermediates:

| Weight    | CSS `font-weight` | Source file                 | Output                        |
| --------- | ----------------- | --------------------------- | ----------------------------- |
| Light     | 300               | `proximanova_light.otf`     | `proximanova-light.woff2`     |
| Regular   | 400               | `proximanova_regular.ttf`   | `proximanova-regular.woff2`   |
| Bold      | 700               | `proximanova_bold.otf`      | `proximanova-bold.woff2`      |
| Extrabold | 800               | `proximanova_extrabold.otf` | `proximanova-extrabold.woff2` |

Black (900) and all italics are deliberately deferred. There is **no Semibold
(600)** in the source set — emphasis uses Regular 400 where a 600 would go, with
real Bold 700 for headings and Extrabold 800 for display. Do **not** add
`font-weight: 500`/`600` rules; the browser would synthesize (faux-bold) them.

## Reproducing the conversion

Requires `woff2_compress` (from Google's `woff2` tools; `brew install woff2`).
`woff2_compress` writes the `.woff2` next to its input, so work in a temp dir:

```sh
WORK=$(mktemp -d)
cp /path/to/proxima_nova/proximanova_light.otf     "$WORK/proximanova-light.otf"
cp /path/to/proxima_nova/proximanova_regular.ttf   "$WORK/proximanova-regular.ttf"
cp /path/to/proxima_nova/proximanova_bold.otf      "$WORK/proximanova-bold.otf"
cp /path/to/proxima_nova/proximanova_extrabold.otf "$WORK/proximanova-extrabold.otf"

for f in "$WORK"/*.otf "$WORK"/*.ttf; do woff2_compress "$f"; done

cp "$WORK"/*.woff2 static/fonts/
rm -rf "$WORK"
```

The `@font-face` declarations and the `--font-sans` token live in `src/app.css`.
Fonts are bundled-only: `src: url('/fonts/...woff2')` resolves to the local
`static/` dir — there is **no** CDN / Google Fonts / network request.
