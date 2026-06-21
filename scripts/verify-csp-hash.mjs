#!/usr/bin/env node
// verify-csp-hash.mjs — No external dependencies.
//
// Reads the inline pre-paint <script> from src/app.html, computes its
// sha256-<base64> hash, then checks that the hash appears in the CSP string
// inside src-tauri/tauri.conf.json. Exits nonzero with a clear message if not.
//
// Run via: node scripts/verify-csp-hash.mjs
// Or:      bun run verify:csp

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

// --- 1. Extract the inline script from app.html ---
const htmlPath = join(root, 'src', 'app.html');
let html;
try {
  html = readFileSync(htmlPath, 'utf8');
} catch (e) {
  console.error(`[verify-csp-hash] ERROR: Could not read ${htmlPath}\n  ${e.message}`);
  process.exit(1);
}

const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) {
  console.error('[verify-csp-hash] ERROR: No inline <script>...</script> found in src/app.html');
  process.exit(1);
}
const scriptContent = scriptMatch[1];

// --- 2. Compute sha256-<base64> ---
const hash = createHash('sha256').update(scriptContent, 'utf8').digest('base64');
const directive = `sha256-${hash}`;
console.log(`[verify-csp-hash] Computed hash: ${directive}`);

// --- 3. Read CSP from tauri.conf.json ---
const confPath = join(root, 'src-tauri', 'tauri.conf.json');
let conf;
try {
  conf = JSON.parse(readFileSync(confPath, 'utf8'));
} catch (e) {
  console.error(`[verify-csp-hash] ERROR: Could not read/parse ${confPath}\n  ${e.message}`);
  process.exit(1);
}

const csp = conf?.app?.security?.csp;
if (typeof csp !== 'string') {
  console.error(
    '[verify-csp-hash] ERROR: app.security.csp not found (or not a string) in tauri.conf.json'
  );
  process.exit(1);
}

// --- 4. Verify ---
if (!csp.includes(directive)) {
  console.error(
    `[verify-csp-hash] MISMATCH: Hash '${directive}' is NOT present in the CSP.` +
      `\n  CSP: ${csp}` +
      `\n\n  The inline script in src/app.html was modified without updating the CSP hash.` +
      `\n  Update src-tauri/tauri.conf.json → app.security.csp → script-src with the hash above.`
  );
  process.exit(1);
}

console.log('[verify-csp-hash] OK — hash is present in tauri.conf.json CSP.');
