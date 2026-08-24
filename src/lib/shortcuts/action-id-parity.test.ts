// AC-5 lexical tripwire: the TS union and the Rust `ActionId` enum must serialize
// the same 11 dotted ids. The behavioural half (serde_json::to_value over every
// variant) lives in lens-core/tests/keymap_config.rs — this half only reads text.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ACTION_IDS } from './registry';

const CONFIG_RS = 'lens-core/src/config.rs';
const RENAME_ATTR = '#[serde(rename = "';

/**
 * Removes Rust comments so a commented-out variant cannot leave its `rename`
 * literal behind and yield a false green. String literals (plain and raw) are
 * tracked so a `//` inside one survives; char literals are only skipped when
 * unambiguous, because `'de` lifetimes are indistinguishable without a parser.
 */
function stripComments(source: string): string {
  let out = '';
  let i = 0;

  while (i < source.length) {
    const c = source[i];
    const next = source[i + 1];

    if (c === '/' && next === '/') {
      while (i < source.length && source[i] !== '\n') i++;
      continue;
    }

    if (c === '/' && next === '*') {
      let depth = 1;
      i += 2;
      while (i < source.length && depth > 0) {
        if (source[i] === '/' && source[i + 1] === '*') {
          depth++;
          i += 2;
        } else if (source[i] === '*' && source[i + 1] === '/') {
          depth--;
          i += 2;
        } else {
          i++;
        }
      }
      continue;
    }

    if (c === 'r' && (next === '"' || next === '#')) {
      let hashes = 0;
      let j = i + 1;
      while (source[j] === '#') {
        hashes++;
        j++;
      }
      if (source[j] === '"') {
        const terminator = `"${'#'.repeat(hashes)}`;
        const end = source.indexOf(terminator, j + 1);
        const stop = end === -1 ? source.length : end + terminator.length;
        out += source.slice(i, stop);
        i = stop;
        continue;
      }
    }

    if (c === '"') {
      out += c;
      i++;
      while (i < source.length) {
        out += source[i];
        if (source[i] === '\\') {
          out += source[i + 1] ?? '';
          i += 2;
          continue;
        }
        const done = source[i] === '"';
        i++;
        if (done) break;
      }
      continue;
    }

    if (c === "'") {
      const escaped = next === '\\';
      const width = escaped ? 4 : 3;
      if (source[i + width - 1] === "'") {
        out += source.slice(i, i + width);
        i += width;
        continue;
      }
    }

    out += c;
    i++;
  }

  return out;
}

/** Brace-scoped body of `enum ActionId { … }`, or `null` if the enum is absent. */
function enumBody(source: string): string | null {
  const start = source.search(/\benum\s+ActionId\b/);
  if (start === -1) return null;

  const open = source.indexOf('{', start);
  if (open === -1) return null;

  let depth = 0;
  for (let i = open; i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}') {
      depth--;
      if (depth === 0) return source.slice(open + 1, i);
    }
  }
  return null;
}

function renameLiterals(source: string): string[] {
  return [...source.matchAll(/#\[serde\(rename\s*=\s*"([^"]*)"\s*\)\]/g)].map((m) => m[1]);
}

describe('ActionId parity with lens-core/src/config.rs', () => {
  const stripped = stripComments(readFileSync(CONFIG_RS, 'utf8'));

  it('has a matching Rust variant for every TS action id', () => {
    const body = enumBody(stripped);
    expect(body, `enum ActionId { … } not found in ${CONFIG_RS}`).not.toBeNull();

    const ids = renameLiterals(body as string);
    expect(ids).toHaveLength(ACTION_IDS.length);
    expect([...ids].sort()).toEqual([...ACTION_IDS].sort());
  });

  it('declares no serde rename outside the ActionId enum', () => {
    const total = stripped.split(RENAME_ATTR).length - 1;
    expect(total).toBe(ACTION_IDS.length);
  });
});
