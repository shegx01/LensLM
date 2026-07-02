import { mockIPC, clearMocks } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { validateModelInteractive } from './enrichment-validation.js';

beforeEach(() => {
  (globalThis as { isTauri?: boolean }).isTauri = true;
});

afterEach(() => {
  clearMocks();
  delete (globalThis as { isTauri?: boolean }).isTauri;
});

describe('validateModelInteractive — IPC wrapper', () => {
  it('invokes the validate_model_interactive command', async () => {
    let invokedCmd: string | null = null;
    mockIPC((cmd) => {
      invokedCmd = cmd;
      if (cmd === 'validate_model_interactive') return { status: 'valid' };
    });

    await validateModelInteractive('ollama', 'llama3.2:3b', 'http://localhost:11434', '');

    expect(invokedCmd).toBe('validate_model_interactive');
  });

  it('maps a valid IPC response to { status: "valid" }', async () => {
    mockIPC((cmd, args) => {
      if (cmd === 'validate_model_interactive') {
        expect(args).toMatchObject({
          provider: 'ollama',
          model: 'llama3.2:3b',
          base_url: 'http://localhost:11434',
          api_key: ''
        });
        return { status: 'valid' };
      }
    });

    const result = await validateModelInteractive(
      'ollama',
      'llama3.2:3b',
      'http://localhost:11434',
      ''
    );

    expect(result.status).toBe('valid');
    expect(result.reason).toBeUndefined();
  });

  it('maps an invalid IPC response to { status: "invalid", reason }', async () => {
    mockIPC((cmd) => {
      if (cmd === 'validate_model_interactive') {
        return {
          status: 'invalid',
          reason: "LLM runtime detected, but enrichment model 'missing:1b' is not installed."
        };
      }
    });

    const result = await validateModelInteractive(
      'ollama',
      'missing:1b',
      'http://localhost:11434',
      ''
    );

    expect(result.status).toBe('invalid');
    expect(result.reason).toMatch(/not installed/i);
  });

  it('passes cloud provider params through to IPC verbatim', async () => {
    const captured: Record<string, unknown> = {};
    mockIPC((cmd, args) => {
      if (cmd === 'validate_model_interactive') {
        Object.assign(captured, args);
        return { status: 'valid' };
      }
    });

    await validateModelInteractive('openai', 'gpt-4o', '', 'sk-test-key');

    expect(captured).toMatchObject({
      provider: 'openai',
      model: 'gpt-4o',
      base_url: '',
      api_key: 'sk-test-key'
    });
  });

  it('maps invalid cloud response with reason', async () => {
    mockIPC((cmd) => {
      if (cmd === 'validate_model_interactive') {
        return {
          status: 'invalid',
          reason: 'Cloud enrichment model probe failed: 401 Unauthorized'
        };
      }
    });

    const result = await validateModelInteractive('openai', 'gpt-4o', '', 'bad-key');

    expect(result.status).toBe('invalid');
    expect(result.reason).toMatch(/401/);
  });

  it('returns valid when reason is undefined in IPC response', async () => {
    mockIPC((cmd) => {
      if (cmd === 'validate_model_interactive') {
        // No reason field — valid path
        return { status: 'valid' };
      }
    });

    const result = await validateModelInteractive(
      'anthropic',
      'claude-3-5-sonnet-latest',
      '',
      'sk-ant-key'
    );
    expect(result.status).toBe('valid');
    expect(result.reason).toBeUndefined();
  });

  it('returns { status: "valid" } when not in a Tauri context', async () => {
    // Remove the isTauri flag set in beforeEach
    delete (globalThis as { isTauri?: boolean }).isTauri;
    const result = await validateModelInteractive('ollama', 'any-model', '', '');
    expect(result.status).toBe('valid');
    expect(result.reason).toBeUndefined();
  });
});
