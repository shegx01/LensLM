import { describe, expect, it } from 'vitest';
import {
  EMBEDDING_MODELS,
  ollamaMatches,
  type EmbeddingModelId,
  type EmbeddingBackend
} from './models.js';

describe('EMBEDDING_MODELS catalog', () => {
  it('contains exactly 8 models (4 fastembed + 4 ollama)', () => {
    expect(EMBEDDING_MODELS).toHaveLength(8);
  });

  it('includes all 4 new ollama model ids', () => {
    const ids = EMBEDDING_MODELS.map((m) => m.id);
    expect(ids).toContain('embeddinggemma');
    expect(ids).toContain('qwen3-embedding:4b');
    expect(ids).toContain('nomic-embed-text-v2-moe');
    expect(ids).toContain('snowflake-arctic-embed2');
  });

  it('includes all 4 original fastembed model ids', () => {
    const ids = EMBEDDING_MODELS.map((m) => m.id);
    expect(ids).toContain('nomic-embed-text-v1.5');
    expect(ids).toContain('mxbai-embed-large');
    expect(ids).toContain('all-minilm');
    expect(ids).toContain('bge-m3');
  });

  describe('backends field', () => {
    it('existing fastembed models have backends: ["fastembed"]', () => {
      const fastembedIds: EmbeddingModelId[] = [
        'nomic-embed-text-v1.5',
        'mxbai-embed-large',
        'all-minilm',
        'bge-m3'
      ];
      for (const id of fastembedIds) {
        const model = EMBEDDING_MODELS.find((m) => m.id === id);
        expect(model, `model ${id} should exist`).toBeDefined();
        expect(model!.backends, `${id} backends`).toEqual(['fastembed']);
      }
    });

    it('new ollama models have backends: ["ollama"]', () => {
      const ollamaIds: EmbeddingModelId[] = [
        'embeddinggemma',
        'qwen3-embedding:4b',
        'nomic-embed-text-v2-moe',
        'snowflake-arctic-embed2'
      ];
      for (const id of ollamaIds) {
        const model = EMBEDDING_MODELS.find((m) => m.id === id);
        expect(model, `model ${id} should exist`).toBeDefined();
        expect(model!.backends, `${id} backends`).toEqual(['ollama']);
      }
    });
  });

  describe('dims', () => {
    it('embeddinggemma has dim 768', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'embeddinggemma');
      expect(m!.dims).toBe(768);
    });

    it('qwen3-embedding:4b has dim 2560', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'qwen3-embedding:4b');
      expect(m!.dims).toBe(2560);
    });

    it('nomic-embed-text-v2-moe has dim 768', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'nomic-embed-text-v2-moe');
      expect(m!.dims).toBe(768);
    });

    it('snowflake-arctic-embed2 has dim 1024', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'snowflake-arctic-embed2');
      expect(m!.dims).toBe(1024);
    });
  });

  describe('ollamaName equals id for new models', () => {
    it('qwen3-embedding:4b ollamaName is exactly "qwen3-embedding:4b"', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'qwen3-embedding:4b');
      expect(m!.ollamaName).toBe('qwen3-embedding:4b');
    });

    it('embeddinggemma ollamaName is exactly "embeddinggemma"', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'embeddinggemma');
      expect(m!.ollamaName).toBe('embeddinggemma');
    });

    it('nomic-embed-text-v2-moe ollamaName is exactly "nomic-embed-text-v2-moe"', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'nomic-embed-text-v2-moe');
      expect(m!.ollamaName).toBe('nomic-embed-text-v2-moe');
    });

    it('snowflake-arctic-embed2 ollamaName is exactly "snowflake-arctic-embed2"', () => {
      const m = EMBEDDING_MODELS.find((m) => m.id === 'snowflake-arctic-embed2');
      expect(m!.ollamaName).toBe('snowflake-arctic-embed2');
    });

    // Iterating the catalog auto-covers future ollama models (issue #80).
    it('every ollama-backend model has ollamaName === id', () => {
      for (const m of EMBEDDING_MODELS.filter((m) => m.backends.includes('ollama'))) {
        expect(m.ollamaName, `${m.id} ollamaName must equal id`).toBe(m.id);
      }
    });
  });
});

describe('ollamaMatches() — exact-tag detection (D3)', () => {
  const qwen3 = EMBEDDING_MODELS.find((m) => m.id === 'qwen3-embedding:4b')!;
  const nomic = EMBEDDING_MODELS.find((m) => m.id === 'nomic-embed-text-v1.5')!;
  const nomicV2 = EMBEDDING_MODELS.find((m) => m.id === 'nomic-embed-text-v2-moe')!;

  describe('qwen3-embedding:4b (id contains colon → exact match only)', () => {
    it('matches exact detected="qwen3-embedding:4b"', () => {
      expect(ollamaMatches('qwen3-embedding:4b', qwen3)).toBe(true);
    });

    it('does NOT match "qwen3-embedding:0.6b"', () => {
      expect(ollamaMatches('qwen3-embedding:0.6b', qwen3)).toBe(false);
    });

    it('does NOT match "qwen3-embedding:8b"', () => {
      expect(ollamaMatches('qwen3-embedding:8b', qwen3)).toBe(false);
    });

    it('does NOT match bare "qwen3-embedding"', () => {
      expect(ollamaMatches('qwen3-embedding', qwen3)).toBe(false);
    });

    it('does NOT match "qwen3-embedding:latest"', () => {
      expect(ollamaMatches('qwen3-embedding:latest', qwen3)).toBe(false);
    });
  });

  describe('nomic-embed-text-v1.5 (no colon in ollamaName → prefix match)', () => {
    it('matches detected="nomic-embed-text"', () => {
      expect(ollamaMatches('nomic-embed-text', nomic)).toBe(true);
    });

    it('matches detected="nomic-embed-text:latest"', () => {
      expect(ollamaMatches('nomic-embed-text:latest', nomic)).toBe(true);
    });

    it('matches detected="nomic-embed-text:v1.5"', () => {
      expect(ollamaMatches('nomic-embed-text:v1.5', nomic)).toBe(true);
    });

    it('does NOT match "qwen3-embedding:4b"', () => {
      expect(ollamaMatches('qwen3-embedding:4b', nomic)).toBe(false);
    });
  });

  describe('nomic-embed-text-v2-moe (no colon in ollamaName → prefix match)', () => {
    it('matches detected="nomic-embed-text-v2-moe"', () => {
      expect(ollamaMatches('nomic-embed-text-v2-moe', nomicV2)).toBe(true);
    });

    it('matches detected="nomic-embed-text-v2-moe:latest"', () => {
      expect(ollamaMatches('nomic-embed-text-v2-moe:latest', nomicV2)).toBe(true);
    });

    it('does NOT confuse with qwen3', () => {
      expect(ollamaMatches('qwen3-embedding:4b', nomicV2)).toBe(false);
    });
  });
});
