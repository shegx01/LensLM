<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';

  async function dispatchCoreAction(): Promise<void> {
    try {
      await invoke<string>('invoke_core_action', { payload: '' });
    } catch (error) {
      console.error('LensLM application contract runtime exception:', error);
    }
  }
</script>

<main>
  <h1>Hello World</h1>
  <button onclick={dispatchCoreAction}>Invoke core action</button>
</main>
