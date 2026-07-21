<!-- Test-only stand-in for AudioPlayer.svelte, used by StudioPanel.svelte.test.ts so
     that suite can assert on the `path` StudioPanel passes without exercising
     AudioPlayer's own playback internals (covered separately). -->
<script lang="ts">
  let { path }: { path: string } = $props();
</script>

<div data-testid="audio-player-stub">{path}</div>
