<!-- Test-only stand-in for AudioPlayer.svelte, used by StudioPanel.svelte.test.ts so
     that suite can assert on the `src` StudioPanel derives without exercising
     AudioPlayer's own playback internals (covered separately). -->
<script lang="ts">
  let { src }: { src: string } = $props();
</script>

<div data-testid="audio-player-stub">{src}</div>
