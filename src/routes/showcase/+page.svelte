<script lang="ts">
  import ThemeSwitcher from '$lib/components/ThemeSwitcher.svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import { Input } from '$lib/components/ui/input/index.js';
  import { Switch } from '$lib/components/ui/switch/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Separator } from '$lib/components/ui/separator/index.js';
  import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';
  import * as Card from '$lib/components/ui/card/index.js';
  import * as Dialog from '$lib/components/ui/dialog/index.js';
  import * as Tooltip from '$lib/components/ui/tooltip/index.js';

  // Semantic tokens for the swatch wall (component-facing names only).
  const swatches: { name: string; varName: string; fg?: string }[] = [
    { name: 'background', varName: '--color-background', fg: '--color-foreground' },
    { name: 'card', varName: '--color-card', fg: '--color-card-foreground' },
    { name: 'popover', varName: '--color-popover', fg: '--color-popover-foreground' },
    { name: 'primary', varName: '--color-primary', fg: '--color-primary-foreground' },
    { name: 'secondary', varName: '--color-secondary', fg: '--color-secondary-foreground' },
    { name: 'muted', varName: '--color-muted', fg: '--color-muted-foreground' },
    { name: 'accent', varName: '--color-accent', fg: '--color-accent-foreground' },
    { name: 'destructive', varName: '--color-destructive', fg: '--color-destructive-foreground' },
    { name: 'border', varName: '--color-border' },
    { name: 'input', varName: '--color-input' },
    { name: 'ring', varName: '--color-ring' }
  ];

  const radii: { name: string; varName: string }[] = [
    { name: 'sm', varName: 'var(--radius-sm)' },
    { name: 'md', varName: 'var(--radius-md)' },
    { name: 'lg', varName: 'var(--radius-lg)' },
    { name: 'xl', varName: 'var(--radius-xl)' }
  ];

  let switchOn = $state(true);
  let dialogOpen = $state(false);
</script>

<div class="bg-background text-foreground min-h-screen">
  <div class="mx-auto flex max-w-4xl flex-col gap-10 px-6 py-10">
    <!-- Header -->
    <header class="flex items-start justify-between gap-4">
      <div class="flex flex-col gap-1">
        <h1 class="text-3xl font-extrabold tracking-tight">Lens Design System</h1>
        <p class="text-muted-foreground text-sm">
          M1-0 foundation — 9 primitives, OKLCH tokens, Proxima Nova. Toggle the theme to verify
          both modes.
        </p>
      </div>
      <ThemeSwitcher />
    </header>

    <Separator />

    <!-- Typography / type scale -->
    <section class="flex flex-col gap-4">
      <h2 class="text-xl font-bold">Typography (Proxima Nova)</h2>
      <div class="flex flex-col gap-3">
        <p class="text-3xl font-extrabold">Display — Extrabold 800</p>
        <p class="text-2xl font-bold">Heading — Bold 700</p>
        <p class="text-base font-normal">
          Body — Regular 400. The quick brown fox jumps over the lazy dog.
        </p>
        <p class="text-sm font-light text-muted-foreground">
          Caption — Light 300. The quick brown fox jumps over the lazy dog.
        </p>
        <p class="text-base font-normal">
          Emphasis (no Semibold 600 available): inline <strong class="font-bold">bold</strong> stands
          in for emphasis; running emphasis stays Regular 400.
        </p>
      </div>
    </section>

    <Separator />

    <!-- Token swatch wall -->
    <section class="flex flex-col gap-4">
      <h2 class="text-xl font-bold">Semantic tokens</h2>
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
        {#each swatches as s (s.name)}
          <div class="border-border overflow-hidden rounded-lg border">
            <div
              class="flex h-16 items-center justify-center text-xs"
              style="background: var({s.varName}); color: {s.fg
                ? `var(${s.fg})`
                : 'var(--color-foreground)'}"
            >
              {s.fg ? 'Aa' : ''}
            </div>
            <div class="bg-card text-card-foreground px-2 py-1 text-xs">{s.name}</div>
          </div>
        {/each}
      </div>
    </section>

    <Separator />

    <!-- Radii -->
    <section class="flex flex-col gap-4">
      <h2 class="text-xl font-bold">Radii</h2>
      <div class="flex flex-wrap gap-4">
        {#each radii as r (r.name)}
          <div class="flex flex-col items-center gap-2">
            <div class="bg-primary size-16" style="border-radius: {r.varName}"></div>
            <span class="text-muted-foreground text-xs">{r.name}</span>
          </div>
        {/each}
      </div>
    </section>

    <Separator />

    <!-- Primitives -->
    <section class="flex flex-col gap-6">
      <h2 class="text-xl font-bold">Primitives</h2>

      <!-- Buttons -->
      <div class="flex flex-col gap-2">
        <h3 class="text-muted-foreground text-sm">Button</h3>
        <div class="flex flex-wrap items-center gap-2">
          <Button>Primary</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="destructive">Destructive</Button>
          <Button variant="link">Link</Button>
          <Button disabled>Disabled</Button>
        </div>
      </div>

      <!-- Input + Switch -->
      <div class="grid gap-6 sm:grid-cols-2">
        <div class="flex flex-col gap-2">
          <h3 class="text-muted-foreground text-sm">Input</h3>
          <Input placeholder="Search notebooks…" />
        </div>
        <div class="flex flex-col gap-2">
          <h3 class="text-muted-foreground text-sm">Switch</h3>
          <div class="flex items-center gap-2">
            <Switch bind:checked={switchOn} id="demo-switch" />
            <label for="demo-switch" class="text-sm">{switchOn ? 'On' : 'Off'}</label>
          </div>
        </div>
      </div>

      <!-- Badges -->
      <div class="flex flex-col gap-2">
        <h3 class="text-muted-foreground text-sm">Badge</h3>
        <div class="flex flex-wrap items-center gap-2">
          <Badge>Default</Badge>
          <Badge variant="secondary">Secondary</Badge>
          <Badge variant="outline">Outline</Badge>
          <Badge variant="destructive">Destructive</Badge>
        </div>
      </div>

      <!-- Card -->
      <div class="flex flex-col gap-2">
        <h3 class="text-muted-foreground text-sm">Card</h3>
        <Card.Root class="max-w-sm">
          <Card.Header>
            <Card.Title>Product Roadmap H2</Card.Title>
            <Card.Description>1 source · updated 2w ago</Card.Description>
          </Card.Header>
          <Card.Content>
            <p class="text-sm">Add sources, then ask anything — answers cite your documents.</p>
          </Card.Content>
          <Card.Footer>
            <Button size="sm">Open</Button>
          </Card.Footer>
        </Card.Root>
      </div>

      <!-- Dialog + Tooltip -->
      <div class="flex flex-wrap items-center gap-4">
        <div class="flex flex-col gap-2">
          <h3 class="text-muted-foreground text-sm">Dialog</h3>
          <Dialog.Root bind:open={dialogOpen}>
            <Dialog.Trigger>
              {#snippet child({ props })}
                <Button {...props}>Open dialog</Button>
              {/snippet}
            </Dialog.Trigger>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title>New notebook</Dialog.Title>
                <Dialog.Description>Give your notebook a name to get started.</Dialog.Description>
              </Dialog.Header>
              <Input placeholder="Notebook name" />
              <Dialog.Footer>
                <Button variant="ghost" onclick={() => (dialogOpen = false)}>Cancel</Button>
                <Button onclick={() => (dialogOpen = false)}>Create</Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Root>
        </div>

        <div class="flex flex-col gap-2">
          <h3 class="text-muted-foreground text-sm">Tooltip</h3>
          <Tooltip.Provider>
            <Tooltip.Root>
              <Tooltip.Trigger>
                {#snippet child({ props })}
                  <Button variant="outline" {...props}>Hover me</Button>
                {/snippet}
              </Tooltip.Trigger>
              <Tooltip.Content>Cites the documents you selected.</Tooltip.Content>
            </Tooltip.Root>
          </Tooltip.Provider>
        </div>
      </div>

      <!-- ScrollArea + Separator -->
      <div class="flex flex-col gap-2">
        <h3 class="text-muted-foreground text-sm">ScrollArea + Separator</h3>
        <ScrollArea class="border-border h-40 w-full max-w-sm rounded-lg border p-3">
          <div class="flex flex-col gap-2">
            {#each Array.from({ length: 12 }, (_, i) => i + 1) as n (n)}
              <div class="text-sm">Source {n} — citation excerpt</div>
              {#if n < 12}<Separator />{/if}
            {/each}
          </div>
        </ScrollArea>
      </div>
    </section>
  </div>
</div>
