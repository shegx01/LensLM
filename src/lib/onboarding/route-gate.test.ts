import { describe, expect, it } from 'vitest';
import { decideOnboardingRoute } from './route-gate.js';

describe('decideOnboardingRoute', () => {
  it('redirects to /onboarding when incomplete and off the onboarding route', () => {
    expect(decideOnboardingRoute(false, '/')).toBe('/onboarding');
    expect(decideOnboardingRoute(false, '/showcase')).toBe('/onboarding');
  });

  it('stays put when incomplete and already on the onboarding route', () => {
    expect(decideOnboardingRoute(false, '/onboarding')).toBeNull();
  });

  it('redirects to / when complete but still on an onboarding route', () => {
    expect(decideOnboardingRoute(true, '/onboarding')).toBe('/');
  });

  it('stays put when complete and off the onboarding route', () => {
    expect(decideOnboardingRoute(true, '/')).toBeNull();
    expect(decideOnboardingRoute(true, '/showcase')).toBeNull();
  });
});
