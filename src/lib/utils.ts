import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Collapse whitespace and truncate to `max` chars with an ellipsis. */
export function truncateLabel(text: string, max = 60): string {
  const collapsed = text.replace(/\s+/g, ' ');
  return collapsed.length > max ? `${collapsed.slice(0, max)}…` : collapsed;
}

export type WithoutChild<T> = T extends { child?: unknown } ? Omit<T, 'child'> : T;
export type WithoutChildren<T> = T extends { children?: unknown } ? Omit<T, 'children'> : T;
export type WithoutChildrenOrChild<T> = WithoutChildren<WithoutChild<T>>;
export type WithElementRef<T, U extends HTMLElement = HTMLElement> = T & { ref?: U | null };
