// =============================================================================
// TCNet Dashboard — Tailwind CSS Utility Helper
// Merges Tailwind classes safely using clsx + tailwind-merge.
// =============================================================================

import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines multiple class values and intelligently merges Tailwind CSS classes,
 * resolving conflicts (e.g. `p-4` and `p-2` → keeps last).
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
