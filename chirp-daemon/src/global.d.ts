/**
 * Missing type from @anthropic-ai/claude-code sdk.d.ts.
 * The SDK references `Dict<T>` without importing or defining it.
 * This is equivalent to Record<string, T>.
 */
type Dict<T> = Record<string, T>;
