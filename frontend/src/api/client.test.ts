import { describe, expect, it, vi } from 'vitest';
import { createSimulatedClient, fromHttpError, normalizeError } from './client';
import { fixtures } from '../mocks/fixtures';

describe('simulation-only API boundary', () => {
  it('returns fresh deterministic fixtures without retaining request text', async () => {
    vi.useFakeTimers();
    const client = createSimulatedClient('normal');
    const first = client.infer({ prompt: 'private text', category: 'qa' });
    await vi.advanceTimersByTimeAsync(450);
    const response = await first;
    expect(response).toEqual(fixtures.normal);
    response.text = 'mutation';
    const second = client.infer({ prompt: 'different text', category: 'coding' });
    await vi.advanceTimersByTimeAsync(450);
    expect(await second).toEqual(fixtures.normal);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('accepts an AbortSignal without network or retry', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(createSimulatedClient('normal').infer({ prompt: 'test', category: 'qa' }, { signal: controller.signal })).rejects.toMatchObject({ kind: 'cancelled' });
  });

  it.each([
    [401, 'authentication'], [422, 'invalid_request'], [429, 'rate_limited'],
    [503, 'unavailable'], [504, 'deadline'], [502, 'upstream'], [500, 'upstream'],
  ])('handles HTTP %s even without JSON', (status, kind) => {
    expect(fromHttpError(status as number)).toMatchObject({ status, kind });
  });

  it('distinguishes validation failure and never echoes raw server messages', () => {
    const error = fromHttpError(502, { error: { code: 'response_validation_failed', message: 'private internal details' }, request_id: 'safe-id' });
    expect(error.kind).toBe('validation');
    expect(error.requestId).toBe('safe-id');
    expect(error.message).not.toContain('private internal details');
    expect(fromHttpError(429, undefined, 60).retryAfterSeconds).toBe(60);
    expect(normalizeError(new Error('private network details')).kind).toBe('network');
    expect(normalizeError(new Error('private network details')).message).not.toContain('private network details');
  });
});
