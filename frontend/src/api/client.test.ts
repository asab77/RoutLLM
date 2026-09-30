import { describe, expect, it, vi } from 'vitest';
import { createHttpClient, createSimulatedClient } from './client';
import { fixtures } from '../mocks/fixtures';

function reply(status: number, body?: unknown, headers: Record<string, string> = {}): Response {
  const text = body === undefined ? '' : typeof body === 'string' ? body : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(headers),
    text: vi.fn().mockResolvedValue(text),
    json: vi.fn().mockImplementation(async () => JSON.parse(text)),
  } as unknown as Response;
}

describe('production HTTP client', () => {
  it('posts AUTO to the relative endpoint without category, validation, or threshold', async () => {
    const fetcher = vi.fn().mockResolvedValue(reply(200, fixtures.normal));
    await createHttpClient(fetcher).infer({
      prompt: 'hello', routing_mode: 'auto', max_output_tokens: 256, temperature: 1,
    });
    expect(fetcher).toHaveBeenCalledWith('/v1/chat', expect.objectContaining({
      method: 'POST', headers: { 'Content-Type': 'application/json' },
    }));
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
      prompt: 'hello', routing_mode: 'auto', max_output_tokens: 256, temperature: 1,
    });
  });

  it('serializes MANUAL with its explicit canonical category only', async () => {
    const fetcher = vi.fn().mockResolvedValue(reply(200, fixtures.fallback));
    await createHttpClient(fetcher).infer({ prompt: 'code', routing_mode: 'manual', category: 'coding' });
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
      prompt: 'code', routing_mode: 'manual', category: 'coding',
    });
  });

  it('parses direct and adaptive responses without losing tiny decimal costs or route facts', async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(reply(200, fixtures.normal))
      .mockResolvedValueOnce(reply(200, fixtures.escalation));
    const client = createHttpClient(fetcher);
    const direct = await client.infer({ prompt: 'hello', routing_mode: 'auto' });
    const adaptive = await client.infer({ prompt: 'reason', routing_mode: 'manual', category: 'reasoning' });
    expect(direct).toMatchObject({ execution_mode: 'direct', estimated_cost_usd: '0.00000135' });
    expect(adaptive).toMatchObject({
      execution_mode: 'adaptive', model_id: 'candidate-gpt-6-luna',
      routing: { selected_model_id: 'candidate-nemotron-3.5-lightning', fallback_used: false },
      execution: { escalated: true },
    });
  });

  it('reads Activity from a relative cursor URL and preserves next_cursor', async () => {
    const page = {
      items: [{
        request_id: 'activity-1', created_at: '2026-09-29T20:00:00Z',
        execution_mode: 'direct', final_model_id: 'fake-small', outcome: 'returned',
        estimated_cost_usd: '0.00000135', cost_complete: true,
      }],
      next_cursor: 'opaque-next',
    };
    const fetcher = vi.fn().mockResolvedValue(reply(200, page));
    const result = await createHttpClient(fetcher).activity({ limit: 10, cursor: 'opaque-current' });
    expect(fetcher).toHaveBeenCalledWith(
      '/v1/activity?limit=10&cursor=opaque-current',
      expect.objectContaining({ signal: undefined }),
    );
    expect(result).toEqual(page);
  });

  it.each([
    [400, 'invalid_request'], [422, 'invalid_request'], [401, 'authentication'],
    [403, 'authentication'], [404, 'invalid_request'], [502, 'upstream'],
    [503, 'unavailable'], [504, 'deadline'],
  ] as const)('normalizes HTTP %s without surfacing arbitrary response text', async (status, kind) => {
    const fetcher = vi.fn().mockResolvedValue(reply(status, '<html>private provider exception</html>'));
    await expect(createHttpClient(fetcher).infer({ prompt: 'x', routing_mode: 'auto' }))
      .rejects.toMatchObject({ status, kind });
    try {
      await createHttpClient(fetcher).infer({ prompt: 'x', routing_mode: 'auto' });
    } catch (error) {
      expect((error as Error).message).not.toContain('private provider exception');
    }
  });

  it('uses allowlisted backend codes and retains a safe request ID', async () => {
    const fetcher = vi.fn().mockResolvedValue(reply(502, {
      error: { code: 'response_validation_failed', message: 'private validator details' },
      request_id: 'safe-request-1',
    }));
    await expect(createHttpClient(fetcher).infer({ prompt: 'x', routing_mode: 'auto' }))
      .rejects.toMatchObject({ kind: 'validation', requestId: 'safe-request-1' });
  });

  it('preserves a numeric Retry-After and ignores unsafe values', async () => {
    const first = vi.fn().mockResolvedValue(reply(429, undefined, { 'Retry-After': '17' }));
    await expect(createHttpClient(first).infer({ prompt: 'x', routing_mode: 'auto' }))
      .rejects.toMatchObject({ kind: 'rate_limited', retryAfterSeconds: 17 });
    const second = vi.fn().mockResolvedValue(reply(429, undefined, { 'Retry-After': 'tomorrow' }));
    await expect(createHttpClient(second).infer({ prompt: 'x', routing_mode: 'auto' }))
      .rejects.toMatchObject({ kind: 'rate_limited', retryAfterSeconds: undefined });
  });

  it.each([undefined, '', 'not-json'])('tolerates empty and malformed error bodies', async body => {
    const fetcher = vi.fn().mockResolvedValue(reply(401, body));
    await expect(createHttpClient(fetcher).infer({ prompt: 'x', routing_mode: 'auto' }))
      .rejects.toMatchObject({ kind: 'authentication' });
  });

  it('normalizes fetch failures without exposing their message', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('private network address'));
    const request = createHttpClient(fetcher).infer({ prompt: 'x', routing_mode: 'auto' });
    await expect(request).rejects.toMatchObject({ kind: 'network' });
    await expect(request).rejects.not.toThrow(/private network address/);
  });
});

describe('explicit simulation fixture boundary', () => {
  it('remains injectable and never uses fetch', async () => {
    vi.useFakeTimers();
    const pending = createSimulatedClient('normal').infer({ prompt: 'test', routing_mode: 'auto' });
    await vi.advanceTimersByTimeAsync(450);
    expect(await pending).toEqual(fixtures.normal);
    expect(fetch).not.toHaveBeenCalled();
  });
});
