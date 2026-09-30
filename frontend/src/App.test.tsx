import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import { InferenceError, createSimulatedClient } from './api/client';
import type { ActivityItem, ChatResponse, InferenceClient } from './api/types';
import { fixtures } from './mocks/fixtures';

function mockClient(response: ChatResponse = fixtures.normal): InferenceClient {
  return { infer: vi.fn().mockResolvedValue(response), activity: vi.fn().mockResolvedValue({ items: [] }) };
}
function prompt(value = 'Explain adaptive routing.') {
  fireEvent.change(screen.getByLabelText('Message RoutLLM'), { target: { value } });
}
function send() { fireEvent.click(screen.getByRole('button', { name: 'Send message' })); }
function settings() { fireEvent.click(screen.getByRole('button', { name: 'Settings' })); }
function advanced() { settings(); fireEvent.click(screen.getByRole('switch', { name: 'Advanced controls' })); }
function expand() { fireEvent.click(screen.getByText('View route').closest('summary')!); }

describe('real chat composition', () => {
  it('shows the production connection state and does not request before submission', () => {
    const client = mockClient();
    render(<App client={client} />);
    expect(screen.getByText('LIVE API')).toBeVisible();
    expect(screen.getByText('Same-origin')).toBeVisible();
    expect(screen.getByRole('form', { name: 'Message composer' })).toBeVisible();
    expect(client.infer).not.toHaveBeenCalled();
    expect(screen.queryByText(/simulation/i)).not.toBeInTheDocument();
  });

  it('sends an honest AUTO request without category, validation, or threshold', async () => {
    const client = mockClient();
    render(<App client={client} />);
    prompt(); send();
    await screen.findByRole('article', { name: 'RoutLLM response' });
    expect(client.infer).toHaveBeenCalledWith({
      prompt: 'Explain adaptive routing.', routing_mode: 'auto',
      max_output_tokens: 256, temperature: 1,
    });
  });

  it('renders AUTO as direct inference without category detection or ML routing claims', async () => {
    render(<App client={mockClient(fixtures.normal)} />);
    prompt(); send();
    expect((await screen.findAllByText('Direct inference'))[0]).toBeVisible();
    expand();
    expect(screen.getByText('Configured default model')).toBeVisible();
    expect(screen.getByText('candidate-nemotron-3.5-lightning')).toBeVisible();
    expect(screen.getByText('$0.00000135')).toBeVisible();
    expect(screen.queryByText(/detected|ML ROUTER|threshold met/i)).not.toBeInTheDocument();
  });

  it('requires a manual category and sends the selected canonical value', async () => {
    const client = mockClient(fixtures.fallback);
    render(<App client={client} />);
    advanced(); fireEvent.click(screen.getByLabelText('Manual category'));
    prompt();
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Task category'), { target: { value: 'coding' } });
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled();
    send(); await screen.findByRole('article', { name: 'RoutLLM response' });
    expect(client.infer).toHaveBeenCalledWith({
      prompt: 'Explain adaptive routing.', routing_mode: 'manual', category: 'coding',
      max_output_tokens: 256, temperature: 1,
    });
  });

  it('renders adaptive category, router, and routing fallback without escalation', async () => {
    render(<App client={mockClient(fixtures.fallback)} />);
    advanced(); fireEvent.click(screen.getByLabelText('Manual category'));
    fireEvent.change(screen.getByLabelText('Task category'), { target: { value: 'coding' } });
    prompt(); send(); await screen.findByRole('article', { name: 'RoutLLM response' }); expand();
    expect(screen.getByText('Manual category: coding')).toBeVisible();
    expect(screen.getByText('Adaptive routing')).toBeVisible();
    expect(screen.getByText('Routing fallback: Yes')).toBeVisible();
    expect(screen.queryByText(/Validation escalation/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Escalated:/)).not.toBeInTheDocument();
  });

  it('keeps initial and final models distinct only when real execution metadata supports it', async () => {
    render(<App client={mockClient(fixtures.escalation)} />);
    prompt(); send(); await screen.findByRole('article', { name: 'RoutLLM response' }); expand();
    expect(screen.getByText('INITIAL MODEL')).toBeVisible();
    expect(screen.getByText('RETURNED MODEL')).toBeVisible();
    expect(screen.getByText('Validation escalation')).toBeVisible();
    expect(screen.getByText('$0.00000505 estimated total')).toBeVisible();
    expect(screen.getByText('No routing fallback')).toBeVisible();
  });

  it('prevents duplicate sends while pending', () => {
    const infer = vi.fn().mockReturnValue(new Promise(() => {}));
    render(<App client={{ infer, activity: vi.fn().mockResolvedValue({ items: [] }) }} />);
    prompt(); send();
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    fireEvent.submit(screen.getByRole('form', { name: 'Message composer' }));
    expect(infer).toHaveBeenCalledTimes(1);
  });

  it('recovers after a safe error and accepts the next request', async () => {
    const infer = vi.fn()
      .mockRejectedValueOnce(new InferenceError('unavailable', 'RoutLLM is temporarily unavailable.', 503))
      .mockResolvedValueOnce(fixtures.normal);
    render(<App client={{ infer, activity: vi.fn().mockResolvedValue({ items: [] }) }} />);
    prompt('first'); send();
    expect(await screen.findByRole('alert')).toHaveTextContent('RoutLLM is temporarily unavailable.');
    prompt('second'); send();
    expect(await screen.findByRole('article', { name: 'RoutLLM response' })).toBeVisible();
    expect(infer).toHaveBeenCalledTimes(2);
  });

  it('keeps validation visibly off and non-operative', () => {
    render(<App client={mockClient()} />); advanced();
    expect(screen.getByText('Response validation')).toBeVisible();
    expect(screen.getByText('Off')).toBeVisible();
    expect(screen.getByText(/Phase 12.2D/)).toBeVisible();
    expect(screen.queryByRole('checkbox', { name: /validation/i })).not.toBeInTheDocument();
  });

  it('loads a real empty Activity page without deriving rows from chat state', async () => {
    render(<App client={mockClient()} />);
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    expect(await screen.findByText('No activity yet')).toBeVisible();
    expect(screen.getByText(/Prompts and responses are not stored/)).toBeVisible();
  });
});

function activityItem(overrides: Partial<ActivityItem> = {}): ActivityItem {
  return {
    request_id: 'activity-1', created_at: '2026-09-29T20:00:00Z',
    execution_mode: 'direct', final_model_id: 'fake-small', provider: 'fake',
    attempt_count: 1, escalated: false, outcome: 'returned', input_tokens: 3,
    output_tokens: 2, latency_ms: 5, estimated_cost_usd: '0.00000135',
    cost_complete: true, ...overrides,
  };
}

describe('Activity view', () => {
  it('renders direct and adaptive rows with distinct fallback and escalation facts', async () => {
    const activity = vi.fn().mockResolvedValue({ items: [
      activityItem(),
      activityItem({
        request_id: 'activity-2', execution_mode: 'adaptive', category: 'coding',
        initial_model_id: 'fake-small', final_model_id: 'fake-large', attempt_count: 2,
        routing_fallback_used: true, escalated: true, estimated_cost_usd: '0.00000505',
      }),
    ] });
    render(<App client={{ infer: vi.fn(), activity }} />);
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    expect(await screen.findByText('fake-small')).toBeVisible();
    expect(screen.getByText('fake-small → fake-large')).toBeVisible();
    expect(screen.getByText('coding')).toBeVisible();
    expect(screen.getByText('Routing fallback')).toBeVisible();
    expect(screen.getByText('Validation escalation')).toBeVisible();
    expect(screen.getByText('$0.00000135')).toBeVisible();
    expect(activity).toHaveBeenCalledWith(expect.objectContaining({ limit: 25, cursor: undefined }));
  });

  it('loads the next cursor once, appends without duplicates, and refreshes by replacement', async () => {
    const activity = vi.fn()
      .mockResolvedValueOnce({ items: [activityItem()], next_cursor: 'opaque-next' })
      .mockResolvedValueOnce({ items: [activityItem(), activityItem({ request_id: 'activity-2' })], next_cursor: null })
      .mockResolvedValueOnce({ items: [activityItem({ request_id: 'activity-refreshed' })], next_cursor: null });
    render(<App client={{ infer: vi.fn(), activity }} />);
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    await screen.findByText('activity-1');
    fireEvent.click(screen.getByRole('button', { name: 'Load more' }));
    await screen.findByText('activity-2');
    expect(screen.getAllByText('activity-1')).toHaveLength(1);
    expect(activity).toHaveBeenNthCalledWith(2, expect.objectContaining({ cursor: 'opaque-next' }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(await screen.findByText('activity-refreshed')).toBeVisible();
    expect(screen.queryByText('activity-1')).not.toBeInTheDocument();
  });

  it('shows a safe initial error and prevents duplicate activity requests', async () => {
    let reject!: (reason: unknown) => void;
    const pending = new Promise((_, fail) => { reject = fail; });
    const activity = vi.fn().mockReturnValue(pending);
    render(<App client={{ infer: vi.fn(), activity }} />);
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    expect(await screen.findByText('Loading activity…')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(activity).toHaveBeenCalledTimes(1);
    await act(async () => reject(new InferenceError('unavailable', 'RoutLLM is temporarily unavailable.', 503)));
    expect(await screen.findByRole('alert')).toHaveTextContent('RoutLLM is temporarily unavailable.');
  });
});

describe('fixture injection boundary', () => {
  it('accepts deterministic simulation only when explicitly injected', async () => {
    vi.useFakeTimers();
    render(<App client={createSimulatedClient('normal')} />);
    prompt(); send();
    await act(async () => { await vi.advanceTimersByTimeAsync(450); });
    expect(screen.getByText(fixtures.normal.text, { normalizer: value => value })).toBeVisible();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('recent prompts only prepare text and never create fake activity or submit', () => {
    const client = mockClient(); render(<App client={client} />);
    const sidebar = screen.getByRole('complementary', { name: 'Conversations' });
    fireEvent.click(within(sidebar).getByRole('button', { name: 'Explain validation and escalation.' }));
    expect(screen.getByLabelText('Message RoutLLM')).toHaveValue('Explain validation and escalation.');
    expect(client.infer).not.toHaveBeenCalled();
  });
});
