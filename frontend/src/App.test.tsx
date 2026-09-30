import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import { InferenceError, createSimulatedClient } from './api/client';
import type { ActivityItem, AdaptiveChatResponse, ChatResponse, InferenceClient } from './api/types';
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
function manual(category: string) {
  advanced();
  fireEvent.click(screen.getByLabelText('Manual category'));
  fireEvent.change(screen.getByLabelText('Task category'), { target: { value: category } });
}
function enableContract() { fireEvent.click(screen.getByRole('checkbox', { name: /Enable output contract/ })); }

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
    expect(screen.getByText('Output-contract escalation')).toBeVisible();
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

  it('keeps output contracts unavailable in AUTO without semantic-quality claims', () => {
    render(<App client={mockClient()} />); advanced();
    expect(screen.getByText('Output contract & escalation')).toBeVisible();
    expect(screen.getByText('Manual routing required')).toBeVisible();
    expect(screen.getByText(/does not judge answer correctness/i)).toBeVisible();
    expect(screen.queryByRole('checkbox', { name: /output contract/i })).not.toBeInTheDocument();
  });

  it.each(['coding', 'qa', 'reasoning', 'summarization'])('does not enable contracts for %s', category => {
    render(<App client={mockClient()} />); manual(category);
    expect(screen.getByText(`Unavailable for ${category}`)).toBeVisible();
    expect(screen.queryByRole('checkbox', { name: /Enable output contract/ })).not.toBeInTheDocument();
  });

  it.each(['classification', 'extraction', 'structured_json'])('offers contracts for %s', category => {
    render(<App client={mockClient()} />); manual(category);
    expect(screen.getByRole('checkbox', { name: /Enable output contract/ })).toBeEnabled();
  });

  it('constructs a trimmed classification contract without changing the prompt or sending threshold', async () => {
    const client = mockClient(fixtures.fallback);
    render(<App client={client} />); manual('classification'); enableContract();
    prompt('Return exactly one configured label.');
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Allowed labels'), { target: { value: ' APPROVE \nREVIEW\nREJECT ' } });
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled();
    send(); await screen.findByRole('article', { name: 'RoutLLM response' });
    expect(client.infer).toHaveBeenCalledWith({
      prompt: 'Return exactly one configured label.', routing_mode: 'manual', category: 'classification',
      max_output_tokens: 256, temperature: 1,
      validation: { format: 'label', allowed_labels: ['APPROVE', 'REVIEW', 'REJECT'] },
    });
  });

  it('blocks duplicate and blank classification labels', () => {
    render(<App client={mockClient()} />); manual('classification'); enableContract(); prompt();
    fireEvent.change(screen.getByLabelText('Allowed labels'), { target: { value: 'YES\n\nNO' } });
    expect(screen.getByText('Labels cannot be blank.')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Allowed labels'), { target: { value: 'YES\nYES' } });
    expect(screen.getByText('Labels must be unique.')).toBeVisible();
  });

  it.each(['extraction', 'structured_json'])('constructs a shallow required object contract for %s', async category => {
    const client = mockClient(fixtures.fallback);
    render(<App client={client} />); manual(category); enableContract(); prompt('Return the requested object.');
    fireEvent.change(screen.getByLabelText('Field 1 name'), { target: { value: ' name ' } });
    fireEvent.change(screen.getByLabelText('Field 1 type'), { target: { value: 'string' } });
    fireEvent.click(screen.getByRole('button', { name: '＋ Add field' }));
    fireEvent.change(screen.getByLabelText('Field 2 name'), { target: { value: 'amount' } });
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled();
    send(); await screen.findByRole('article', { name: 'RoutLLM response' });
    expect(client.infer).toHaveBeenCalledWith(expect.objectContaining({
      prompt: 'Return the requested object.', routing_mode: 'manual', category,
      validation: {
        format: 'json', root_type: 'object', required_fields: ['name', 'amount'],
        field_types: { name: 'string' },
      },
    }));
  });

  it('blocks invalid JSON fields and clears stale contracts on mode/category changes', () => {
    render(<App client={mockClient()} />); manual('extraction'); enableContract(); prompt();
    expect(screen.getByText('Field names cannot be blank.')).toBeVisible();
    fireEvent.change(screen.getByLabelText('Field 1 name'), { target: { value: 'name' } });
    fireEvent.click(screen.getByRole('button', { name: '＋ Add field' }));
    fireEvent.change(screen.getByLabelText('Field 2 name'), { target: { value: 'name' } });
    expect(screen.getByText('Field names must be unique.')).toBeVisible();
    fireEvent.click(screen.getByLabelText('Auto (recommended)'));
    fireEvent.click(screen.getByLabelText('Manual category'));
    expect(screen.getByRole('checkbox', { name: /Enable output contract/ })).not.toBeChecked();
    fireEvent.click(screen.getByRole('checkbox', { name: /Enable output contract/ }));
    fireEvent.change(screen.getByLabelText('Task category'), { target: { value: 'qa' } });
    expect(screen.getByText('Unavailable for qa')).toBeVisible();
    fireEvent.change(screen.getByLabelText('Task category'), { target: { value: 'extraction' } });
    expect(screen.getByRole('checkbox', { name: /Enable output contract/ })).not.toBeChecked();
  });

  it('renders a one-attempt contract pass without escalation', async () => {
    const response: AdaptiveChatResponse = {
      ...(fixtures.fallback as AdaptiveChatResponse), category: 'classification',
      execution: { attempts: 1, escalated: false, validation_outcome: 'passed', total_estimated_cost_usd: '0.000074', total_latency_ms: 426 },
    };
    render(<App client={mockClient(response)} />); prompt(); send();
    await screen.findByRole('article', { name: 'RoutLLM response' }); expand();
    expect(screen.getByText('Contract passed')).toBeVisible();
    expect(screen.getByText('1 attempt')).toBeVisible();
    expect(screen.queryByText('Output-contract escalation')).not.toBeInTheDocument();
  });

  it('summarizes three attempts without inventing an intermediate model or failure reason', async () => {
    const base = fixtures.escalation as AdaptiveChatResponse;
    const response: AdaptiveChatResponse = { ...base, execution: { ...base.execution!, attempts: 3 } };
    render(<App client={mockClient(response)} />); prompt(); send();
    await screen.findByRole('article', { name: 'RoutLLM response' }); expand();
    expect(screen.getByText('3 total attempts')).toBeVisible();
    expect(screen.getByText('Intermediate model details are not exposed.')).toBeVisible();
    expect(screen.queryByText(/invalid json|missing field|failed constraint/i)).not.toBeInTheDocument();
  });

  it('uses the safe terminal output-contract failure message without invented facts', async () => {
    const infer = vi.fn().mockRejectedValue(new InferenceError('validation', 'Output contract was not satisfied after the available attempts.', 502, 'validation-1'));
    render(<App client={{ infer, activity: vi.fn().mockResolvedValue({ items: [] }) }} />);
    prompt(); send();
    const error = await screen.findByRole('alert');
    expect(error).toHaveTextContent('Output contract was not satisfied after the available attempts.');
    expect(error).not.toHaveTextContent(/model|constraint|cost|latency/i);
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
        routing_fallback_used: true, escalated: true, validation_outcome: 'passed', estimated_cost_usd: '0.00000505',
      }),
    ] });
    render(<App client={{ infer: vi.fn(), activity }} />);
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    expect(await screen.findByText('fake-small')).toBeVisible();
    expect(screen.getByText('fake-small → fake-large')).toBeVisible();
    expect(screen.getByText('coding')).toBeVisible();
    expect(screen.getByText('Routing fallback')).toBeVisible();
    expect(screen.getByText('Output-contract escalation')).toBeVisible();
    expect(screen.getByText('Contract passed')).toBeVisible();
    expect(screen.getByText('$0.00000135')).toBeVisible();
    expect(activity).toHaveBeenCalledWith(expect.objectContaining({ limit: 25, cursor: undefined }));
  });

  it('shows terminal validation failure without inventing missing execution facts', async () => {
    const activity = vi.fn().mockResolvedValue({ items: [activityItem({
      request_id: 'activity-failed', execution_mode: 'adaptive', category: 'structured_json',
      final_model_id: undefined, provider: undefined, attempt_count: undefined,
      escalated: undefined, validation_outcome: undefined, outcome: 'validation_failed',
      input_tokens: undefined, output_tokens: undefined, latency_ms: undefined,
      estimated_cost_usd: undefined, cost_complete: false,
    })] });
    render(<App client={{ infer: vi.fn(), activity }} />);
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    expect(await screen.findByText('Validation failed')).toBeVisible();
    const row = screen.getByText('activity-failed').closest('tr')!;
    expect(within(row).queryByText(/attempt|escalation|Contract passed/)).not.toBeInTheDocument();
    expect(within(row).getAllByText('—').length).toBeGreaterThan(1);
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
