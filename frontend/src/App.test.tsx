import { act, fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import App from './App';
import { ResponsePanel } from './components/ResponsePanel';
import { RequestMetrics } from './components/RequestMetrics';
import { fixtures } from './mocks/fixtures';
import * as api from './api/client';

function settings() { fireEvent.click(screen.getByRole('button', { name: 'Settings' })); }
function advanced() { settings(); fireEvent.click(screen.getByRole('switch', { name: 'Advanced controls' })); }
function prompt(value = 'Explain adaptive routing.') { fireEvent.change(screen.getByLabelText('Message RoutLLM'), { target: { value } }); }
function submit(scenario = 'normal') {
  if (scenario !== 'normal') {
    settings(); fireEvent.click(screen.getByText('Demo scenarios'));
    fireEvent.change(screen.getByLabelText('Response scenario'), { target: { value: scenario } }); settings();
  }
  prompt(); fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
}
async function finish() { await act(async () => { await vi.advanceTimersByTimeAsync(450); }); }
function expand() { fireEvent.click(screen.getByText('View route').closest('summary')!); }
function systemPreference(dark: boolean) {
  const listeners = new Set<() => void>();
  const preference = { matches: dark, addEventListener: vi.fn((_event, fn: () => void) => listeners.add(fn)), removeEventListener: vi.fn((_event, fn: () => void) => listeners.delete(fn)) };
  vi.stubGlobal('matchMedia', vi.fn(() => preference));
  return { preference, change(value: boolean) { act(() => { preference.matches = value; listeners.forEach(fn => fn()); }); } };
}

describe('conversation-first interface', () => {
  it('starts in simulation with a composer, no old dashboard controls, and no automatic inference', () => {
    const client = vi.spyOn(api, 'createSimulatedClient');
    render(<App />);
    expect(screen.getByText('SIMULATION')).toBeVisible();
    expect(screen.getByText('No provider calls')).toBeVisible();
    const composer = screen.getByRole('form', { name: 'Message composer' });
    expect(composer).toBeVisible();
    const actions = screen.getByRole('button', { name: 'Send message' }).parentElement!;
    expect(actions).toHaveClass('composer-actions');
    expect(actions.parentElement).toBe(composer);
    expect(screen.getByLabelText('Message RoutLLM').parentElement).toBe(composer);
    expect(actions).not.toContainElement(screen.getByLabelText('Message RoutLLM'));
    expect(screen.queryByText('Explore the route behind the reply')).not.toBeInTheDocument();
    expect(screen.getByText('Enter to send · Shift+Enter for a new line · Conversation is not saved')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Run Adaptive Inference' })).not.toBeInTheDocument();
    for (const label of ['Task category', 'Max output tokens', 'Quality threshold', 'Temperature']) expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Advanced controls' })).not.toBeInTheDocument();
    expect(client).not.toHaveBeenCalled();
  });

  it('blocks empty and whitespace-only sends', () => {
    const client = vi.spyOn(api, 'createSimulatedClient'); render(<App />);
    const send = screen.getByRole('button', { name: 'Send message' });
    expect(send).toBeDisabled(); prompt(' \n '); expect(send).toBeDisabled();
    fireEvent.submit(screen.getByRole('form')); expect(client).not.toHaveBeenCalled();
  });

  it('sends the exact request with fixture QA, locks duplicates, and shows user text', async () => {
    vi.useFakeTimers();
    const infer = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.spyOn(api, 'createSimulatedClient').mockReturnValue({ infer });
    render(<App />); submit();
    expect(infer).toHaveBeenCalledWith({ prompt: 'Explain adaptive routing.', category: 'qa', max_output_tokens: 256, temperature: 1, quality_threshold: 0.8 });
    expect(screen.getByText('Explain adaptive routing.')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /New chat/ })).toBeDisabled();
    fireEvent.submit(screen.getByRole('form')); expect(infer).toHaveBeenCalledTimes(1);
  });

  it('supports Shift+Enter newline, Enter send, and does not send during IME composition', async () => {
    const user = userEvent.setup();
    const infer = vi.fn().mockReturnValue(new Promise(() => {}));
    vi.spyOn(api, 'createSimulatedClient').mockReturnValue({ infer });
    render(<App />);
    await user.type(screen.getByLabelText('Message RoutLLM'), 'First{Shift>}{Enter}{/Shift}Second');
    expect(screen.getByLabelText('Message RoutLLM')).toHaveValue('First\nSecond');
    fireEvent.keyDown(screen.getByLabelText('Message RoutLLM'), { key: 'Enter', isComposing: true });
    expect(infer).not.toHaveBeenCalled();
    await user.keyboard('{Enter}'); expect(infer).toHaveBeenCalledTimes(1);
  });

  it('renders a deterministic reply and collapsed trace with exact tiny cost', async () => {
    vi.useFakeTimers(); render(<App />); submit(); await finish();
    expect(screen.getByText(fixtures.normal.text, { normalizer: value => value })).toBeVisible();
    const disclosure = screen.getByText('View route').closest('details')!;
    expect(disclosure).not.toHaveAttribute('open');
    expect(within(disclosure).getByText(/184 ms · 12 → 5 tokens · \$0.00000135/)).toBeVisible();
    expand(); expect(disclosure).toHaveAttribute('open');
    expect(screen.getByText('candidate-nemotron-3.5-lightning')).toBeVisible();
    expect(screen.getByText(/Detected: QA/)).toHaveTextContent('(fixture)');
    expect(screen.getByText('✓ Threshold met')).toBeVisible();
    expect(screen.getByText('No fallback')).toBeVisible();
    expect(screen.getByText('quality_threshold_met')).toBeVisible();
    expect(screen.getByText('sim-threshold-001')).toBeVisible();
    expect(screen.queryByText(/confidence|80% correct|predicted quality|savings/i)).not.toBeInTheDocument();
    expect(screen.queryByText('RETURNED MODEL')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Hide route').closest('summary')!); expect(disclosure).not.toHaveAttribute('open');
  });

  it('preserves multiple turns and New chat clears contents without clearing theme or controls', async () => {
    vi.useFakeTimers(); render(<App />); submit(); await finish(); submit(); await finish();
    expect(screen.getAllByRole('article', { name: 'RoutLLM response' })).toHaveLength(2);
    settings(); fireEvent.click(screen.getByLabelText('Dark')); settings();
    fireEvent.click(screen.getByRole('button', { name: /New chat/ }));
    expect(screen.queryByRole('article', { name: 'RoutLLM response' })).not.toBeInTheDocument();
    expect(screen.queryByText('Explain adaptive routing.')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Message RoutLLM')).toHaveValue('');
    expect(document.querySelector('.app')).toHaveAttribute('data-theme', 'dark');
  });

  it('keeps recent examples clearly labeled and only prepares a draft, never auto-sends', () => {
    const client = vi.spyOn(api, 'createSimulatedClient'); render(<App />);
    const sidebar = screen.getByRole('complementary', { name: 'Conversations' });
    expect(within(sidebar).getByText(/Example prompts, not saved history/)).toBeVisible();
    fireEvent.click(within(sidebar).getByRole('button', { name: 'Explain validation and escalation.' }));
    expect(screen.getByLabelText('Message RoutLLM')).toHaveValue('Explain validation and escalation.');
    expect(client).not.toHaveBeenCalled();
  });

  it('keeps Activity empty of telemetry and restores the current conversation', async () => {
    vi.useFakeTimers(); render(<App />); submit(); await finish();
    fireEvent.click(screen.getByRole('button', { name: /Activity/ }));
    expect(screen.getByText('Durable telemetry integration will appear here.')).toBeVisible();
    expect(screen.queryByRole('region', { name: 'Request metrics' })).not.toBeInTheDocument();
    expect(screen.getByRole('main').textContent).not.toMatch(/\$|\d+\s*ms|\d+%/);
    fireEvent.click(screen.getByText('Back to conversation →'));
    expect(screen.getByText(fixtures.normal.text, { normalizer: value => value })).toBeVisible();
  });
});

describe('settings and optional controls', () => {
  it('opens a small Settings surface with Advanced controls OFF, then toggles and closes consistently', () => {
    render(<App />); settings();
    const toggle = screen.getByRole('switch', { name: 'Advanced controls' });
    expect(toggle).not.toBeChecked();
    fireEvent.click(toggle); expect(screen.getByRole('complementary', { name: 'Advanced controls' })).toBeVisible();
    fireEvent.click(toggle); expect(screen.queryByRole('complementary', { name: 'Advanced controls' })).not.toBeInTheDocument();
    fireEvent.click(toggle); fireEvent.click(screen.getByRole('button', { name: 'Close advanced controls' }));
    expect(toggle).not.toBeChecked(); expect(screen.queryByRole('complementary', { name: 'Advanced controls' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Settings' })).toHaveFocus();
  });

  it('dismisses Settings with Escape and outside interaction', () => {
    render(<App />); settings();
    expect(screen.getByRole('switch')).toHaveFocus();
    fireEvent.keyDown(screen.getByRole('switch'), { key: 'Escape' });
    expect(screen.queryByRole('region', { name: 'Settings' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Settings' })).toHaveFocus();
    settings(); fireEvent.pointerDown(screen.getByRole('main'));
    expect(screen.queryByRole('region', { name: 'Settings' })).not.toBeInTheDocument();
  });

  it('defaults to fixture auto-detection, locks threshold, and exposes all seven manual categories', () => {
    render(<App />); advanced();
    expect(screen.getByRole('radio', { name: /Auto-detect/ })).toBeChecked();
    expect(screen.queryByLabelText('Task category')).not.toBeInTheDocument();
    expect(screen.getByLabelText(/Quality threshold/)).toHaveTextContent('0.80');
    expect(screen.getByLabelText(/Quality threshold/).tagName).toBe('OUTPUT');
    expect(screen.getByText(/not a correctness score/)).toBeVisible();
    expect(screen.getByText(/Production detection is not implemented/)).toBeVisible();
    fireEvent.click(screen.getByLabelText('Manual override'));
    expect(within(screen.getByLabelText('Task category')).getAllByRole('option').map(option => option.textContent)).toEqual(['classification', 'coding', 'extraction', 'structured_json', 'qa', 'reasoning', 'summarization']);
    expect(screen.getByText('Off')).toBeVisible();
  });

  it('uses manual category and generation values in the exact typed request', async () => {
    vi.useFakeTimers();
    const original = api.createSimulatedClient('normal'); const infer = vi.fn(original.infer);
    vi.spyOn(api, 'createSimulatedClient').mockReturnValue({ infer });
    render(<App />); advanced(); fireEvent.click(screen.getByLabelText('Manual override'));
    fireEvent.change(screen.getByLabelText('Task category'), { target: { value: 'coding' } });
    fireEvent.change(screen.getByLabelText('Max output tokens'), { target: { value: '512' } });
    fireEvent.change(screen.getByLabelText('Temperature'), { target: { value: '0.5' } });
    settings(); submit(); await finish(); expand();
    expect(infer).toHaveBeenCalledWith({ prompt: 'Explain adaptive routing.', category: 'coding', max_output_tokens: 512, temperature: 0.5, quality_threshold: 0.8 });
    expect(screen.getByText(/Manual category: coding/)).toHaveTextContent('(override)');
  });

  it.each(['0', '-1', '513', '1.5', ''])('blocks invalid max output tokens %s even after closing controls', value => {
    render(<App />); advanced(); prompt();
    fireEvent.change(screen.getByLabelText('Max output tokens'), { target: { value } });
    expect(screen.getByLabelText('Max output tokens')).toHaveAttribute('aria-invalid', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Close advanced controls' }));
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent('Check token and temperature limits');
  });

  it.each(['-0.1', '2.1', ''])('blocks invalid temperature %s', value => {
    render(<App />); advanced(); prompt();
    fireEvent.change(screen.getByLabelText('Temperature'), { target: { value } });
    expect(screen.getByLabelText('Temperature')).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
  });

  it.each([['1', '0'], ['256', '1'], ['512', '2']])('accepts boundary/default values %s tokens and %s temperature', (tokens, temperature) => {
    render(<App />); advanced(); prompt();
    fireEvent.change(screen.getByLabelText('Max output tokens'), { target: { value: tokens } });
    fireEvent.change(screen.getByLabelText('Temperature'), { target: { value: temperature } });
    expect(screen.getByRole('button', { name: 'Send message' })).toBeEnabled();
  });
});

describe('themes', () => {
  it.each([true, false])('System defaults to OS preference (dark=%s), reacts to changes, and cleans its listener', dark => {
    const system = systemPreference(dark); const { container, unmount } = render(<App />); settings();
    expect(screen.getByRole('radio', { name: 'System' })).toBeChecked();
    expect(container.firstChild).toHaveAttribute('data-theme', dark ? 'dark' : 'light');
    system.change(!dark); expect(container.firstChild).toHaveAttribute('data-theme', dark ? 'light' : 'dark');
    expect(window.matchMedia).toHaveBeenCalledWith('(prefers-color-scheme: dark)');
    unmount(); expect(system.preference.removeEventListener).toHaveBeenCalledWith('change', expect.any(Function));
  });
  it('supports labeled Dark and Light overrides, then returns to System without persistence', () => {
    const system = systemPreference(true); const { container } = render(<App />); settings();
    fireEvent.click(screen.getByRole('radio', { name: 'Light' })); expect(container.firstChild).toHaveAttribute('data-theme', 'light');
    system.change(false); fireEvent.click(screen.getByRole('radio', { name: 'Dark' }));
    expect(container.firstChild).toHaveAttribute('data-theme', 'dark');
    system.change(true); system.change(false); expect(container.firstChild).toHaveAttribute('data-theme', 'dark');
    fireEvent.click(screen.getByRole('radio', { name: 'System' })); expect(container.firstChild).toHaveAttribute('data-theme', 'light');
    expect(Storage.prototype.setItem).not.toHaveBeenCalled();
  });
});

describe('fixture semantics and safety', () => {
  it('renders fallback without inventing escalation', async () => {
    vi.useFakeTimers(); render(<App />); submit('fallback'); await finish(); expand();
    expect(screen.getByText('No model met threshold; fallback selected')).toBeVisible();
    expect(screen.getByText('Fallback used: Yes')).toBeVisible();
    expect(screen.getByText('candidate-claude-sonnet-5')).toBeVisible();
    expect(screen.queryByText('RETURNED MODEL')).not.toBeInTheDocument();
  });
  it('distinguishes initial and returned models on escalation without labeling it fallback', async () => {
    vi.useFakeTimers(); render(<App />); submit('escalation'); await finish(); expand();
    expect(screen.getByText('INITIAL MODEL')).toBeVisible(); expect(screen.getByText('RETURNED MODEL')).toBeVisible();
    expect(screen.getByText(fixtures.escalation.routing.selected_model_id)).toBeVisible();
    expect(screen.getByText(fixtures.escalation.model_id)).toBeVisible(); expect(screen.getByText('No fallback')).toBeVisible();
    expect(screen.getByText('$0.00000505 estimated total')).toBeVisible();
    expect(screen.getByText(/Escalated: Yes · Validation: passed/)).toBeVisible();
  });
  it('shows a deterministic error without retry or a false success trace', async () => {
    vi.useFakeTimers(); const client = vi.spyOn(api, 'createSimulatedClient');
    render(<App />); submit('error'); await finish();
    expect(screen.getByRole('alert')).toHaveTextContent('Inference is currently unavailable');
    expect(screen.queryByText('Routed by RoutLLM')).not.toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(60_000); }); expect(client).toHaveBeenCalledTimes(1);
  });
  it('keeps response content as inert text', () => {
    const text = '<img src="https://example.invalid" onerror="alert(1)"><script>alert(1)</script>';
    const { container } = render(<ResponsePanel response={{ ...fixtures.normal, text }} />);
    expect(screen.getByText(text)).toBeVisible(); expect(container.querySelector('img, script')).toBeNull();
  });
  it('does not fabricate absent cumulative cost', () => {
    render(<RequestMetrics response={{ ...fixtures.escalation, execution: { attempts: 2, escalated: true, validation_outcome: 'passed', total_latency_ms: 424 } }} />);
    expect(screen.getByText('Total cost unavailable')).toBeVisible();
  });
  it('clears conversation on remount and never invokes network or persistence', async () => {
    vi.useFakeTimers(); const { unmount } = render(<App />); submit(); await finish(); unmount(); render(<App />);
    expect(screen.getByLabelText('Message RoutLLM')).toHaveValue(''); expect(screen.queryByText(fixtures.normal.text, { normalizer: value => value })).not.toBeInTheDocument();
    expect(fetch).not.toHaveBeenCalled(); expect(Storage.prototype.setItem).not.toHaveBeenCalled(); expect(Storage.prototype.getItem).not.toHaveBeenCalled();
  });
});
