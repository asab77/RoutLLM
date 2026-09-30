import { useImperativeHandle, useLayoutEffect, useRef } from 'react';
import type { Ref } from 'react';
interface Props { ref: Ref<HTMLTextAreaElement>; value: string; onChange: (value: string) => void; onSend: () => void; pending: boolean; validSettings: boolean; settingsMessage?: string; onOpenControls: () => void }
export function Composer({ ref, value, onChange, onSend, pending, validSettings, settingsMessage, onOpenControls }: Props) {
  const textarea = useRef<HTMLTextAreaElement>(null);
  useImperativeHandle(ref, () => textarea.current!);
  function resize() {
    const element = textarea.current;
    if (!element) return;
    // Reset before measuring so deletion and wider layouts can shrink the field.
    element.style.height = '0px';
    element.style.overflowY = 'hidden';
    const height = element.scrollHeight;
    element.style.height = `${Math.min(220, Math.max(34, height))}px`;
    element.style.overflowY = height > 220 ? 'auto' : 'hidden';
  }
  useLayoutEffect(resize, [value]);
  useLayoutEffect(() => {
    const element = textarea.current;
    if (!element) return;
    let previousWidth = element.getBoundingClientRect().width;
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(entries => {
      const width = entries[0]?.contentRect.width;
      if (width !== undefined && width !== previousWidth) { previousWidth = width; resize(); }
    });
    observer?.observe(element);
    window.addEventListener('resize', resize);
    return () => { observer?.disconnect(); window.removeEventListener('resize', resize); };
  }, []);
  const disabled = pending || !value.trim() || !validSettings || value.length > 12_000;
  return <div className="composer-dock"><form className="composer" aria-label="Message composer" onSubmit={event => { event.preventDefault(); if (!disabled) onSend(); }}>
    <label className="sr-only" htmlFor="message">Message RoutLLM</label><textarea ref={textarea} id="message" placeholder="Message RoutLLM…" rows={1} maxLength={12_000} required value={value} disabled={pending} onChange={event => onChange(event.target.value)} aria-describedby="composer-help" onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (!disabled) onSend(); } }} />
    <div className="composer-actions"><button className="send-button" type="submit" aria-label="Send message" disabled={disabled}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M12 19V5m-6 6 6-6 6 6"/></svg></button></div>
  </form>{!validSettings && <p className="settings-error" role="alert">{settingsMessage ?? 'Check request settings'} in <button onClick={onOpenControls}>Advanced controls</button>.</p>}<p className="composer-help" id="composer-help">Enter to send · Shift+Enter for a new line · Conversation is not saved</p></div>;
}
