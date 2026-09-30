import { useEffect, useRef, useState } from 'react';
import type { Theme } from '../useTheme';
interface Props { advanced: boolean; onAdvanced: (value: boolean) => void; theme: Theme; onTheme: (value: Theme) => void }
export function Settings({ advanced, onAdvanced, theme, onTheme }: Props) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const first = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!open) return;
    first.current?.focus();
    const dismiss = (event: PointerEvent) => { if (!container.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, [open]);
  return <div className="settings" ref={container} onKeyDown={event => { if (event.key === 'Escape') { setOpen(false); trigger.current?.focus(); } }}>
    <button ref={trigger} className="icon-button" aria-label="Settings" aria-expanded={open} aria-controls="settings-popover" onClick={() => setOpen(value => !value)}><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m9 3-1 3-3 1-2 5 2 5 3 1 1 3h6l1-3 3-1 2-5-2-5-3-1-1-3Z"/><circle cx="12" cy="12" r="3"/></svg></button>
    {open && <section className="settings-popover" id="settings-popover" aria-label="Settings"><h2>Settings</h2><label className="switch-row">Advanced controls<input ref={first} type="checkbox" role="switch" checked={advanced} onChange={event => onAdvanced(event.target.checked)} /></label><fieldset className="theme-picker"><legend>Theme</legend>{(['dark', 'light', 'system'] as const).map(mode => <label key={mode}><input type="radio" name="theme" value={mode} checked={theme === mode} onChange={() => onTheme(mode)} />{mode[0].toUpperCase() + mode.slice(1)}</label>)}</fieldset><details><summary>About</summary><p>RoutLLM uses direct inference in Auto mode and the adaptive router when you choose a category manually.</p></details></section>}
  </div>;
}
