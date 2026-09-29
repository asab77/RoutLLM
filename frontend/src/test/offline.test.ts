import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

function sources(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() && entry.name !== 'test' ? sources(path) : entry.isFile() && /\.(tsx?|css)$/.test(path) && !path.endsWith('.test.ts') && !path.endsWith('.test.tsx') ? [path] : [];
  });
}
describe('structural offline boundary', () => {
  it('keeps the send control fixed-size in a separate bottom-aligned action region', () => {
    const css = readFileSync('src/styles.css', 'utf8');
    const button = css.match(/\.send-button \{([^}]+)\}/)![1];
    for (const property of ['width', 'height', 'min-width', 'max-width', 'min-height', 'max-height']) expect(button).toContain(`${property}: 34px;`);
    expect(button).toContain('flex-shrink: 0;');
    expect(button).toContain('border-radius: 50%;');
    expect(button).not.toMatch(/position:\s*absolute|height:\s*100%/);
    const actions = css.match(/\.composer-actions \{([^}]+)\}/)![1];
    expect(actions).toContain('flex: 0 0 34px;');
    expect(actions).toContain('align-items: flex-end;');
    expect(actions).not.toContain('absolute');
  });
  it('has no networking, storage, external assets, secrets, or HTML injection in application source', () => {
    for (const path of sources('src')) {
      const source = readFileSync(path, 'utf8');
      expect(source, path).not.toMatch(/\b(fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|localStorage|sessionStorage|indexedDB|serviceWorker|dangerouslySetInnerHTML)\b|document\.cookie|https?:\/\/|VITE_[A-Z_]+/);
      expect(source, path).not.toMatch(/AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{24,}/);
    }
  });
  it('retains the production no-connect CSP and loopback-only tooling without a proxy', () => {
    const config = readFileSync('vite.config.ts', 'utf8');
    expect(config).toContain("connect-src 'none'"); expect(config).toContain("host: '127.0.0.1'"); expect(config).not.toMatch(/proxy\s*:/);
  });
  it('defines both semantic palettes and reduced-motion behavior without remote CSS imports', () => {
    const css = readFileSync('src/styles.css', 'utf8');
    for (const token of ['background', 'surface', 'surface-elevated', 'border', 'text', 'text-muted', 'accent', 'accent-subtle', 'success', 'warning', 'error']) expect(css.match(new RegExp(`--${token}:`, 'g'))).toHaveLength(2);
    expect(css).toContain('prefers-reduced-motion'); expect(css).not.toMatch(/@import/);
  });
});
