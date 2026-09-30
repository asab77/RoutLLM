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
  it('keeps networking isolated to the relative HTTP adapter and has no storage, secrets, or HTML injection', () => {
    for (const path of sources('src')) {
      const source = readFileSync(path, 'utf8');
      if (path.endsWith('api/client.ts')) {
        expect(source).toContain("'/v1/chat'");
        expect(source).toContain('`/v1/activity?${parameters.toString()}`');
        expect(source).not.toMatch(/https?:\/\/|VITE_[A-Z_]+/);
      } else {
        expect(source, path).not.toMatch(/\b(fetch|XMLHttpRequest|WebSocket|EventSource|sendBeacon|localStorage|sessionStorage|indexedDB|serviceWorker|dangerouslySetInnerHTML)\b|document\.cookie|https?:\/\/|VITE_[A-Z_]+/);
      }
      expect(source, path).not.toMatch(/AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{24,}/);
    }
  });
  it('allows only same-origin production connections and keeps loopback-only tooling without a proxy', () => {
    const config = readFileSync('vite.config.ts', 'utf8');
    expect(config).toContain("connect-src 'self'"); expect(config).not.toMatch(/connect-src[^;]*\*/); expect(config).toContain("host: '127.0.0.1'"); expect(config).not.toMatch(/proxy\s*:/);
  });
  it('uses the HTTP client in production composition and exposes no fixture scenario toggle', () => {
    const app = readFileSync('src/App.tsx', 'utf8');
    const settings = readFileSync('src/components/Settings.tsx', 'utf8');
    expect(app).toContain('const productionClient = createHttpClient()');
    expect(app).not.toContain('createSimulatedClient');
    expect(settings).not.toMatch(/scenario|fixture/i);
  });
  it('defines both semantic palettes and reduced-motion behavior without remote CSS imports', () => {
    const css = readFileSync('src/styles.css', 'utf8');
    for (const token of ['background', 'surface', 'surface-elevated', 'border', 'text', 'text-muted', 'accent', 'accent-subtle', 'success', 'warning', 'error']) expect(css.match(new RegExp(`--${token}:`, 'g'))).toHaveLength(2);
    expect(css).toContain('prefers-reduced-motion'); expect(css).not.toMatch(/@import/);
  });
});
