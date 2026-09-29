import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, expect, vi } from 'vitest';

let forbiddenCalls = 0;

beforeEach(() => {
  forbiddenCalls = 0;
  const forbidden = () => {
    forbiddenCalls += 1;
    throw new Error('Network and persistence are forbidden in frontend tests.');
  };
  vi.stubGlobal('fetch', vi.fn(forbidden));
  vi.stubGlobal('XMLHttpRequest', vi.fn(forbidden));
  vi.stubGlobal('WebSocket', vi.fn(forbidden));
  vi.stubGlobal('EventSource', vi.fn(forbidden));
  vi.stubGlobal('indexedDB', { open: vi.fn(forbidden) });
  vi.stubGlobal('navigator', Object.assign(Object.create(navigator), { sendBeacon: vi.fn(forbidden) }));
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(forbidden);
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(forbidden);
  vi.spyOn(document, 'cookie', 'set').mockImplementation(forbidden);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  expect(forbiddenCalls, 'No network or persistence attempt, even if caught').toBe(0);
});
