import { createRef } from 'react';
import { act, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { Composer } from './Composer';

it('measures wrapped content, caps scrolling, and shrinks on deletion or clearing', () => {
  let height = 24;
  vi.spyOn(HTMLTextAreaElement.prototype, 'scrollHeight', 'get').mockImplementation(() => height);
  const ref = createRef<HTMLTextAreaElement>();
  const props = { ref, onChange: vi.fn(), onSend: vi.fn(), pending: false, validSettings: true, onOpenControls: vi.fn() };
  const { rerender } = render(<Composer {...props} value="Hi" />);
  const input = screen.getByLabelText('Message RoutLLM');
  expect(ref.current).toBe(input);
  expect(input).toHaveStyle({ height: '34px', overflowY: 'hidden' });
  height = 110; rerender(<Composer {...props} value={'Wrapped words '.repeat(30)} />);
  expect(input).toHaveStyle({ height: '110px', overflowY: 'hidden' });
  height = 350; rerender(<Composer {...props} value={'Many lines\n'.repeat(30)} />);
  expect(input).toHaveStyle({ height: '220px', overflowY: 'auto' });
  height = 60; rerender(<Composer {...props} value="Less text" />);
  expect(input).toHaveStyle({ height: '60px', overflowY: 'hidden' });
  height = 24; rerender(<Composer {...props} value="" />);
  expect(input).toHaveStyle({ height: '34px', overflowY: 'hidden' });
});

it('remeasures for layout width changes, ignores height-only notifications, and cleans up', () => {
  let height = 50;
  const measure = vi.spyOn(HTMLTextAreaElement.prototype, 'scrollHeight', 'get').mockImplementation(() => height);
  let notify: (entries: { contentRect: { width: number } }[]) => void = () => {};
  const disconnect = vi.fn();
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: typeof notify) { notify = callback; }
    observe = vi.fn();
    disconnect = disconnect;
  });
  const { unmount } = render(<Composer ref={createRef()} value="A wrapping message" onChange={vi.fn()} onSend={vi.fn()} pending={false} validSettings onOpenControls={vi.fn()} />);
  const input = screen.getByLabelText('Message RoutLLM');
  height = 150; act(() => notify([{ contentRect: { width: 200 } }]));
  expect(input).toHaveStyle({ height: '150px' });
  measure.mockClear(); act(() => notify([{ contentRect: { width: 200 } }]));
  expect(measure).not.toHaveBeenCalled();
  height = 40; act(() => notify([{ contentRect: { width: 600 } }]));
  expect(input).toHaveStyle({ height: '40px' });
  height = 90; act(() => window.dispatchEvent(new Event('resize')));
  expect(input).toHaveStyle({ height: '90px' });
  unmount(); expect(disconnect).toHaveBeenCalledOnce();
});
