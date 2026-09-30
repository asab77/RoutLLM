import type { ReactNode } from 'react';
export function Header({ children, onMenu, sidebarOpen }: { children: ReactNode; onMenu: () => void; sidebarOpen: boolean }) {
  return <header className="topbar"><div className="topbar-title"><button className="icon-button menu-button" aria-label="Toggle conversations" aria-expanded={sidebarOpen} onClick={onMenu}>☰</button><span>RoutLLM <span className="muted">/ Chat</span></span></div><div className="topbar-actions"><div className="simulation-status"><strong><span aria-hidden="true">●</span> LIVE API</strong><small>Same-origin</small></div>{children}</div></header>;
}
