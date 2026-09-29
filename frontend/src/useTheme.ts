import { useEffect, useState } from 'react';
export type Theme = 'dark' | 'light' | 'system';
export function useTheme() {
  const [theme, setTheme] = useState<Theme>('system');
  const [systemDark, setSystemDark] = useState(() => window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false);
  useEffect(() => {
    const preference = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!preference) return;
    const update = () => setSystemDark(preference.matches);
    update(); preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);
  return [theme, setTheme, theme === 'system' ? systemDark ? 'dark' : 'light' : theme] as const;
}
