import { categories } from '../api/types';
import type { Category } from '../api/types';
export function validGeneration(tokens: string, temperature: string) {
  return tokens.trim() !== '' && Number.isInteger(Number(tokens)) && Number(tokens) >= 1 && Number(tokens) <= 512 && temperature.trim() !== '' && Number.isFinite(Number(temperature)) && Number(temperature) >= 0 && Number(temperature) <= 2;
}
interface Props { onClose: () => void; manual: boolean; onManual: (value: boolean) => void; category: Category; onCategory: (value: Category) => void; tokens: string; onTokens: (value: string) => void; temperature: string; onTemperature: (value: string) => void; pending: boolean }
export function AdvancedControls({ onClose, manual, onManual, category, onCategory, tokens, onTokens, temperature, onTemperature, pending }: Props) {
  function close() { onClose(); document.querySelector<HTMLButtonElement>('button[aria-label="Settings"]')?.focus(); }
  return <aside className="advanced-controls" aria-label="Advanced controls" onKeyDown={event => { if (event.key === 'Escape') close(); }}>
    <div className="drawer-heading"><h2>Advanced controls</h2><button className="icon-button" aria-label="Close advanced controls" onClick={close}>×</button></div><p className="drawer-description">Configure routing and generation settings.</p>
    <fieldset disabled={pending} className="control-group"><legend>Routing</legend><fieldset className="task-detection"><legend>Task detection</legend><label><input type="radio" name="detection" checked={!manual} onChange={() => onManual(false)} />Auto-detect <small>(recommended)</small></label><label><input type="radio" name="detection" checked={manual} onChange={() => onManual(true)} />Manual override</label></fieldset>
      <p className="field-help">Auto-detect uses a fixed QA category in this simulation. Production detection is not implemented.</p>
      {manual && <div className="field"><label htmlFor="category">Task category</label><select id="category" value={category} onChange={event => onCategory(event.target.value as Category)}>{categories.map(item => <option key={item}>{item}</option>)}</select></div>}
      <div className="field"><label htmlFor="threshold">Quality threshold <span className="small-label">Locked</span></label><output id="threshold" aria-describedby="threshold-help">0.80</output><p className="field-help" id="threshold-help">Minimum predicted acceptability required by the routing policy. This is not a correctness score.</p></div>
    </fieldset>
    <fieldset disabled={pending} className="control-group"><legend>Generation</legend><div className="field"><label htmlFor="tokens">Max output tokens</label><input id="tokens" type="number" min="1" max="512" step="1" value={tokens} onChange={event => onTokens(event.target.value)} aria-invalid={!validGeneration(tokens, '1')} aria-describedby="tokens-help" /><p className="field-help" id="tokens-help">Integer from 1 to 512.</p></div><div className="field"><label htmlFor="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="any" value={temperature} onChange={event => onTemperature(event.target.value)} aria-invalid={!validGeneration('256', temperature)} aria-describedby="temperature-help" /><p className="field-help" id="temperature-help">From 0 to 2. Default 1.0.</p></div></fieldset>
    <section className="control-group"><h3>Validation</h3><div className="locked-setting"><span>Response validation</span><span>Off</span></div><p className="field-help">Validation controls arrive in Phase 12.2D.</p></section><p className="drawer-footnote">Local demonstration values only. The production routing policy is unchanged.</p>
  </aside>;
}
