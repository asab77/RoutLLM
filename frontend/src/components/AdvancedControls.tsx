import { categories, jsonFieldTypes } from '../api/types';
import type { Category, JsonFieldType, ValidationContract } from '../api/types';
export function validGeneration(tokens: string, temperature: string) {
  return tokens.trim() !== '' && Number.isInteger(Number(tokens)) && Number(tokens) >= 1 && Number(tokens) <= 512 && temperature.trim() !== '' && Number.isFinite(Number(temperature)) && Number(temperature) >= 0 && Number(temperature) <= 2;
}

export interface ContractField { id: number; name: string; type: JsonFieldType | '' }
export const contractCategories = ['classification', 'extraction', 'structured_json'] as const;
export function supportsOutputContract(category: Category | ''): category is typeof contractCategories[number] {
  return contractCategories.includes(category as typeof contractCategories[number]);
}

export interface ContractResult { contract?: ValidationContract; error?: string }
export function buildOutputContract(
  enabled: boolean,
  manual: boolean,
  category: Category | '',
  labelText: string,
  fields: ContractField[],
): ContractResult {
  if (!enabled) return {};
  if (!manual || !supportsOutputContract(category)) return { error: 'Output contracts require a supported Manual category.' };
  if (category === 'classification') {
    const lines = labelText.split(/\r?\n/);
    while (lines.length && !lines[0].trim()) lines.shift();
    while (lines.length && !lines.at(-1)?.trim()) lines.pop();
    const labels = lines.map(label => label.trim());
    if (!labels.length) return { error: 'Add at least one allowed label.' };
    if (labels.some(label => !label)) return { error: 'Labels cannot be blank.' };
    if (labels.length > 64) return { error: 'Use no more than 64 labels.' };
    if (labels.some(label => label.length > 128)) return { error: 'Labels must be 128 characters or fewer.' };
    if (new Set(labels).size !== labels.length) return { error: 'Labels must be unique.' };
    return { contract: { format: 'label', allowed_labels: labels } };
  }
  if (!fields.length) return { error: 'Add at least one required field.' };
  if (fields.length > 64) return { error: 'Use no more than 64 fields.' };
  const names = fields.map(field => field.name.trim());
  if (names.some(name => !name)) return { error: 'Field names cannot be blank.' };
  if (names.some(name => name.length > 128)) return { error: 'Field names must be 128 characters or fewer.' };
  if (new Set(names).size !== names.length) return { error: 'Field names must be unique.' };
  const fieldTypes = Object.fromEntries(fields.flatMap((field, index) =>
    field.type ? [[names[index], field.type]] : []
  ));
  return {
    contract: {
      format: 'json', root_type: 'object', required_fields: names,
      ...(Object.keys(fieldTypes).length ? { field_types: fieldTypes } : {}),
    },
  };
}

interface Props { onClose: () => void; manual: boolean; onManual: (value: boolean) => void; category: Category | ''; onCategory: (value: Category | '') => void; tokens: string; onTokens: (value: string) => void; temperature: string; onTemperature: (value: string) => void; pending: boolean; contractEnabled: boolean; onContractEnabled: (value: boolean) => void; labelText: string; onLabelText: (value: string) => void; contractFields: ContractField[]; onContractField: (id: number, change: Partial<Omit<ContractField, 'id'>>) => void; onAddContractField: () => void; onRemoveContractField: (id: number) => void; contractError?: string }
export function AdvancedControls({ onClose, manual, onManual, category, onCategory, tokens, onTokens, temperature, onTemperature, pending, contractEnabled, onContractEnabled, labelText, onLabelText, contractFields, onContractField, onAddContractField, onRemoveContractField, contractError }: Props) {
  function close() { onClose(); document.querySelector<HTMLButtonElement>('button[aria-label="Settings"]')?.focus(); }
  const supported = manual && supportsOutputContract(category);
  return <aside className="advanced-controls" aria-label="Advanced controls" onKeyDown={event => { if (event.key === 'Escape') close(); }}>
    <div className="drawer-heading"><h2>Advanced controls</h2><button className="icon-button" aria-label="Close advanced controls" onClick={close}>×</button></div><p className="drawer-description">Configure routing and generation settings.</p>
    <fieldset disabled={pending} className="control-group"><legend>Routing</legend><fieldset className="task-detection"><legend>Routing mode</legend><label><input type="radio" name="detection" checked={!manual} onChange={() => onManual(false)} />Auto <small>(recommended)</small></label><label><input type="radio" name="detection" checked={manual} onChange={() => onManual(true)} />Manual category</label></fieldset>
      <p className="field-help">Auto uses the configured direct model. Manual routes using the category you select.</p>
      {manual && <div className="field"><label htmlFor="category">Task category</label><select id="category" value={category} required onChange={event => onCategory(event.target.value as Category | '')}><option value="" disabled>Select a category</option>{categories.map(item => <option key={item}>{item}</option>)}</select></div>}
      <div className="field"><label htmlFor="threshold">Quality threshold <span className="small-label">Locked</span></label><output id="threshold" aria-describedby="threshold-help">0.80</output><p className="field-help" id="threshold-help">Minimum predicted acceptability required by the routing policy. This is not a correctness score.</p></div>
    </fieldset>
    <fieldset disabled={pending} className="control-group"><legend>Generation</legend><div className="field"><label htmlFor="tokens">Max output tokens</label><input id="tokens" type="number" min="1" max="512" step="1" value={tokens} onChange={event => onTokens(event.target.value)} aria-invalid={!validGeneration(tokens, '1')} aria-describedby="tokens-help" /><p className="field-help" id="tokens-help">Integer from 1 to 512.</p></div><div className="field"><label htmlFor="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="any" value={temperature} onChange={event => onTemperature(event.target.value)} aria-invalid={!validGeneration('256', temperature)} aria-describedby="temperature-help" /><p className="field-help" id="temperature-help">From 0 to 2. Default 1.0.</p></div></fieldset>
    <fieldset disabled={pending} className="control-group output-contract"><legend>Output contract &amp; escalation</legend>
      <p className="field-help">Check explicit output structure and allow bounded model escalation when the contract is not satisfied. This does not judge answer correctness.</p>
      {!manual ? <div className="locked-setting"><span>Output contract</span><span>Manual routing required</span></div>
        : !category ? <div className="locked-setting"><span>Output contract</span><span>Select a category</span></div>
        : !supported ? <><div className="locked-setting"><span>Output contract</span><span>Unavailable for {category}</span></div><p className="field-help">The deterministic validator does not evaluate semantic quality for this category.</p></>
        : <>
          <label className="contract-toggle"><input type="checkbox" checked={contractEnabled} onChange={event => onContractEnabled(event.target.checked)} /> Enable output contract &amp; escalation</label>
          {contractEnabled && category === 'classification' && <div className="field"><label htmlFor="allowed-labels">Allowed labels</label><textarea id="allowed-labels" rows={4} value={labelText} onChange={event => onLabelText(event.target.value)} aria-invalid={Boolean(contractError)} aria-describedby={contractError ? 'contract-help contract-error' : 'contract-help'} placeholder={'APPROVE\nREVIEW\nREJECT'} /><p className="field-help" id="contract-help">One exact label per line. Ask for one of the same labels in your prompt.</p></div>}
          {contractEnabled && category !== 'classification' && <div className="contract-fields"><p className="field-help" id="contract-help">Add required top-level object fields. Types are optional; ask for the same JSON shape in your prompt.</p>{contractFields.map((field, index) => <div className="contract-field" key={field.id}><div className="field"><label htmlFor={`contract-name-${field.id}`}>Field {index + 1} name</label><input id={`contract-name-${field.id}`} value={field.name} maxLength={129} onChange={event => onContractField(field.id, { name: event.target.value })} aria-invalid={Boolean(contractError)} aria-describedby={contractError ? 'contract-help contract-error' : 'contract-help'} /></div><div className="field"><label htmlFor={`contract-type-${field.id}`}>Field {index + 1} type</label><select id={`contract-type-${field.id}`} value={field.type} onChange={event => onContractField(field.id, { type: event.target.value as JsonFieldType | '' })}><option value="">Any type</option>{jsonFieldTypes.map(type => <option key={type}>{type}</option>)}</select></div><button type="button" className="text-button remove-field" onClick={() => onRemoveContractField(field.id)} aria-label={`Remove field ${index + 1}`}>Remove</button></div>)}<button type="button" className="text-button add-field" disabled={contractFields.length >= 64} onClick={onAddContractField}>＋ Add field</button></div>}
          {contractEnabled && contractError && <p className="contract-error" id="contract-error" role="alert">{contractError}</p>}
        </>}
    </fieldset><p className="drawer-footnote">The server owns model configuration and routing policy.</p>
  </aside>;
}
