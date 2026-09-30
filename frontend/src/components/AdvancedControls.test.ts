import { describe, expect, it } from 'vitest';
import { buildOutputContract } from './AdvancedControls';
import type { ContractField } from './AdvancedControls';

const field = (id: number, name: string, type: ContractField['type'] = ''): ContractField => ({ id, name, type });

describe('output contract construction', () => {
  it('omits disabled contracts and rejects unsupported modes and categories', () => {
    expect(buildOutputContract(false, true, 'classification', 'YES', [])).toEqual({});
    expect(buildOutputContract(true, false, 'classification', 'YES', [])).toHaveProperty('error');
    expect(buildOutputContract(true, true, 'qa', 'YES', [])).toHaveProperty('error');
  });

  it('trims labels and enforces blank, duplicate, count, and length bounds', () => {
    expect(buildOutputContract(true, true, 'classification', ' YES \nNO ', [])).toEqual({
      contract: { format: 'label', allowed_labels: ['YES', 'NO'] },
    });
    expect(buildOutputContract(true, true, 'classification', 'YES\n \nNO', [])).toHaveProperty('error', 'Labels cannot be blank.');
    expect(buildOutputContract(true, true, 'classification', 'YES\nYES', [])).toHaveProperty('error', 'Labels must be unique.');
    expect(buildOutputContract(true, true, 'classification', Array.from({ length: 65 }, (_, index) => `L${index}`).join('\n'), [])).toHaveProperty('error', 'Use no more than 64 labels.');
    expect(buildOutputContract(true, true, 'classification', 'x'.repeat(129), [])).toHaveProperty('error', 'Labels must be 128 characters or fewer.');
  });

  it('builds shallow JSON object contracts and omits unspecified field types', () => {
    expect(buildOutputContract(true, true, 'extraction', '', [
      field(1, ' name ', 'string'), field(2, 'amount'),
    ])).toEqual({ contract: {
      format: 'json', root_type: 'object', required_fields: ['name', 'amount'],
      field_types: { name: 'string' },
    } });
    expect(buildOutputContract(true, true, 'structured_json', '', [field(1, 'payload')])).toEqual({
      contract: { format: 'json', root_type: 'object', required_fields: ['payload'] },
    });
  });

  it('enforces required, unique, count, and name-length field constraints', () => {
    expect(buildOutputContract(true, true, 'extraction', '', [])).toHaveProperty('error', 'Add at least one required field.');
    expect(buildOutputContract(true, true, 'extraction', '', [field(1, ' ')])).toHaveProperty('error', 'Field names cannot be blank.');
    expect(buildOutputContract(true, true, 'extraction', '', [field(1, 'a'), field(2, ' a ')])).toHaveProperty('error', 'Field names must be unique.');
    expect(buildOutputContract(true, true, 'extraction', '', Array.from({ length: 65 }, (_, index) => field(index, `f${index}`)))).toHaveProperty('error', 'Use no more than 64 fields.');
    expect(buildOutputContract(true, true, 'extraction', '', [field(1, 'x'.repeat(129))])).toHaveProperty('error', 'Field names must be 128 characters or fewer.');
  });
});
