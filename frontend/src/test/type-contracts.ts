import type { ChatRequest } from '../api/types';

export const manualContractRequest: ChatRequest = {
  prompt: 'Return YES or NO.',
  routing_mode: 'manual',
  category: 'classification',
  validation: { format: 'label', allowed_labels: ['YES', 'NO'] },
};

export const invalidAutoContractRequest: ChatRequest = {
  prompt: 'Return YES or NO.',
  routing_mode: 'auto',
  // @ts-expect-error AUTO requests must remain unable to carry validation.
  validation: { format: 'label', allowed_labels: ['YES', 'NO'] },
};
