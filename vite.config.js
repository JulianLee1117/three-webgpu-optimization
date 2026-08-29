import { defineConfig } from 'vite';

const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  resolve: {
    dedupe: ['three'],
  },
  server: {
    host: '127.0.0.1',
    headers: isolationHeaders,
  },
  preview: {
    host: '127.0.0.1',
    headers: isolationHeaders,
  },
});
