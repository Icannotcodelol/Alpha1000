import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const allowedHosts = process.env.VITE_ALLOWED_HOSTS
  ? process.env.VITE_ALLOWED_HOSTS.split(',').map((host) => host.trim())
  : [];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    allowedHosts: ['alpha1000game.loca.lt', 'alpha1000api.loca.lt'],
  },
  preview: {
    host: true,
    allowedHosts,
  },
});
