import { createServer as createPortProbe } from 'node:net';
import { createServer as createViteServer } from 'vite';

export async function startVerificationServer({ rootDir, configuredBaseUrl, environmentVariable }) {
  if (configuredBaseUrl) {
    const baseUrl = configuredBaseUrl.replace(/\/$/, '');
    const response = await fetch(baseUrl).catch(() => null);
    if (!response?.ok) {
      throw new Error(`${environmentVariable} is not reachable: ${baseUrl}`);
    }
    return { baseUrl, close: async () => {} };
  }

  const initialPort = await findAvailablePort();
  const server = await createViteServer({
    root: rootDir,
    logLevel: 'error',
    server: {
      hmr: false,
      host: '127.0.0.1',
      port: initialPort,
      strictPort: false,
      watch: null,
    },
  });

  try {
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') {
      throw new Error('Vite did not expose a local verification port');
    }
    return {
      baseUrl: `http://127.0.0.1:${address.port}`,
      close: () => server.close(),
    };
  } catch (error) {
    await server.close();
    throw error;
  }
}

async function findAvailablePort() {
  const probe = createPortProbe();
  await new Promise((resolve, reject) => {
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', resolve);
  });
  const address = probe.address();
  await new Promise((resolve, reject) => {
    probe.close((error) => (error ? reject(error) : resolve()));
  });
  if (!address || typeof address === 'string') {
    throw new Error('Could not reserve a local verification port');
  }
  return address.port;
}
