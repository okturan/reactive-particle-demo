import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

import { startVerificationServer } from './verification-server.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputPath = path.join(rootDir, 'docs', 'assets', 'reactive-particle-hand.png');
const server = await startVerificationServer({
  rootDir,
  configuredBaseUrl: process.env.SHOWCASE_BASE_URL,
  environmentVariable: 'SHOWCASE_BASE_URL',
});

let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  });

  await page.addInitScript(() => {
    localStorage.removeItem('particle-demo-particle-settings');
    localStorage.removeItem('particle-demo-face-calibration');
  });

  await page.goto(`${server.baseUrl}/?mode=hand&verify=1&syntheticHand=sweep`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  await page.waitForFunction(
    () => {
      const state = window.__particleDemoVerify?.getState?.();
      return Boolean(state?.handText.includes('1 HAND') && state?.gustAge >= 0 && state?.gustAge < 1.2);
    },
    null,
    { timeout: 30_000 },
  );
  await page.waitForTimeout(220);

  await mkdir(path.dirname(outputPath), { recursive: true });
  await page.screenshot({
    path: outputPath,
    fullPage: false,
    animations: 'disabled',
  });
  console.log(`Captured ${path.relative(rootDir, outputPath)}`);
} finally {
  await browser?.close();
  await server.close();
}
