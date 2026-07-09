import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const baseUrl = process.env.FACE_VERIFY_BASE_URL || 'http://127.0.0.1:5173';
let serverProcess = null;

try {
  await ensureServer();
  const report = await verifyFaceTrackingFilters();
  const status = report.failures.length ? 'FAIL' : 'PASS';
  console.log(
    `${status} face: shortCache=${report.dropout.keptAfterShortMiss}, longClear=${report.dropout.clearedAfterLongMiss}, ` +
      `jitter=${report.filter.jitterMove.toFixed(4)}, jump=${report.filter.jumpStep.toFixed(4)}/${report.filter.jumpTargetDistance.toFixed(4)}, ` +
      `pose=${report.pose.scaleStep.toFixed(3)}/${report.pose.rollStep.toFixed(3)}/${report.pose.yawStep.toFixed(3)}, ` +
      `speed=${report.filter.speed.toFixed(3)}, eye=${report.expression.squintIgnition.toFixed(3)}/${report.expression.tiredIgnition.toFixed(3)}, ` +
      `mounted=${report.mounted.active.face[3].toFixed(3)}, lit=${report.lit}`,
  );
  for (const failure of report.failures) {
    console.error(`  - ${failure}`);
  }
  if (report.failures.length) {
    process.exitCode = 1;
  }
} finally {
  if (serverProcess) {
    serverProcess.kill('SIGTERM');
  }
}

async function verifyFaceTrackingFilters() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 960, height: 540 }, deviceScaleFactor: 1 });
  const failures = [];

  page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error' && !/XNNPACK|TFLite|GL Driver|WebGL/.test(text)) {
      failures.push(`console: ${text}`);
    }
  });

  await page.addInitScript(() => {
    localStorage.removeItem('particle-demo-particle-settings');
    localStorage.removeItem('particle-demo-face-calibration');
  });

  await page.goto(`${baseUrl}/?mode=face&verify=1&syntheticHand=none`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  await page.waitForFunction(() => window.__particleDemoVerify, null, { timeout: 30_000 });

  const dropout = await page.evaluate(() => window.__particleDemoVerify.testFaceDropoutCache());
  const invalid = await page.evaluate(() => window.__particleDemoVerify.testFaceInvalidFrameGuard());
  const filter = await page.evaluate(() => window.__particleDemoVerify.testFaceFilterResponse());
  const pose = await page.evaluate(() => window.__particleDemoVerify.testFacePoseOutlierDamping());
  const expression = await page.evaluate(() => window.__particleDemoVerify.testFaceExpressionResponse());
  const mounted = await page.evaluate(() => window.__particleDemoVerify.testFaceStateApplication());
  const settings = await verifySettingsIsolation(page);
  await page.waitForTimeout(140);
  const screenshot = await page.screenshot({ type: 'png' });
  const lit = countLitSamples(screenshot);

  if (!dropout.keptAfterShortMiss) {
    failures.push('face center cache cleared on a short detection miss');
  }
  if (!dropout.clearedAfterLongMiss) {
    failures.push('face center cache did not clear after a long detection miss');
  }
  if (!dropout.shortVisible || dropout.shortStrength < 0.75) {
    failures.push(`short face miss should hold visible pose: ${JSON.stringify(dropout)}`);
  }
  if (dropout.longVisible) {
    failures.push(`long face miss should report invisible: ${JSON.stringify(dropout)}`);
  }
  if (invalid.usable) {
    failures.push(`invalid face frame was considered usable: ${JSON.stringify(invalid)}`);
  }
  if (!invalid.visible || !invalid.held || invalid.strength < 0.75) {
    failures.push(`invalid face frame should hold the previous pose: ${JSON.stringify(invalid)}`);
  }
  if (invalid.health.heldCount !== 1 || invalid.health.liveCount !== 0) {
    failures.push(`invalid face frame should update held health: ${JSON.stringify(invalid.health)}`);
  }
  if (!(filter.jitterMove > 0 && filter.jitterMove < 0.012)) {
    failures.push(`face jitter damping out of range: ${filter.jitterMove}`);
  }
  if (!(filter.jumpStep > 0.04 && filter.jumpStep < filter.jumpTargetDistance * 0.36)) {
    failures.push(`face jump response out of range: ${JSON.stringify(filter)}`);
  }
  if (pose.scaleStep > 0.075 || pose.rollStep > 0.13 || pose.yawStep > 0.19 || pose.eyeStep > 0.37) {
    failures.push(`face pose outlier not contained: ${JSON.stringify(pose)}`);
  }
  if (pose.scaleStep <= 0.005 || pose.rollStep <= 0.005 || pose.yawStep <= 0.005 || pose.eyeStep <= 0.005) {
    failures.push(`face pose outlier damping stalled entirely: ${JSON.stringify(pose)}`);
  }
  if (expression.neutralIgnition > 0.02) {
    failures.push(`neutral eyes should not ignite: ${JSON.stringify(expression)}`);
  }
  if (expression.tiredIgnition > 0.08) {
    failures.push(`tired eyes should stay below ignition: ${JSON.stringify(expression)}`);
  }
  if (expression.squintIgnition < 0.45) {
    failures.push(`intentional squint should ignite eyes: ${JSON.stringify(expression)}`);
  }
  if (expression.blinkIgnition < 0.75) {
    failures.push(`blink should strongly ignite eyes: ${JSON.stringify(expression)}`);
  }
  if (mounted.active.mode !== 1) {
    failures.push(`synthetic face did not switch shader to face mode: ${JSON.stringify(mounted.active)}`);
  }
  if (mounted.active.face[3] < 0.65 || mounted.active.face[2] < 0.5) {
    failures.push(`synthetic face uniform not mounted strongly enough: ${JSON.stringify(mounted.active.face)}`);
  }
  if (Math.hypot(mounted.active.velocity[0], mounted.active.velocity[1]) < 0.4) {
    failures.push(`synthetic face velocity did not reach uniforms: ${JSON.stringify(mounted.active.velocity)}`);
  }
  if (mounted.active.expression[0] < 0.35 || mounted.active.expression[2] < 0.45 || mounted.active.expression[3] > -0.18) {
    failures.push(`synthetic face expression weak or wrong direction: ${JSON.stringify(mounted.active.expression)}`);
  }
  if (mounted.active.cameraMode !== '1 FACE') {
    failures.push(`synthetic face did not update camera status: ${mounted.active.cameraMode}`);
  }
  if (mounted.active.health.liveCount !== 1 || mounted.active.health.missCount !== 0) {
    failures.push(`synthetic face did not mark live health: ${JSON.stringify(mounted.active.health)}`);
  }
  if (mounted.faded.face[3] >= mounted.active.face[3] || mounted.faded.expression[2] >= mounted.active.expression[2]) {
    failures.push(`invisible face state did not fade uniforms: ${JSON.stringify(mounted.faded)}`);
  }
  if (mounted.faded.health.missCount < 1) {
    failures.push(`invisible face state did not record a miss: ${JSON.stringify(mounted.faded.health)}`);
  }
  if (lit < 500) {
    failures.push(`synthetic face render lit sample low ${lit}`);
  }
  failures.push(...settings.failures);

  await browser.close();
  return { failures, dropout, invalid, filter, pose, expression, mounted, settings, lit };
}

async function ensureServer() {
  if (await canReachServer()) {
    return;
  }

  const viteBin = path.join(rootDir, 'node_modules', '.bin', process.platform === 'win32' ? 'vite.cmd' : 'vite');
  if (!existsSync(viteBin)) {
    throw new Error('Vite binary not found. Run npm install first.');
  }

  serverProcess = spawn(viteBin, ['--host', '127.0.0.1'], {
    cwd: rootDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, BROWSER: 'none' },
  });

  serverProcess.stdout.on('data', (chunk) => process.stdout.write(`[vite] ${chunk}`));
  serverProcess.stderr.on('data', (chunk) => process.stderr.write(`[vite] ${chunk}`));

  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (await canReachServer()) {
      return;
    }
    await delay(250);
  }

  throw new Error(`Timed out waiting for ${baseUrl}`);
}

async function canReachServer() {
  try {
    const response = await fetch(baseUrl, { method: 'GET' });
    return response.ok;
  } catch {
    return false;
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function verifySettingsIsolation(page) {
  const failures = [];
  const state = await page.evaluate(() => {
    const readCalibration = () => ({
      mode: document.querySelector('#app')?.dataset.mode,
      hidden: document.querySelector('#calibrationPanel')?.hidden,
      pressed: document.querySelector('#calibrationToggle')?.getAttribute('aria-pressed') || '',
      ariaDisabled: document.querySelector('#calibrationToggle')?.getAttribute('aria-disabled') || '',
      disabledInputs: Array.from(document.querySelectorAll('#calibrationControls input')).map((input) => input.disabled),
    });
    const sectionState = () => {
      const readSection = (name) => {
        const section = document.querySelector(`.settings-section[data-section="${name}"]`);
        return {
          inactive: section?.classList.contains('is-inactive') || false,
          ariaDisabled: section?.getAttribute('aria-disabled') || '',
          disabledInputs: Array.from(section?.querySelectorAll('input') || []).map((input) => input.disabled),
        };
      };
      return {
        mode: document.querySelector('#app')?.dataset.mode,
        hand: readSection('hand'),
        face: readSection('face'),
        shared: readSection('shared'),
      };
    };

    const faceOpacityInput = document.querySelector('#setting-faceCameraOpacity');
    const handSmoothingInput = document.querySelector('#setting-handSmoothing');
    const initialFaceMode = sectionState();
    const initialCalibration = readCalibration();

    document.querySelector('#calibrationToggle').click();
    const calibrationOpen = readCalibration();

    faceOpacityInput.value = '0.12';
    faceOpacityInput.dispatchEvent(new Event('input', { bubbles: true }));
    const lowOpacity = getComputedStyle(document.querySelector('#app')).getPropertyValue('--face-camera-opacity').trim();
    faceOpacityInput.value = '0.68';
    faceOpacityInput.dispatchEvent(new Event('input', { bubbles: true }));
    const highOpacity = getComputedStyle(document.querySelector('#app')).getPropertyValue('--face-camera-opacity').trim();

    document.querySelector('#handModeButton').click();
    const handMode = sectionState();
    const handModeCalibration = readCalibration();
    const handInputDisabledAfterSwitch = handSmoothingInput.disabled;
    const faceInputDisabledAfterSwitch = faceOpacityInput.disabled;

    document.querySelector('#calibrationToggle').click();
    const calibrationReopenedFromHand = readCalibration();

    document.querySelector('#faceModeButton').click();
    const faceModeAgain = sectionState();

    return {
      initialFaceMode,
      initialCalibration,
      calibrationOpen,
      lowOpacity,
      highOpacity,
      handMode,
      handModeCalibration,
      calibrationReopenedFromHand,
      faceModeAgain,
      handInputDisabledAfterSwitch,
      faceInputDisabledAfterSwitch,
    };
  });

  const all = (values, expected) => values.length > 0 && values.every((value) => value === expected);
  if (state.initialFaceMode.mode !== 'face') {
    failures.push(`expected initial face mode, got ${state.initialFaceMode.mode}`);
  }
  if (!state.initialCalibration.hidden || state.initialCalibration.pressed !== 'false' || state.initialCalibration.ariaDisabled !== 'false') {
    failures.push(`face tune should start closed but available in face mode: ${JSON.stringify(state.initialCalibration)}`);
  }
  if (!all(state.initialCalibration.disabledInputs, true)) {
    failures.push(`closed tune inputs should be disabled: ${JSON.stringify(state.initialCalibration.disabledInputs)}`);
  }
  if (state.calibrationOpen.hidden || state.calibrationOpen.pressed !== 'true' || state.calibrationOpen.ariaDisabled !== 'false') {
    failures.push(`face tune did not open in face mode: ${JSON.stringify(state.calibrationOpen)}`);
  }
  if (!all(state.calibrationOpen.disabledInputs, false)) {
    failures.push(`open tune inputs should be enabled: ${JSON.stringify(state.calibrationOpen.disabledInputs)}`);
  }
  if (!state.initialFaceMode.hand.inactive || state.initialFaceMode.hand.ariaDisabled !== 'true') {
    failures.push(`hand settings should be inactive in face mode: ${JSON.stringify(state.initialFaceMode.hand)}`);
  }
  if (!all(state.initialFaceMode.hand.disabledInputs, true)) {
    failures.push(`hand inputs should be disabled in face mode: ${JSON.stringify(state.initialFaceMode.hand.disabledInputs)}`);
  }
  if (state.initialFaceMode.face.inactive || state.initialFaceMode.face.ariaDisabled !== 'false') {
    failures.push(`face settings should be active in face mode: ${JSON.stringify(state.initialFaceMode.face)}`);
  }
  if (!all(state.initialFaceMode.face.disabledInputs, false)) {
    failures.push(`face inputs should be enabled in face mode: ${JSON.stringify(state.initialFaceMode.face.disabledInputs)}`);
  }
  if (state.initialFaceMode.shared.inactive || state.initialFaceMode.shared.ariaDisabled !== 'false') {
    failures.push(`shared settings should stay active: ${JSON.stringify(state.initialFaceMode.shared)}`);
  }
  if (Number(state.lowOpacity) !== 0.12 || Number(state.highOpacity) !== 0.68) {
    failures.push(`camera opacity CSS var did not update live: ${state.lowOpacity} -> ${state.highOpacity}`);
  }
  if (state.handMode.mode !== 'hand') {
    failures.push(`mode switch to hand failed: ${state.handMode.mode}`);
  }
  if (!state.handModeCalibration.hidden || state.handModeCalibration.pressed !== 'false' || state.handModeCalibration.ariaDisabled !== 'true') {
    failures.push(`face tune should close and disable in hand mode: ${JSON.stringify(state.handModeCalibration)}`);
  }
  if (!all(state.handModeCalibration.disabledInputs, true)) {
    failures.push(`hand mode should disable tune inputs: ${JSON.stringify(state.handModeCalibration.disabledInputs)}`);
  }
  if (state.handMode.face.inactive !== true || state.handMode.face.ariaDisabled !== 'true' || !all(state.handMode.face.disabledInputs, true)) {
    failures.push(`face settings should be disabled in hand mode: ${JSON.stringify(state.handMode.face)}`);
  }
  if (state.handMode.hand.inactive !== false || state.handMode.hand.ariaDisabled !== 'false' || !all(state.handMode.hand.disabledInputs, false)) {
    failures.push(`hand settings should be enabled in hand mode: ${JSON.stringify(state.handMode.hand)}`);
  }
  if (state.handInputDisabledAfterSwitch || !state.faceInputDisabledAfterSwitch) {
    failures.push(`specific inputs had wrong disabled state after hand switch: ${JSON.stringify(state)}`);
  }
  if (state.calibrationReopenedFromHand.mode !== 'face' || state.calibrationReopenedFromHand.hidden || state.calibrationReopenedFromHand.pressed !== 'true') {
    failures.push(`tune button should switch back to face and open: ${JSON.stringify(state.calibrationReopenedFromHand)}`);
  }
  if (!all(state.calibrationReopenedFromHand.disabledInputs, false)) {
    failures.push(`reopened tune inputs should be enabled: ${JSON.stringify(state.calibrationReopenedFromHand.disabledInputs)}`);
  }
  if (state.faceModeAgain.mode !== 'face' || state.faceModeAgain.hand.inactive !== true || state.faceModeAgain.face.inactive !== false) {
    failures.push(`switching back to face did not restore settings sections: ${JSON.stringify(state.faceModeAgain)}`);
  }

  return { failures, state };
}

function countLitSamples(buffer) {
  let lit = 0;
  for (let index = 33; index < buffer.length - 4; index += 17) {
    if (buffer[index] + buffer[index + 1] + buffer[index + 2] > 45) {
      lit += 1;
    }
  }
  return lit;
}
