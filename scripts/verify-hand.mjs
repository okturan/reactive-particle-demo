import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startVerificationServer } from './verification-server.mjs';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let baseUrl = '';
const profiles = [
  {
    name: 'open',
    waitMs: 4500,
    expect(state) {
      const failures = [];
      if (state.forceCount > 2) failures.push(`open palm should not upload many fingertip forces: ${state.forceCount}`);
      if (state.forceEnergy > 0.75) failures.push(`open palm fingertip force too high ${state.forceEnergy}`);
      if (state.palm[2] <= 0.75) failures.push(`open palm strength low ${state.palm[2]}`);
      if (state.palm[2] > 2.2) failures.push(`open palm strength too high ${state.palm[2]}`);
      if (state.gustTriggerCount !== 0) failures.push(`open palm triggered ${state.gustTriggerCount} gusts`);
      if (!state.gestureText.includes('PALM')) failures.push(`open gesture not PALM: ${state.gestureText}`);
      return failures;
    },
  },
  {
    name: 'pinch',
    waitMs: 4500,
    expect(state) {
      const failures = [];
      if (state.pinchEnergy <= 0.18) failures.push(`pinchEnergy low ${state.pinchEnergy}`);
      if (!state.gestureText.includes('PINCH')) failures.push(`pinch gesture not PINCH: ${state.gestureText}`);
      return failures;
    },
  },
  {
    name: 'point',
    waitMs: 4500,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`point hand text bad: ${state.handText}`);
      if (state.forceCount !== 1) failures.push(`point compact forceCount expected 1, got ${state.forceCount}`);
      if (state.activeForceSlots !== 1) failures.push(`point active force slots expected 1, got ${state.activeForceSlots}`);
      if (state.forceSources?.[0] !== 1) failures.push(`point source should be index finger slot 1, got ${state.forceSources}`);
      if (sampleSummary.maxPalm > 0.2) failures.push(`point palm strength too high: ${sampleSummary.maxPalm}`);
      if (sampleSummary.maxPinch > 0.18) failures.push(`point pinch too high: ${sampleSummary.maxPinch}`);
      if (sampleSummary.maxForceEnergy > 1.15) failures.push(`point forceEnergy too high: ${sampleSummary.maxForceEnergy}`);
      if (state.gestureText.includes('PALM') || state.gestureText.includes('PINCH')) {
        failures.push(`point gesture should stay local: ${state.gestureText}`);
      }
      return failures;
    },
  },
  {
    name: 'glitch',
    waitMs: 1350,
    sampleCount: 42,
    sampleEveryMs: 50,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`glitch hand text bad: ${state.handText}`);
      if (sampleSummary.unstableFingerSamples < 1) failures.push('glitch did not expose an unstable fingertip frame');
      if (sampleSummary.maxForcePositionStep > 1.1) {
        failures.push(`glitch force jumped too far: ${sampleSummary.maxForcePositionStep}`);
      }
      if (sampleSummary.maxLiveForceDuringUnstable > 2) {
        failures.push(`glitch activated too many live forces during unstable fingertip: ${sampleSummary.maxLiveForceDuringUnstable}`);
      }
      return failures;
    },
  },
  {
    name: 'two',
    waitMs: 4700,
    sampleCount: 12,
    sampleEveryMs: 140,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('2 HAND')) failures.push(`two-hand text bad: ${state.handText}`);
      if (sampleSummary.shockActiveCount < 1) failures.push(`shock not active in sample window: ${state.shockAge}`);
      return failures;
    },
  },
  {
    name: 'cross',
    waitMs: 500,
    sampleCount: 44,
    sampleEveryMs: 120,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('2 HAND')) failures.push(`cross hand text bad: ${state.handText}`);
      if (sampleSummary.slotIdentitySamples < 8) {
        failures.push(`cross had too few two-slot identity samples: ${sampleSummary.slotIdentitySamples}`);
      }
      if (sampleSummary.slotIdentityMismatchCount > 0) {
        failures.push(`cross slot identity flipped ${sampleSummary.slotIdentityMismatchCount} times`);
      }
      return failures;
    },
  },
  {
    name: 'drop',
    waitMs: 5200,
    expect(state) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`drop hand text bad: ${state.handText}`);
      if (state.forceCount > 2) failures.push(`drop retained stale fingertip forces: ${state.forceCount}`);
      if (state.forceCount !== state.activeForceSlots) {
        failures.push(`drop force loop not compacted: ${state.forceCount}/${state.activeForceSlots}`);
      }
      if (state.forceEnergy > 2.4) failures.push(`drop forceEnergy too high: ${state.forceEnergy}`);
      if (!state.gestureText.includes('PALM')) failures.push(`drop gesture not PALM: ${state.gestureText}`);
      return failures;
    },
  },
  {
    name: 'dropout',
    waitMs: 1650,
    sampleCount: 46,
    sampleEveryMs: 50,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`dropout did not recover to one hand: ${state.handText}`);
      if (state.handSlotResetCount !== 0) failures.push(`dropout reset hand slots: ${state.handSlotResetCount}`);
      if (sampleSummary.maxPalm <= 0.25) failures.push(`dropout palm strength low: ${sampleSummary.maxPalm}`);
      if (sampleSummary.heldHandSamples < 1) failures.push('dropout did not hold cached palm during zero-hand frames');
      if (sampleSummary.maxHeldHandCount < 1) failures.push(`dropout held too few hands: ${sampleSummary.maxHeldHandCount}`);
      if (sampleSummary.zeroHandSamples > 0) failures.push(`dropout surfaced ${sampleSummary.zeroHandSamples} zero-hand UI samples`);
      return failures;
    },
  },
  {
    name: 'stall',
    waitMs: 1800,
    sampleCount: 20,
    sampleEveryMs: 50,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`stall did not recover to one hand: ${state.handText}`);
      if (state.handSlotResetCount !== 0) failures.push(`stall reset hand slots: ${state.handSlotResetCount}`);
      if (sampleSummary.zeroHandSamples > 0) failures.push(`stall surfaced ${sampleSummary.zeroHandSamples} zero-hand UI samples`);
      if (sampleSummary.heldHandSamples < 3) failures.push(`stall held too briefly: ${sampleSummary.heldHandSamples}`);
      if (sampleSummary.maxHeldHandCount < 1) failures.push(`stall held too few hands: ${sampleSummary.maxHeldHandCount}`);
      if (sampleSummary.maxPalm <= 0.18) failures.push(`stall palm faded too far: ${sampleSummary.maxPalm}`);
      return failures;
    },
  },
  {
    name: 'flicker',
    waitMs: 700,
    sampleCount: 24,
    sampleEveryMs: 120,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`flicker hand text bad: ${state.handText}`);
      if (sampleSummary.maxForceEnergy > 0.16) failures.push(`flicker forceEnergy too high: ${sampleSummary.maxForceEnergy}`);
      if (sampleSummary.maxPalm > 0.18) failures.push(`flicker palm strength too high: ${sampleSummary.maxPalm}`);
      if (sampleSummary.maxPinch > 0.18) failures.push(`flicker pinch too high: ${sampleSummary.maxPinch}`);
      return failures;
    },
  },
  {
    name: 'weak',
    waitMs: 1300,
    sampleCount: 12,
    sampleEveryMs: 80,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('0 HAND')) failures.push(`weak hand should be rejected: ${state.handText}`);
      if (state.forceCount !== 0 || state.activeForceSlots !== 0) {
        failures.push(`weak hand created fingertip forces: ${state.forceCount}/${state.activeForceSlots}`);
      }
      if (sampleSummary.maxPalm > 0.12) failures.push(`weak hand created palm field: ${sampleSummary.maxPalm}`);
      if (sampleSummary.maxPinch > 0.12) failures.push(`weak hand created pinch: ${sampleSummary.maxPinch}`);
      return failures;
    },
  },
  {
    name: 'corrupt',
    waitMs: 3600,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('0 HAND')) failures.push(`corrupt frame should be rejected: ${state.handText}`);
      if (state.invalidHandFrameCount < 1) failures.push(`corrupt frame was not counted: ${state.invalidHandFrameCount}`);
      if (state.activeForceSlots !== 0) failures.push(`corrupt frame left active forces: ${state.activeForceSlots}`);
      if (sampleSummary.maxPalm > 0.2) failures.push(`corrupt palm strength too high: ${sampleSummary.maxPalm}`);
      if (sampleSummary.maxPinch > 0.18) failures.push(`corrupt pinch too high: ${sampleSummary.maxPinch}`);
      return failures;
    },
  },
  {
    name: 'sweep',
    waitMs: 700,
    sampleCount: 28,
    sampleEveryMs: 120,
    expect(state, sampleSummary) {
      const failures = [];
      if (!state.handText.includes('1 HAND')) failures.push(`sweep hand text bad: ${state.handText}`);
      if (sampleSummary.maxPalm <= 0.25) failures.push(`sweep palm strength low: ${sampleSummary.maxPalm}`);
      if (sampleSummary.maxPalmSpeed <= state.gustSpeedThreshold) {
        failures.push(`sweep palm speed low: ${sampleSummary.maxPalmSpeed}/${state.gustSpeedThreshold}`);
      }
      if (state.gustTriggerCount < 1) failures.push('sweep gust trigger count stayed at zero');
      if (!sampleSummary.observedGust) failures.push('sweep did not trigger a gust wake');
      return failures;
    },
  },
];
const requestedProfileNames = (process.env.HAND_VERIFY_PROFILES || '')
  .split(',')
  .map((name) => name.trim())
  .filter(Boolean);
const unknownProfileNames = requestedProfileNames.filter((name) => !profiles.some((profile) => profile.name === name));
if (unknownProfileNames.length) {
  throw new Error(`Unknown HAND_VERIFY_PROFILES: ${unknownProfileNames.join(', ')}`);
}
const selectedProfiles = requestedProfileNames.length
  ? profiles.filter((profile) => requestedProfileNames.includes(profile.name))
  : profiles;
const profileRepeatValue = process.env.HAND_VERIFY_REPEAT || '1';
const profileRepeatCount = Number(profileRepeatValue);
if (!/^\d+$/.test(profileRepeatValue) || !Number.isSafeInteger(profileRepeatCount) || profileRepeatCount < 1 || profileRepeatCount > 100) {
  throw new Error('HAND_VERIFY_REPEAT must be an integer from 1 to 100');
}

let verificationServer = null;

try {
  verificationServer = await startVerificationServer({
    rootDir,
    configuredBaseUrl: process.env.HAND_VERIFY_BASE_URL?.trim(),
    environmentVariable: 'HAND_VERIFY_BASE_URL',
  });
  baseUrl = verificationServer.baseUrl;
  const adaptiveReport = await verifyAdaptiveTrackingFps();
  const densityReport = await verifyParticleDensityControl();
  const pointerReport = await verifyPointerInputToggle();
  const liveGuardReport = await verifyLiveLoopGuards();
  const reports = [];
  for (const profile of selectedProfiles) {
    for (let run = 1; run <= profileRepeatCount; run += 1) {
      reports.push({ ...(await verifyProfile(profile)), run });
    }
  }

  const adaptiveStatus = adaptiveReport.failures.length ? 'FAIL' : 'PASS';
  console.log(
    `${adaptiveStatus} adaptive: normal=${adaptiveReport.normalFps}fps, ` +
      `renderPressure=${adaptiveReport.pressuredFps}fps, trackingPressure=${adaptiveReport.trackingPressuredFps}fps`,
  );
  for (const failure of adaptiveReport.failures) {
    console.error(`  - ${failure}`);
  }

  const densityStatus = densityReport.failures.length ? 'FAIL' : 'PASS';
  console.log(
    `${densityStatus} density: auto=${densityReport.auto.low.activeParticleCount}->${densityReport.auto.high.activeParticleCount}, ` +
      `high=${densityReport.high.low.activeParticleCount}->${densityReport.high.high.activeParticleCount}, ` +
      `visual=${densityReport.auto.low.litPixels}->${densityReport.auto.high.litPixels}`,
  );
  for (const failure of densityReport.failures) {
    console.error(`  - ${failure}`);
  }

  const pointerStatus = pointerReport.failures.length ? 'FAIL' : 'PASS';
  console.log(
    `${pointerStatus} pointer: off=${pointerReport.off.forceCount}/${pointerReport.off.activeForceSlots}, ` +
      `on=${pointerReport.on.forceCount}/${pointerReport.on.activeForceSlots}`,
  );
  for (const failure of pointerReport.failures) {
    console.error(`  - ${failure}`);
  }

  const liveGuardStatus = liveGuardReport.failures.length ? 'FAIL' : 'PASS';
  console.log(
    `${liveGuardStatus} live-guards: hand=${liveGuardReport.errorGuard.hand.cameraMode}/${liveGuardReport.errorGuard.hand.trackingPressure.toFixed(2)}, ` +
      `face=${liveGuardReport.errorGuard.face.cameraMode}/${liveGuardReport.errorGuard.face.trackingPressure.toFixed(2)}, ` +
      `reset=${liveGuardReport.streamReset.forceCount}/${liveGuardReport.streamReset.forceEnergy.toFixed(3)}, ` +
      `compact=${liveGuardReport.forceCompaction.forceCount}/${liveGuardReport.forceCompaction.activeForceSlots}`,
  );
  for (const failure of liveGuardReport.failures) {
    console.error(`  - ${failure}`);
  }

  for (const report of reports) {
    const status = report.failures.length ? 'FAIL' : 'PASS';
    const runLabel = profileRepeatCount > 1 ? `#${report.run}` : '';
    console.log(
      `${status} ${report.profile}${runLabel}: ${report.state.gestureText}, ${report.state.handText}, ` +
        `forces=${report.state.forceCount}/${report.state.activeForceSlots}, forceEnergy=${report.state.forceEnergy.toFixed(3)}, ` +
        `live=${report.state.liveForceCount}, unstable=${report.sampleSummary.maxUnstableFingerCount}, ` +
        `palm=${report.state.palm[2].toFixed(3)}/${report.sampleSummary.maxPalmSpeed.toFixed(3)}, pinch=${report.state.pinchEnergy.toFixed(3)}, ` +
        `shockAge=${report.state.shockAge.toFixed(3)}, maxForce=${report.sampleSummary.maxForceEnergy.toFixed(3)}, ` +
        `gust=${report.sampleSummary.gustActiveCount}, shock=${report.sampleSummary.shockActiveCount}, ` +
        `zero=${report.sampleSummary.zeroHandSamples}, held=${report.sampleSummary.maxHeldHandCount}/${report.sampleSummary.maxHeldForceCount}, ` +
        `invalid=${report.state.invalidHandFrameCount}, ` +
        `resets=${report.state.handSlotResetCount}, slots=${report.sampleSummary.slotIdentityMismatchCount}, ` +
        `debugReads=${report.state.debugLandmarkReadCount}, sources=${JSON.stringify(report.state.forceSources)}, lit=${report.lit}`,
    );
    for (const failure of report.failures) {
      console.error(`  - ${failure}`);
    }
  }

  if (
    adaptiveReport.failures.length ||
    densityReport.failures.length ||
    pointerReport.failures.length ||
    liveGuardReport.failures.length ||
    reports.some((report) => report.failures.length)
  ) {
    process.exitCode = 1;
  }
} finally {
  await verificationServer?.close();
}

async function verifyParticleDensityControl() {
  const browser = await chromium.launch({ headless: true });
  const failures = [];
  const auto = await readParticleDensityTransition(browser, 'auto', failures);
  const high = await readParticleDensityTransition(browser, 'high', failures);

  for (const report of [auto, high]) {
    const label = report.qualityMode;
    if (report.low.activeParticleCount !== 22_000 || report.low.particleBudget !== 22_000) {
      failures.push(`${label} density did not decrease immediately: ${JSON.stringify(report.low)}`);
    }
    if (report.high.activeParticleCount !== 86_000 || report.high.particleBudget !== 86_000) {
      failures.push(`${label} density did not increase immediately: ${JSON.stringify(report.high)}`);
    }
    if (report.low.drawRangeCount !== report.low.activeParticleCount) {
      failures.push(`${label} low density draw range is stale: ${JSON.stringify(report.low)}`);
    }
    if (report.high.drawRangeCount !== report.high.activeParticleCount) {
      failures.push(`${label} high density draw range is stale: ${JSON.stringify(report.high)}`);
    }
    if (report.high.litPixels < report.low.litPixels * 1.5) {
      failures.push(`${label} density lacks a visible render difference: ${JSON.stringify(report)}`);
    }
  }

  await browser.close();
  return { failures, auto, high };
}

async function readParticleDensityTransition(browser, qualityMode, failures) {
  const page = await browser.newPage({ viewport: { width: 960, height: 540 }, deviceScaleFactor: 1 });
  page.on('pageerror', (error) => failures.push(`density ${qualityMode} pageerror: ${error.message}`));
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error' && !/XNNPACK|TFLite|GL Driver|WebGL/.test(text)) {
      failures.push(`density ${qualityMode} console: ${text}`);
    }
  });
  await page.addInitScript(() => {
    localStorage.removeItem('particle-demo-particle-settings');
    localStorage.removeItem('particle-demo-face-calibration');
  });
  const qualityQuery = qualityMode === 'high' ? '&quality=high' : '';
  await page.goto(`${baseUrl}/?mode=hand&settings=1&verify=1&syntheticHand=none${qualityQuery}`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  await page.waitForFunction(() => window.__particleDemoVerify?.getState().drawRangeCount > 0, null, {
    timeout: 30_000,
  });

  const setDensity = async (value) => {
    const immediate = await page.evaluate((nextValue) => {
      const densityInput = document.querySelector('#setting-particleDensity');
      densityInput.value = String(nextValue);
      densityInput.dispatchEvent(new Event('input', { bubbles: true }));
      const state = window.__particleDemoVerify.getState();
      return {
        activeParticleCount: state.activeParticleCount,
        particleBudget: state.particleBudget,
        drawRangeCount: state.drawRangeCount,
        particleDensity: state.particleDensity,
      };
    }, value);
    await page.waitForTimeout(180);
    const visual = await page.evaluate(() => {
      const source = document.querySelector('#scene');
      const probe = document.createElement('canvas');
      probe.width = 240;
      probe.height = 150;
      const context = probe.getContext('2d', { willReadFrequently: true });
      context.drawImage(source, 0, 0, probe.width, probe.height);
      const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
      let litPixels = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        const luminance = pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722;
        if (luminance > 12) {
          litPixels += 1;
        }
      }
      return { litPixels };
    });
    return { ...immediate, ...visual };
  };

  const low = await setDensity(0.22);
  const high = await setDensity(0.86);
  const quality = await page.evaluate(() => window.__particleDemoVerify.getState().qualityMode);
  const report = { qualityMode: quality, low, high };

  await page.close();
  return report;
}

async function verifyAdaptiveTrackingFps() {
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
  const url = `${baseUrl}/?mode=hand&verify=1&syntheticHand=open`;
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForFunction(() => window.__particleDemoVerify && window.__particleDemoVerify.getState().handText.includes('HAND'));

  const normal = await page.evaluate(() => {
    window.__particleDemoVerify.setRenderPressure(0);
    window.__particleDemoVerify.setTrackingPressure(0);
    return window.__particleDemoVerify.getState();
  });
  const pressured = await page.evaluate(() => {
    window.__particleDemoVerify.setRenderPressure(1);
    window.__particleDemoVerify.setTrackingPressure(0);
    return window.__particleDemoVerify.getState();
  });
  const trackingPressured = await page.evaluate(() => {
    window.__particleDemoVerify.setRenderPressure(0);
    window.__particleDemoVerify.setTrackingPressure(1);
    return window.__particleDemoVerify.getState();
  });
  const duplicateGate = await page.evaluate(() => window.__particleDemoVerify.testVideoFrameGate(12.5, 12.5));
  const freshGate = await page.evaluate(() => window.__particleDemoVerify.testVideoFrameGate(12.5, 12.55));

  if (normal.effectiveTrackingFps !== 45) {
    failures.push(`normal effectiveTrackingFps expected 45, got ${normal.effectiveTrackingFps}`);
  }
  if (pressured.effectiveTrackingFps !== 30) {
    failures.push(`pressured effectiveTrackingFps expected 30, got ${pressured.effectiveTrackingFps}`);
  }
  if (trackingPressured.effectiveTrackingFps !== 30) {
    failures.push(`tracking pressure effectiveTrackingFps expected 30, got ${trackingPressured.effectiveTrackingFps}`);
  }
  if (!(pressured.effectiveTrackingFps < normal.effectiveTrackingFps)) {
    failures.push(`pressure did not reduce tracking FPS: ${normal.effectiveTrackingFps} -> ${pressured.effectiveTrackingFps}`);
  }
  if (!(trackingPressured.effectiveTrackingFps < normal.effectiveTrackingFps)) {
    failures.push(
      `tracking pressure did not reduce tracking FPS: ${normal.effectiveTrackingFps} -> ${trackingPressured.effectiveTrackingFps}`,
    );
  }
  if (normal.debugLandmarkReadCount !== 0) {
    failures.push(`hidden debug path read landmarks ${normal.debugLandmarkReadCount} times`);
  }
  if (!duplicateGate.duplicate || duplicateGate.duplicateFrameSkipCount !== 1) {
    failures.push(`duplicate video frame gate failed: ${JSON.stringify(duplicateGate)}`);
  }
  if (freshGate.duplicate || freshGate.duplicateFrameSkipCount !== 0) {
    failures.push(`fresh video frame gate failed: ${JSON.stringify(freshGate)}`);
  }

  await browser.close();
  return {
    failures,
    normalFps: normal.effectiveTrackingFps,
    pressuredFps: pressured.effectiveTrackingFps,
    trackingPressuredFps: trackingPressured.effectiveTrackingFps,
  };
}

async function verifyPointerInputToggle() {
  const browser = await chromium.launch({ headless: true });
  const failures = [];
  const off = await readPointerState(browser, false, failures);
  const on = await readPointerState(browser, true, failures);

  if (off.pointerEnabled) failures.push('pointer should be disabled by default');
  if (off.forceCount !== 0 || off.activeForceSlots !== 0 || off.forceEnergy > 0.04) {
    failures.push(`pointer disabled still created force: ${JSON.stringify(pickPointerState(off))}`);
  }
  if (!on.pointerEnabled) failures.push('pointer should be enabled with mouse=1');
  if (on.forceCount !== 1 || on.activeForceSlots !== 1 || on.forceEnergy < 0.18) {
    failures.push(`pointer enabled did not create force: ${JSON.stringify(pickPointerState(on))}`);
  }

  await browser.close();
  return { failures, off, on };
}

async function readPointerState(browser, enabled, failures) {
  const page = await browser.newPage({ viewport: { width: 960, height: 540 }, deviceScaleFactor: 1 });
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
  const url = `${baseUrl}/?mode=hand&verify=1&syntheticHand=none${enabled ? '&mouse=1' : ''}`;
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForFunction(() => window.__particleDemoVerify && window.__particleDemoVerify.getState().handText.includes('HAND'));
  await page.evaluate(() => {
    const canvas = document.querySelector('#scene');
    canvas.dispatchEvent(
      new PointerEvent('pointermove', {
        bubbles: true,
        clientX: Math.round(window.innerWidth * 0.5),
        clientY: Math.round(window.innerHeight * 0.5),
        pointerId: 1,
        pointerType: 'mouse',
      }),
    );
  });
  await page.waitForTimeout(420);
  const state = await page.evaluate(() => window.__particleDemoVerify.getState());
  await page.close();
  return state;
}

function pickPointerState(state) {
  return {
    pointerEnabled: state.pointerEnabled,
    forceCount: state.forceCount,
    activeForceSlots: state.activeForceSlots,
    forceEnergy: Number(state.forceEnergy.toFixed(3)),
    handText: state.handText,
    gestureText: state.gestureText,
  };
}

async function verifyLiveLoopGuards() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 960, height: 540 }, deviceScaleFactor: 1 });
  const failures = [];

  page.on('pageerror', (error) => failures.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error' && !/XNNPACK|TFLite|GL Driver|WebGL|verify tracking error|verify face tracking error/.test(text)) {
      failures.push(`console: ${text}`);
    }
  });

  await page.addInitScript(() => {
    localStorage.removeItem('particle-demo-particle-settings');
    localStorage.removeItem('particle-demo-face-calibration');
  });
  await page.goto(`${baseUrl}/?mode=hand&verify=1&syntheticHand=open`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  await page.waitForFunction(() => window.__particleDemoVerify && window.__particleDemoVerify.getState().handText.includes('HAND'));

  const errorGuard = await page.evaluate(() => window.__particleDemoVerify.testTrackingErrorGuard());
  const streamReset = await page.evaluate(() => window.__particleDemoVerify.testStreamResetClearsTracking());
  const forceCompaction = await page.evaluate(() => window.__particleDemoVerify.testForceCompaction());

  if (errorGuard.hand.cameraMode !== 'TRACK ERR') {
    failures.push(`hand tracking error did not surface TRACK ERR: ${JSON.stringify(errorGuard.hand)}`);
  }
  if (errorGuard.face.cameraMode !== 'TRACK ERR') {
    failures.push(`face tracking error did not surface TRACK ERR: ${JSON.stringify(errorGuard.face)}`);
  }
  if (errorGuard.hand.trackingPressure < 0.65 || errorGuard.face.trackingPressure < 0.65) {
    failures.push(`tracking error did not raise pressure: ${JSON.stringify(errorGuard)}`);
  }
  if (errorGuard.hand.health.missCount < 1 || errorGuard.face.health.missCount < 1) {
    failures.push(`tracking error did not update health misses: ${JSON.stringify(errorGuard)}`);
  }
  if (streamReset.hasPalmCache || streamReset.hasFaceCache) {
    failures.push(`stream reset left tracking caches: ${JSON.stringify(streamReset)}`);
  }
  if (streamReset.handLastSeen > -9999) {
    failures.push(`stream reset left hand lastSeen active: ${streamReset.handLastSeen}`);
  }
  if (streamReset.forceCount !== 0 || streamReset.forceEnergy > 0.001 || streamReset.palm[2] > 0.001 || streamReset.face[3] > 0.001) {
    failures.push(`stream reset left uniforms active: ${JSON.stringify(streamReset)}`);
  }
  if (streamReset.health.liveCount !== 0 || streamReset.health.missCount !== 0) {
    failures.push(`stream reset did not clear tracking health: ${JSON.stringify(streamReset.health)}`);
  }
  if (forceCompaction.forceCount !== 2 || forceCompaction.activeForceSlots !== 2) {
    failures.push(`fading force gaps were not compacted: ${JSON.stringify(forceCompaction)}`);
  }
  if (forceCompaction.compactedSources.join(',') !== '1,4' || forceCompaction.inactiveEnergy > 0.001) {
    failures.push(`fading force compaction corrupted slots: ${JSON.stringify(forceCompaction)}`);
  }

  await browser.close();
  return { failures, errorGuard, streamReset, forceCompaction };
}

async function verifyProfile(profile) {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
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

  const url = `${baseUrl}/?mode=hand&settings=1&debug=1&verify=1&syntheticHand=${profile.name}`;
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForFunction(
    () => window.__particleDemoVerify && window.__particleDemoVerify.getState().handText.includes('HAND'),
    null,
    { timeout: 30_000 },
  );
  await page.evaluate(() => window.__particleDemoVerify.resetHandTrackingForTest());
  await page.waitForTimeout(profile.waitMs);

  if (profile.name === 'drop') {
    await page
      .waitForFunction(() => {
        const state = window.__particleDemoVerify.getState();
        return state.handText.includes('1 HAND') && state.gestureText.includes('PALM');
      }, null, { timeout: 8_000 })
      .catch(() => null);
  }

  let observedGust = false;
  if (profile.name === 'sweep') {
    observedGust = await page
      .waitForFunction(() => {
        const { gustAge } = window.__particleDemoVerify.getState();
        return gustAge >= 0 && gustAge < 1.2;
      }, null, { timeout: 8_000 })
      .then(() => true)
      .catch(() => false);
  }

  const samples = [];
  const sampleCount = profile.sampleCount || 1;
  const sampleEveryMs = profile.sampleEveryMs || 0;
  for (let index = 0; index < sampleCount; index += 1) {
    if (index > 0 && sampleEveryMs > 0) {
      await page.waitForTimeout(sampleEveryMs);
    }
    samples.push(await page.evaluate(() => window.__particleDemoVerify.getState()));
  }

  const state = samples[samples.length - 1];
  const sampleSummary = summarizeSamples(samples);
  sampleSummary.observedGust = observedGust || sampleSummary.gustActiveCount > 0;
  const lit = await countLitCanvasPixels(page);

  if (state.mode !== 0) failures.push(`wrong mode ${state.mode}`);
  if (!state.handText.includes('HAND')) failures.push(`bad hand text ${state.handText}`);
  if (state.debugLandmarkReadCount < 1) failures.push('debug overlay did not read landmarks');
  if (lit < 500) failures.push(`lit sample low ${lit}`);
  failures.push(...profile.expect(state, sampleSummary));

  await browser.close();
  return { profile: profile.name, failures, state, sampleSummary, lit };
}

function summarizeSamples(samples) {
  let maxForceCount = 0;
  let maxActiveForceSlots = 0;
  let maxForceEnergy = 0;
  let maxForcePositionStep = 0;
  let maxPalm = 0;
  let maxPalmSpeed = 0;
  let maxPinch = 0;
  let unstableFingerSamples = 0;
  let maxUnstableFingerCount = 0;
  let maxLiveForceDuringUnstable = 0;
  let gustActiveCount = 0;
  let shockActiveCount = 0;
  let zeroHandSamples = 0;
  let heldForceSamples = 0;
  let maxHeldForceCount = 0;
  let heldHandSamples = 0;
  let maxHeldHandCount = 0;
  let slotIdentitySamples = 0;
  let slotIdentityMismatchCount = 0;
  let initialSlotIdentity = null;
  let previousPrimaryForce = null;
  for (const state of samples) {
    maxForceCount = Math.max(maxForceCount, state.forceCount);
    maxActiveForceSlots = Math.max(maxActiveForceSlots, state.activeForceSlots);
    maxForceEnergy = Math.max(maxForceEnergy, state.forceEnergy);
    const primaryForce = state.forces?.[0];
    if (primaryForce && primaryForce[2] > 0.02) {
      if (previousPrimaryForce) {
        maxForcePositionStep = Math.max(
          maxForcePositionStep,
          Math.hypot(primaryForce[0] - previousPrimaryForce[0], primaryForce[1] - previousPrimaryForce[1]),
        );
      }
      previousPrimaryForce = primaryForce;
    }
    maxPalm = Math.max(maxPalm, state.palm[2]);
    maxPalmSpeed = Math.max(maxPalmSpeed, state.palmSpeed || 0);
    maxPinch = Math.max(maxPinch, state.pinchEnergy);
    if ((state.unstableFingerCount || 0) > 0) {
      unstableFingerSamples += 1;
      maxLiveForceDuringUnstable = Math.max(maxLiveForceDuringUnstable, state.liveForceCount || 0);
    }
    maxUnstableFingerCount = Math.max(maxUnstableFingerCount, state.unstableFingerCount || 0);
    if (state.gustAge >= 0 && state.gustAge < 1.2) {
      gustActiveCount += 1;
    }
    if (state.shockAge >= 0 && state.shockAge < 1.85) {
      shockActiveCount += 1;
    }
    if (state.handText.includes('0 HAND')) {
      zeroHandSamples += 1;
    }
    if ((state.heldForceCount || 0) > 0) {
      heldForceSamples += 1;
    }
    maxHeldForceCount = Math.max(maxHeldForceCount, state.heldForceCount || 0);
    if ((state.trackingHealth?.heldCount || 0) > 0) {
      heldHandSamples += 1;
    }
    maxHeldHandCount = Math.max(maxHeldHandCount, state.trackingHealth?.heldCount || 0);
    const [slot0, slot1] = state.handSlots || [];
    const bothSlotsFresh = slot0?.lastSeenAge < 350 && slot1?.lastSeenAge < 350;
    if (bothSlotsFresh) {
      slotIdentitySamples += 1;
      const currentIdentity = `${slot0.handedness || '-'}:${slot1.handedness || '-'}`;
      if (!initialSlotIdentity) {
        initialSlotIdentity = currentIdentity;
      } else if (currentIdentity !== initialSlotIdentity) {
        slotIdentityMismatchCount += 1;
      }
    }
  }
  return {
    maxForceCount,
    maxActiveForceSlots,
    maxForceEnergy,
    maxForcePositionStep,
    maxPalm,
    maxPalmSpeed,
    maxPinch,
    unstableFingerSamples,
    maxUnstableFingerCount,
    maxLiveForceDuringUnstable,
    gustActiveCount,
    shockActiveCount,
    zeroHandSamples,
    heldForceSamples,
    maxHeldForceCount,
    heldHandSamples,
    maxHeldHandCount,
    slotIdentitySamples,
    slotIdentityMismatchCount,
  };
}

async function countLitCanvasPixels(page) {
  return page.evaluate(() => {
    const source = document.querySelector('#scene');
    const probe = document.createElement('canvas');
    probe.width = 240;
    probe.height = 150;
    const context = probe.getContext('2d', { willReadFrequently: true });
    context.drawImage(source, 0, 0, probe.width, probe.height);
    const pixels = context.getImageData(0, 0, probe.width, probe.height).data;
    let lit = 0;
    for (let index = 0; index < pixels.length; index += 4) {
      const luminance = pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722;
      if (luminance > 12) {
        lit += 1;
      }
    }
    return lit;
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
