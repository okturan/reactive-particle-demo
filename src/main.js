import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FaceLandmarker, FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import logoUrl from '../logomark-dark.svg?url';

const PARTICLE_COUNT = 100_000;
const MIN_PARTICLE_COUNT = 18_000;
const DEFAULT_PARTICLE_COUNT = 54_000;
const MAX_FORCES = 10;
const TARGET_RENDER_FPS = 60;
const FPS_DISPLAY_CAP = 60;
const MIN_RENDER_FRAME_MS = 1000 / TARGET_RENDER_FPS;
const FRAME_SKIP_EPSILON_MS = 0.1;
const MIN_ADAPTIVE_TRACKING_FPS = 30;
const FORCE_LOOP_EPSILON = 0.02;
const TRACKING_MODES = {
  HAND: 'hand',
  FACE: 'face',
};
const CAMERA_STORAGE_KEY = 'particle-demo-camera-device-id';
const CALIBRATION_STORAGE_KEY = 'particle-demo-face-calibration';
const PARTICLE_SETTINGS_STORAGE_KEY = 'particle-demo-particle-settings';
const DEFAULT_CAMERA_VALUE = '__particle_demo_browser_default_camera__';
const PHONE_CAMERA_PATTERN = /iphone|continuity|ipad|phone/i;
const HAND_TIP_INDICES = [4, 8, 12, 16, 20];
const HAND_TIP_SET = new Set(HAND_TIP_INDICES);
const HAND_TIP_SLOTS = new Map(HAND_TIP_INDICES.map((tip, index) => [tip, index]));
const FINGER_CHAINS = [
  { tip: 4, dip: 3, pip: 2, mcp: 1 },
  { tip: 8, dip: 7, pip: 6, mcp: 5 },
  { tip: 12, dip: 11, pip: 10, mcp: 9 },
  { tip: 16, dip: 15, pip: 14, mcp: 13 },
  { tip: 20, dip: 19, pip: 18, mcp: 17 },
];
const LONG_FINGER_TIPS = [8, 12, 16, 20];
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];
const FACE_LANDMARKS = {
  top: 10,
  chin: 152,
  leftCheek: 234,
  rightCheek: 454,
  leftEye: 33,
  rightEye: 263,
  leftEyeInner: 133,
  rightEyeInner: 362,
  leftEyeTop: 159,
  leftEyeBottom: 145,
  rightEyeTop: 386,
  rightEyeBottom: 374,
  leftTemple: 127,
  rightTemple: 356,
  leftJaw: 172,
  rightJaw: 397,
  nose: 1,
  mouthTop: 13,
  mouthBottom: 14,
  mouthLeft: 61,
  mouthRight: 291,
};
const FACE_DEBUG_POINTS = [
  10, 21, 54, 67, 103, 109, 127, 132, 136, 148, 152, 162, 172, 176, 234, 251, 284, 297,
  323, 332, 356, 361, 365, 377, 389, 397, 454, 1, 4, 33, 61, 70, 105, 133, 145, 159, 263,
  291, 300, 334, 362, 374, 386,
];
const LOGO_FACE_ANCHORS = {
  eyeMid: new THREE.Vector2(0, 0.58),
  noseTip: new THREE.Vector2(0.08, -0.34),
  mouth: new THREE.Vector2(-0.02, -1.3),
  eyeToMouth: 1.82,
  cheekWidth: 5.55,
};
const DEFAULT_FACE_EYE_ANCHORS = new THREE.Vector4(-1.08, 0.58, 1.08, 0.58);
const DEFAULT_FACE_CALIBRATION = {
  maskScale: 1.22,
  maskX: 0,
  maskY: 0.13,
  eyeOriginX: 0,
  eyeOriginY: 0.13,
  eyeSpread: 1.31,
  eyeHeight: 1,
  eyeScale: 0.92,
  eyeShape: 1.75,
  eyeIntensity: 1.73,
  blinkResponse: 1.15,
};
const CALIBRATION_CONTROLS = [
  { key: 'maskScale', label: 'Mask scale', min: 0.75, max: 1.45, step: 0.01 },
  { key: 'maskX', label: 'Mask origin X', min: -1.2, max: 1.2, step: 0.01 },
  { key: 'maskY', label: 'Mask origin Y', min: -1.2, max: 1.2, step: 0.01 },
  { key: 'eyeOriginX', label: 'Eye origin X', min: -0.8, max: 0.8, step: 0.01 },
  { key: 'eyeOriginY', label: 'Eye origin Y', min: -0.8, max: 0.8, step: 0.01 },
  { key: 'eyeSpread', label: 'Eye spread', min: 0.55, max: 1.6, step: 0.01 },
  { key: 'eyeHeight', label: 'Eye height', min: 0.35, max: 2.1, step: 0.01 },
  { key: 'eyeScale', label: 'Eye beam size', min: 0.35, max: 2.4, step: 0.01 },
  { key: 'eyeShape', label: 'Eye sharpness', min: 0.45, max: 2.2, step: 0.01 },
  { key: 'eyeIntensity', label: 'Eye intensity', min: 0, max: 2.4, step: 0.01 },
  { key: 'blinkResponse', label: 'Blink response', min: 0, max: 2.2, step: 0.01 },
];
const DEFAULT_PARTICLE_SETTINGS = {
  handSmoothing: 0.76,
  handDeadzone: 0.018,
  handMaxStep: 0.16,
  trackingFps: 45,
  fingerStrength: 1.08,
  palmStrength: 1.56,
  pinchStrength: 0.96,
  particleForce: 1.16,
  particleCurl: 0.96,
  particleDepth: 1.18,
  handContactRadius: 1,
  handPalmRadius: 1.36,
  handPinchReach: 1,
  handWake: 1.16,
  idleMotion: 0.78,
  particleDensity: 0.54,
  bloom: 0.45,
  faceCameraOpacity: 0.42,
  faceFollow: 0.82,
  faceMotion: 2.2,
};
const PARTICLE_SETTING_CONTROLS = [
  {
    key: 'handSmoothing',
    section: 'Hand',
    label: 'Hand smooth',
    min: 0,
    max: 0.92,
    step: 0.01,
    hint: 'Higher values reduce fingertip jitter, with a little more input lag.',
  },
  {
    key: 'handDeadzone',
    section: 'Hand',
    label: 'Deadzone',
    min: 0,
    max: 0.08,
    step: 0.001,
    precision: 3,
    hint: 'Ignores tiny landmark noise before it reaches the particle force field.',
  },
  {
    key: 'handMaxStep',
    section: 'Hand',
    label: 'Max jump',
    min: 0.08,
    max: 0.5,
    step: 0.01,
    hint: 'Limits how far a tracked point can move in one frame after filtering.',
  },
  {
    key: 'trackingFps',
    section: 'Shared',
    label: 'Track FPS',
    min: 24,
    max: 60,
    step: 1,
    precision: 0,
    hint: 'Caps MediaPipe inference separately from the 60 FPS renderer.',
  },
  {
    key: 'fingerStrength',
    section: 'Hand',
    label: 'Finger force',
    min: 0,
    max: 1.5,
    step: 0.01,
    hint: 'Scales each fingertip repulsion field before it enters the shader.',
  },
  {
    key: 'palmStrength',
    section: 'Hand',
    label: 'Palm force',
    min: 0,
    max: 1.5,
    step: 0.01,
    hint: 'Scales the open-palm pressure field that blooms the cloud outward.',
  },
  {
    key: 'pinchStrength',
    section: 'Hand',
    label: 'Pinch force',
    min: 0,
    max: 1.5,
    step: 0.01,
    hint: 'Scales the pinch gravity well and vortex effect.',
  },
  {
    key: 'particleForce',
    section: 'Hand',
    label: 'Field power',
    min: 0,
    max: 1.6,
    step: 0.01,
    hint: 'Global multiplier for hand-driven particle displacement.',
  },
  {
    key: 'particleCurl',
    section: 'Hand',
    label: 'Curl',
    min: 0,
    max: 1.8,
    step: 0.01,
    hint: 'Adds sideways swirl to finger wakes, pinches, gusts, and shockwaves.',
  },
  {
    key: 'particleDepth',
    section: 'Hand',
    label: 'Depth push',
    min: 0,
    max: 1.8,
    step: 0.01,
    hint: 'Controls how much hand forces push particles toward the camera.',
  },
  {
    key: 'handContactRadius',
    section: 'Hand',
    label: 'Contact size',
    min: 0.55,
    max: 1.75,
    step: 0.01,
    hint: 'Scales fingertip contact radius. Lower values make finger pushes more precise.',
  },
  {
    key: 'handPalmRadius',
    section: 'Hand',
    label: 'Palm reach',
    min: 0.55,
    max: 1.75,
    step: 0.01,
    hint: 'Scales the open-palm pressure disk without changing fingertip size.',
  },
  {
    key: 'handPinchReach',
    section: 'Hand',
    label: 'Pinch reach',
    min: 0.55,
    max: 1.8,
    step: 0.01,
    hint: 'Controls how far the pinch gravity well reaches into the particle cloud.',
  },
  {
    key: 'handWake',
    section: 'Hand',
    label: 'Wake',
    min: 0,
    max: 1.8,
    step: 0.01,
    hint: 'Scales velocity trails from fast hand and finger movement.',
  },
  {
    key: 'idleMotion',
    section: 'Shared',
    label: 'Idle sway',
    min: 0,
    max: 1.6,
    step: 0.01,
    hint: 'Controls the logo cloud breathing motion when no hand force is active.',
  },
  {
    key: 'particleDensity',
    section: 'Shared',
    label: 'Density',
    min: 0.18,
    max: 1,
    step: 0.01,
    hint: 'Caps how many particles are drawn; lower this first for stable FPS.',
  },
  {
    key: 'bloom',
    section: 'Shared',
    label: 'Bloom',
    min: 0,
    max: 1.4,
    step: 0.01,
    hint: 'Controls post-process glow and particle emission brightness.',
  },
  {
    key: 'faceCameraOpacity',
    section: 'Face',
    label: 'Camera opacity',
    min: 0,
    max: 0.78,
    step: 0.01,
    hint: 'Controls the webcam backdrop opacity in face mode only.',
  },
  {
    key: 'faceFollow',
    section: 'Face',
    label: 'Face follow',
    min: 0,
    max: 1,
    step: 0.01,
    hint: '0 makes the mask trail behind your face. 1 locks it almost instantly to your face.',
  },
  {
    key: 'faceMotion',
    section: 'Face',
    label: 'Head shake',
    min: 0,
    max: 4,
    step: 0.01,
    hint: 'Scales the visible particle wobble, swirl, and depth kick when you move your head fast.',
  },
];
const HAND_FILTER = { kind: 'hand' };
const FACE_FILTER = { kind: 'face' };
const SETTING_SECTIONS = ['Shared', 'Hand', 'Face'];
const FACE_POSITION_DEADZONE = 0.018;
const FACE_SCALE_DEADZONE = 0.007;
const FACE_ROLL_DEADZONE = 0.01;
const FACE_EXPRESSION_DEADZONE = 0.025;
const FACE_EYE_ANCHOR_DEADZONE = 0.035;
const FACE_JUMP_CONTAIN_START = 0.42;
const FACE_JUMP_CONTAIN_END = 1.12;
const FACE_SCALE_MAX_STEP = 0.045;
const FACE_ROLL_MAX_STEP = 0.075;
const FACE_YAW_MAX_STEP = 0.12;
const FACE_EYE_ANCHOR_MAX_STEP = 0.24;
const FACE_RESULT_FRESH_MS = 420;
const FACE_DROPOUT_HOLD_MS = 420;
const FACE_CACHE_RESET_GRACE_MS = 1100;
const HAND_RESULT_FRESH_MS = 560;
const HAND_DROPOUT_HOLD_MS = 820;
const HAND_SLOT_RESET_GRACE_MS = 1700;
const HAND_MIN_TRACKING_CONFIDENCE = 0.38;
const HAND_MIN_FRAME_QUALITY = 0.28;
const HAND_ASSIGNMENT_SWITCH_MARGIN = 0.24;
const HAND_ASSIGNMENT_HANDEDNESS_LOCK_MS = 1600;
const HAND_ASSIGNMENT_HANDEDNESS_PENALTY = 0.78;
const EYE_IGNITION_START = 0.46;
const EYE_IGNITION_END = 0.86;

const canvas = document.querySelector('#scene');
const app = document.querySelector('#app');
const brandLogo = document.querySelector('#brandMark');
const video = document.querySelector('#handVideo');
const faceVideoBackdrop = document.querySelector('#faceVideoBackdrop');
const handOverlay = document.querySelector('#handOverlay');
const cameraStatus = document.querySelector('#cameraStatus');
const handDebugMeta = document.querySelector('#handDebugMeta');
const handDebugHands = document.querySelector('#handDebugHands');
const handDebugFps = document.querySelector('#handDebugFps');
const handDebugGesture = document.querySelector('#handDebugGesture');
const cameraSelect = document.querySelector('#cameraSelect');
const cameraRefreshButton = document.querySelector('#cameraRefreshButton');
const pointerToggle = document.querySelector('#pointerToggle');
const debugToggle = document.querySelector('#debugToggle');
const calibrationToggle = document.querySelector('#calibrationToggle');
const calibrationPanel = document.querySelector('#calibrationPanel');
const calibrationControls = document.querySelector('#calibrationControls');
const calibrationValues = document.querySelector('#calibrationValues');
const calibrationReset = document.querySelector('#calibrationReset');
const settingsToggle = document.querySelector('#settingsToggle');
const settingsPanel = document.querySelector('#settingsPanel');
const settingsControls = document.querySelector('#settingsControls');
const settingsValues = document.querySelector('#settingsValues');
const settingsReset = document.querySelector('#settingsReset');
const perfStatus = document.querySelector('#perfStatus');
const handDebugPanel = document.querySelector('#handDebug');
const handModeButton = document.querySelector('#handModeButton');
const faceModeButton = document.querySelector('#faceModeButton');

const pointer = {
  active: false,
  x: 0,
  y: 0,
  strength: 0,
};

const params = new URLSearchParams(window.location.search);
const verifyMode = params.has('verify');
const syntheticHandProfile = verifyMode ? params.get('syntheticHand') || '' : '';
const syntheticHandMode = Boolean(syntheticHandProfile);
let pointerEnabled = params.has('mouse');
let debugEnabled = params.has('debug');
let calibrationEnabled = params.has('tune');
let settingsEnabled = params.has('settings');
let trackingMode =
  params.get('mode') === TRACKING_MODES.FACE || params.has('tune')
    ? TRACKING_MODES.FACE
    : TRACKING_MODES.HAND;
const qualityMode = params.get('quality') === 'high' ? 'high' : 'auto';
let faceCalibration = loadFaceCalibration();
let particleSettings = loadParticleSettings();

let scene;
let camera;
let renderer;
let composer;
let points;
let starField;
let fingerMarkers = [];
let pinchMarkers = [];
let palmMarker;
let shockMarker;
let visionFileset = null;
let handLandmarker;
let faceLandmarker;
let handLandmarkerPromise = null;
let faceLandmarkerPromise = null;
let latestHandResult = null;
let latestFaceResult = null;
let latestDetectionAt = 0;
let latestFaceDetectionAt = 0;
let lastTrackingAt = 0;
let lastTrackingErrorAt = 0;
let lastProcessedVideoTime = -1;
let detectionRunCount = 0;
let duplicateFrameSkipCount = 0;
let invalidHandFrameCount = 0;
let debugLandmarkReadCount = 0;
let handSlotResetCount = 0;
let trackingLoopGeneration = 0;
let trackingIntervalId = 0;
let trackingVideoFrameHandle = 0;
let lastSyntheticHandAt = 0;
let syntheticHandStartedAt = 0;
let lastClapTime = -10;
let lastGustTime = -10;
let lastTwoHandDistance = 10;
let cameraMode = 'CAMERA';
let cameraSettings = { frameRate: 0, width: 0, height: 0 };
let activeCameraDeviceId = '';
let reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let bloomPass;
let renderPixelRatio = getInitialPixelRatio();
let activeParticleCount = qualityMode === 'high' ? PARTICLE_COUNT : getParticleBudget();
let lastRenderAt = 0;

const previousPalmCenters = new Map();
const previousTipPoints = new Map();
const previousPinches = new Map();
const previousFingerScores = new Map();
const previousFingerStability = new Map();
const previousFingerHold = new Map();
const previousFingerContinuity = new Map();
const previousFingerLocal = new Map();
const previousHandGestures = Array.from({ length: 2 }, () => ({
  openPalm: 0,
  pinch: 0,
  palmActive: false,
  pinchActive: false,
  lastSeenAt: -10000,
  handedness: '',
}));
const previousFaceCenters = new Map();
const smoothedFace = {
  scale: 0.36,
  roll: 0,
  mouth: 0,
  blink: 0,
  yaw: 0,
  shakeX: 0,
  shakeY: 0,
  shake: 0,
  eyeAnchors: DEFAULT_FACE_EYE_ANCHORS.clone(),
  strength: 0,
};
const handDebugContext = handOverlay.getContext('2d');
const clock = new THREE.Clock();
const uniforms = {
  uTime: { value: 0 },
  uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
  uIntro: { value: 0 },
  uMode: { value: trackingMode === TRACKING_MODES.FACE ? 1 : 0 },
  uPalm: { value: new THREE.Vector4(0, 0, 0, 0) },
  uPalmVelocity: { value: new THREE.Vector2(0, 0) },
  uFace: { value: new THREE.Vector4(0, 0, 0.36, 0) },
  uFaceVelocity: { value: new THREE.Vector2(0, 0) },
  uFaceRotation: { value: 0 },
  uFaceExpression: { value: new THREE.Vector4(0, 0, 0, 0) },
  uEyeCalib: { value: new THREE.Vector4(0, 0, faceCalibration.eyeScale, faceCalibration.eyeSpread) },
  uEyeFineTune: { value: new THREE.Vector4(0, 0, faceCalibration.eyeShape, faceCalibration.eyeIntensity) },
  uFaceEyeAnchors: { value: DEFAULT_FACE_EYE_ANCHORS.clone() },
  uMotionSettings: {
    value: new THREE.Vector4(
      particleSettings.particleForce,
      particleSettings.particleCurl,
      particleSettings.particleDepth,
      particleSettings.idleMotion,
    ),
  },
  uHandTuning: {
    value: new THREE.Vector4(
      particleSettings.handContactRadius,
      particleSettings.handPalmRadius,
      particleSettings.handPinchReach,
      particleSettings.handWake,
    ),
  },
  uVisualSettings: {
    value: new THREE.Vector4(particleSettings.bloom, particleSettings.faceFollow, particleSettings.faceMotion, 0),
  },
  uForceCount: { value: 0 },
  uForce: {
    value: Array.from({ length: MAX_FORCES }, () => new THREE.Vector4(0, 0, 0, 0)),
  },
  uForceVelocity: {
    value: Array.from({ length: MAX_FORCES }, () => new THREE.Vector2(0, 0)),
  },
  uPinch: {
    value: Array.from({ length: 2 }, () => new THREE.Vector4(0, 0, 0, 0)),
  },
  uGustTime: { value: -100 },
  uGustOrigin: { value: new THREE.Vector2(0, 0) },
  uGustVelocity: { value: new THREE.Vector2(0, 0) },
  uShockTime: { value: -100 },
  uShockCenter: { value: new THREE.Vector2(0, 0) },
};
const tmpForceUniform = new THREE.Vector4();
const tmpPalmUniform = new THREE.Vector4();
const tmpPinchUniform = new THREE.Vector4();
const tmpVector4 = new THREE.Vector4();
const uploadedForceSourceIds = Array.from({ length: MAX_FORCES }, () => -1);
const reusableHandState = createHandState();
const handScratchFingerScores = Array.from({ length: 2 }, () =>
  FINGER_CHAINS.map(({ tip }) => ({ tip, strength: 0, stability: 0, radius: 0 })),
);
const handScratchInteractiveFingers = Array.from({ length: 2 }, () => []);
const handScratchSeenSlots = [false, false];
const handScratchTipPoints = Array.from({ length: MAX_FORCES }, () => new THREE.Vector3());
const handScratchPinchPoints = Array.from({ length: 2 }, () => new THREE.Vector3());
const handScratchDebugLandmarks = [];
const syntheticHandLandmarks = Array.from({ length: 2 }, () =>
  Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 })),
);
const syntheticWorldHandLandmarks = Array.from({ length: 2 }, () =>
  Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 })),
);
const syntheticActiveHandLandmarks = [];
const syntheticActiveWorldHandLandmarks = [];
const syntheticHandedness = [
  [{ score: 0.98, categoryName: 'Right', displayName: 'Right' }],
  [{ score: 0.98, categoryName: 'Left', displayName: 'Left' }],
];
const syntheticActiveHandedness = [];
const syntheticHandResult = {
  landmarks: syntheticActiveHandLandmarks,
  worldLandmarks: syntheticActiveWorldHandLandmarks,
  handedness: syntheticActiveHandedness,
};
const handScratchEntries = Array.from({ length: 2 }, () => ({
  landmarks: null,
  worldLandmarks: null,
  confidence: 0,
  handedness: '',
  rawPalm: new THREE.Vector3(),
  slot: 0,
}));
const handScratchAssignments = Array.from({ length: 2 }, () => ({
  landmarks: null,
  worldLandmarks: null,
  confidence: 0,
  handedness: '',
  rawPalm: new THREE.Vector3(),
  slot: 0,
}));
const handDebugStats = {
  fps: 0,
  inferenceMs: 0,
  lastDetectAt: 0,
  lastVideoFrameAt: 0,
  openPalm: 0,
  pinch: 0,
  forces: 0,
  quality: 0,
  activeHands: 0,
};
const trackingHealth = {
  mode: trackingMode,
  liveCount: 0,
  heldCount: 0,
  missCount: 0,
  lastLiveAt: 0,
  lastUpdateAt: 0,
  staleMs: 0,
};
const renderStats = {
  fps: 0,
  pressure: 0,
  frameCount: 0,
  lastSecondAt: performance.now(),
  lastQualityAt: performance.now(),
};
const trackingStats = {
  pressure: 0,
};

let bootStarted = false;

export function bootParticleEngine() {
  if (bootStarted) {
    return;
  }

  bootStarted = true;
  void boot();
}

async function boot() {
  brandLogo.src = logoUrl;
  setupScene();
  window.addEventListener('resize', handleResize);
  canvas.addEventListener('pointermove', handlePointerMove);
  canvas.addEventListener('pointerleave', handlePointerLeave);
  canvas.addEventListener('pointerdown', handlePointerMove);
  canvas.addEventListener('pointerup', handlePointerLeave);
  cameraSelect.addEventListener('change', handleCameraSelection);
  cameraRefreshButton.addEventListener('click', refreshCameraDevices);
  navigator.mediaDevices?.addEventListener?.('devicechange', refreshCameraDevices);
  pointerToggle.addEventListener('click', togglePointerInput);
  debugToggle.addEventListener('click', toggleDebugPanel);
  calibrationToggle.addEventListener('click', toggleCalibrationPanel);
  calibrationReset.addEventListener('click', resetFaceCalibration);
  settingsToggle.addEventListener('click', toggleSettingsPanel);
  settingsReset.addEventListener('click', resetParticleSettings);
  handModeButton.addEventListener('click', () => setTrackingMode(TRACKING_MODES.HAND));
  faceModeButton.addEventListener('click', () => setTrackingMode(TRACKING_MODES.FACE));
  setupCalibrationControls();
  setupParticleSettingsControls();
  updateModeControls();
  updatePointerToggle();
  updateDebugToggle();
  updateCalibrationToggle();
  updateSettingsToggle();
  applyFaceCalibration();
  applyParticleSettings();
  installVerifyHooks();

  const particleState = await createLogoParticleState(PARTICLE_COUNT);
  points = createParticleSystem(particleState);
  scene.add(points);
  handleResize();
  setupHands();
  requestAnimationFrame(animate);
}

function setupScene() {
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x020305, 0.038);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 12);

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    preserveDrawingBuffer: verifyMode,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(0x020305, 1);
  renderer.setPixelRatio(renderPixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.58, 0.3, 0.18);
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  starField = createStarField();
  scene.add(starField);
  setupInteractionMarkers();
}

function installVerifyHooks() {
  if (!verifyMode) {
    return;
  }

  window.__particleDemoVerify = {
    setRenderPressure(value) {
      renderStats.pressure = THREE.MathUtils.clamp(Number(value) || 0, 0, 1);
    },
    setTrackingPressure(value) {
      trackingStats.pressure = THREE.MathUtils.clamp(Number(value) || 0, 0, 1);
    },
    runDetectionForTest(timestamp = performance.now(), mediaTime = video.currentTime) {
      runHandDetection(timestamp, mediaTime);
      return this.getState();
    },
    resetHandTrackingForTest() {
      resetHandTrackingCaches({ resetCounters: true });
      resetHandUniforms();
      return this.getState();
    },
    testTrackingErrorGuard() {
      resetTrackingForStreamChange();
      trackingMode = TRACKING_MODES.HAND;
      updateModeControls();
      handleTrackingError(new Error('verify tracking error'));
      const handError = this.getState();
      const handStatus = {
        cameraMode,
        trackingPressure: handError.trackingPressure,
        health: { ...handError.trackingHealth },
      };

      resetTrackingForStreamChange();
      trackingMode = TRACKING_MODES.FACE;
      updateModeControls();
      handleTrackingError(new Error('verify face tracking error'));
      const faceError = this.getState();
      const faceStatus = {
        cameraMode,
        trackingPressure: faceError.trackingPressure,
        health: { ...faceError.trackingHealth },
      };

      resetTrackingForStreamChange();
      trackingMode = TRACKING_MODES.HAND;
      updateModeControls();
      setCameraState('LIVE', '');

      return {
        hand: handStatus,
        face: faceStatus,
      };
    },
    testStreamResetClearsTracking() {
      previousPalmCenters.set('palm:0', {
        position: new THREE.Vector3(0.5, -0.2, 1),
        velocity: new THREE.Vector2(1, -1),
        target: new THREE.Vector3(0.5, -0.2, 1),
        delta: new THREE.Vector3(),
        rawVelocity: new THREE.Vector2(),
        speed: 0.4,
      });
      previousHandGestures[0].lastSeenAt = performance.now();
      previousHandGestures[0].handedness = 'Right';
      previousFaceCenters.set('face:center', {
        position: new THREE.Vector3(0.2, 0.1, 2.05),
        velocity: new THREE.Vector2(0.4, 0.1),
        target: new THREE.Vector3(0.2, 0.1, 2.05),
        delta: new THREE.Vector3(),
        rawVelocity: new THREE.Vector2(),
        speed: 0.2,
      });
      uniforms.uForceCount.value = 1;
      uniforms.uForce.value[0].set(1, 1, 1, 0.4);
      uniforms.uPalm.value.set(1, 1, 1, 1);
      uniforms.uFace.value.set(1, 1, 0.7, 1);
      updateTrackingHealth(TRACKING_MODES.HAND, 1, 0);

      resetTrackingForStreamChange();
      const state = this.getState();
      return {
        hasPalmCache: previousPalmCenters.size > 0,
        hasFaceCache: previousFaceCenters.size > 0,
        handLastSeen: previousHandGestures[0].lastSeenAt,
        forceCount: state.forceCount,
        forceEnergy: state.forceEnergy,
        palm: state.palm,
        face: uniforms.uFace.value.toArray(),
        health: { ...state.trackingHealth },
      };
    },
    testVideoFrameGate(firstMediaTime = 1, secondMediaTime = firstMediaTime) {
      lastProcessedVideoTime = -1;
      duplicateFrameSkipCount = 0;
      rememberProcessedVideoFrame(firstMediaTime);
      const duplicate = isDuplicateVideoFrame(secondMediaTime);
      if (duplicate) {
        duplicateFrameSkipCount += 1;
      }
      return {
        duplicate,
        duplicateFrameSkipCount,
        lastProcessedVideoTime,
      };
    },
    testFaceDropoutCache() {
      previousFaceCenters.clear();
      withVelocity('face:center', new THREE.Vector3(0.42, -0.18, 2.05), previousFaceCenters, 1 / 60, FACE_FILTER);
      smoothedFace.scale = 0.52;
      smoothedFace.strength = 1;

      latestFaceResult = null;
      latestFaceDetectionAt = performance.now();
      const shortMissState = readFaceFrame(1 / 60);
      const keptAfterShortMiss = previousFaceCenters.has('face:center');

      latestFaceDetectionAt = performance.now() - FACE_CACHE_RESET_GRACE_MS - 20;
      const longMissState = readFaceFrame(1 / 60);
      const clearedAfterLongMiss = !previousFaceCenters.has('face:center');

      previousFaceCenters.clear();
      return {
        keptAfterShortMiss,
        clearedAfterLongMiss,
        shortVisible: shortMissState.visible,
        shortStrength: shortMissState.strength,
        shortCenter: shortMissState.center.toArray(),
        longVisible: longMissState.visible,
      };
    },
    testFaceInvalidFrameGuard() {
      previousFaceCenters.clear();
      withVelocity('face:center', new THREE.Vector3(0.18, -0.12, 2.05), previousFaceCenters, 1 / 60, FACE_FILTER);
      smoothedFace.scale = 0.52;
      smoothedFace.strength = 1;

      const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      landmarks[FACE_LANDMARKS.leftCheek] = { x: Number.NaN, y: 0.46, z: 0 };
      latestFaceResult = {
        faceLandmarks: [landmarks],
        faceBlendshapes: [],
        facialTransformationMatrixes: [],
      };
      latestFaceDetectionAt = performance.now();
      const usable = hasUsableFaceResult(latestFaceResult);
      const state = readFaceFrame(1 / 60);
      const result = {
        usable,
        visible: state.visible,
        held: state.held,
        center: state.center.toArray(),
        strength: state.strength,
        health: { ...trackingHealth },
      };

      latestFaceResult = null;
      previousFaceCenters.clear();
      return result;
    },
    testFaceFilterResponse() {
      const previousFollow = particleSettings.faceFollow;
      particleSettings.faceFollow = 0.82;
      previousFaceCenters.clear();

      const first = withVelocity('face:center', new THREE.Vector3(0, 0, 2.05), previousFaceCenters, 1 / 60, FACE_FILTER);
      const jitter = withVelocity('face:center', new THREE.Vector3(0.014, -0.01, 2.05), previousFaceCenters, 1 / 60, FACE_FILTER);
      const jitterMove = Math.hypot(jitter.position.x, jitter.position.y);
      const beforeJump = jitter.position.clone();
      const jump = withVelocity('face:center', new THREE.Vector3(0.9, -0.2, 2.05), previousFaceCenters, 1 / 60, FACE_FILTER);

      const result = {
        first: first.position.toArray(),
        jitterMove,
        jumpStep: jump.position.distanceTo(beforeJump),
        jumpTargetDistance: new THREE.Vector3(0.9, -0.2, 2.05).distanceTo(beforeJump),
        speed: jump.speed,
      };

      previousFaceCenters.clear();
      particleSettings.faceFollow = previousFollow;
      return result;
    },
    testFacePoseOutlierDamping() {
      smoothedFace.scale = 0.52;
      smoothedFace.roll = 0.04;
      smoothedFace.yaw = 0.05;
      smoothedFace.eyeAnchors.set(-1.05, 0.58, 1.05, 0.58);

      const before = {
        scale: smoothedFace.scale,
        roll: smoothedFace.roll,
        yaw: smoothedFace.yaw,
        eyeAnchors: smoothedFace.eyeAnchors.toArray(),
      };
      applyFacePoseSmoothing({
        rawScale: 0.9,
        rawRoll: 0.82,
        rawMouth: 0.06,
        rawBlink: 0.02,
        rawYaw: -0.92,
        rawEyeAnchors: new THREE.Vector4(-2.4, 1.35, 2.25, 1.18),
        alpha: 0.64,
        motionBoost: 0.95,
      });
      const after = {
        scale: smoothedFace.scale,
        roll: smoothedFace.roll,
        yaw: smoothedFace.yaw,
        eyeAnchors: smoothedFace.eyeAnchors.toArray(),
      };

      smoothedFace.scale = 0.52;
      smoothedFace.roll = 0.04;
      smoothedFace.yaw = 0.05;
      smoothedFace.eyeAnchors.set(-1.05, 0.58, 1.05, 0.58);
      return {
        before,
        after,
        scaleStep: Math.abs(after.scale - before.scale),
        rollStep: Math.abs(after.roll - before.roll),
        yawStep: Math.abs(after.yaw - before.yaw),
        eyeStep: Math.hypot(
          after.eyeAnchors[0] - before.eyeAnchors[0],
          after.eyeAnchors[1] - before.eyeAnchors[1],
          after.eyeAnchors[2] - before.eyeAnchors[2],
          after.eyeAnchors[3] - before.eyeAnchors[3],
        ),
      };
    },
    testFaceExpressionResponse() {
      const neutral = getBlinkStrength(createTestFaceLandmarks(0.18), {
        eyeBlinkLeft: 0.02,
        eyeBlinkRight: 0.02,
        eyeSquintLeft: 0.16,
        eyeSquintRight: 0.16,
        eyeWideLeft: 0.12,
        eyeWideRight: 0.12,
      });
      const tired = getBlinkStrength(createTestFaceLandmarks(0.12), {
        eyeBlinkLeft: 0.06,
        eyeBlinkRight: 0.06,
        eyeSquintLeft: 0.34,
        eyeSquintRight: 0.34,
        eyeWideLeft: 0.04,
        eyeWideRight: 0.04,
      });
      const squint = getBlinkStrength(createTestFaceLandmarks(0.08), {
        eyeBlinkLeft: 0.08,
        eyeBlinkRight: 0.08,
        eyeSquintLeft: 0.68,
        eyeSquintRight: 0.68,
        eyeWideLeft: 0.02,
        eyeWideRight: 0.02,
      });
      const blink = getBlinkStrength(createTestFaceLandmarks(0.045), {
        eyeBlinkLeft: 0.82,
        eyeBlinkRight: 0.82,
        eyeSquintLeft: 0.18,
        eyeSquintRight: 0.18,
        eyeWideLeft: 0,
        eyeWideRight: 0,
      });
      return {
        neutral,
        tired,
        squint,
        blink,
        neutralIgnition: getEyeIgnitionAmount(neutral),
        tiredIgnition: getEyeIgnitionAmount(tired),
        squintIgnition: getEyeIgnitionAmount(squint),
        blinkIgnition: getEyeIgnitionAmount(blink),
      };
    },
    testFaceStateApplication() {
      resetTrackingForStreamChange();
      trackingMode = TRACKING_MODES.FACE;
      updateModeControls();
      particleSettings.faceFollow = 1;
      applyParticleSettings();

      const firstState = createSyntheticFaceStateForTest({
        center: new THREE.Vector3(0.08, -0.04, 2.05),
        velocity: new THREE.Vector2(0.4, -0.1),
        speed: 0.18,
        scale: 0.62,
        roll: 0.08,
        mouth: 0.18,
        blink: 0.06,
        yaw: 0.22,
        shake: new THREE.Vector3(0.02, -0.01, 0.08),
        strength: 1,
      });
      applyFaceState(firstState);

      const expressiveState = createSyntheticFaceStateForTest({
        center: new THREE.Vector3(0.64, -0.24, 2.05),
        velocity: new THREE.Vector2(3.8, -1.6),
        speed: 0.82,
        scale: 0.68,
        roll: -0.18,
        mouth: 0.72,
        blink: 0.84,
        yaw: -0.58,
        shake: new THREE.Vector3(0.24, -0.12, 0.7),
        strength: 1.05,
      });
      for (let index = 0; index < 5; index += 1) {
        applyFaceState(expressiveState);
      }
      updateTrackingHealth(TRACKING_MODES.FACE, 1, 0);
      drawTrackingDebug(null, expressiveState);

      const active = {
        mode: uniforms.uMode.value,
        face: uniforms.uFace.value.toArray(),
        velocity: uniforms.uFaceVelocity.value.toArray(),
        rotation: uniforms.uFaceRotation.value,
        expression: uniforms.uFaceExpression.value.toArray(),
        eyeAnchors: uniforms.uFaceEyeAnchors.value.toArray(),
        cameraMode,
        health: { ...trackingHealth },
      };

      applyFaceState({ ...expressiveState, visible: false });
      updateTrackingHealth(TRACKING_MODES.FACE, 0, 0);
      const faded = {
        face: uniforms.uFace.value.toArray(),
        velocity: uniforms.uFaceVelocity.value.toArray(),
        expression: uniforms.uFaceExpression.value.toArray(),
        health: { ...trackingHealth },
      };

      for (let index = 0; index < 3; index += 1) {
        applyFaceState(expressiveState);
      }
      updateTrackingHealth(TRACKING_MODES.FACE, 1, 0);
      drawTrackingDebug(null, expressiveState);

      return { active, faded };
    },
    getState() {
      let forceEnergy = 0;
      let activeForceSlots = 0;
      for (const force of uniforms.uForce.value) {
        forceEnergy += force.z;
        if (force.z > FORCE_LOOP_EPSILON) {
          activeForceSlots += 1;
        }
      }
      let pinchEnergy = 0;
      for (const pinch of uniforms.uPinch.value) {
        pinchEnergy += pinch.z;
      }
      return {
        mode: uniforms.uMode.value,
        syntheticHandProfile,
        pointerEnabled,
        forceCount: uniforms.uForceCount.value,
        activeForceSlots,
        liveForceCount: reusableHandState.activeForceCount,
        heldForceCount: reusableHandState.heldForceCount,
        unstableFingerCount: reusableHandState.unstableFingerCount,
        forceEnergy,
        handSlots: previousHandGestures.map((slot) => ({
          handedness: slot.handedness,
          lastSeenAge: performance.now() - slot.lastSeenAt,
        })),
        forceSources: uploadedForceSourceIds.slice(0, uniforms.uForceCount.value),
        forces: uniforms.uForce.value.map((force) => force.toArray()),
        palm: uniforms.uPalm.value.toArray(),
        palmVelocity: uniforms.uPalmVelocity.value.toArray(),
        pinchEnergy,
        pinches: uniforms.uPinch.value.map((pinch) => pinch.toArray()),
        activeParticleCount,
        renderPixelRatio,
        renderPressure: renderStats.pressure,
        trackingPressure: trackingStats.pressure,
        adaptivePressure: getAdaptivePressure(),
        inferenceMs: handDebugStats.inferenceMs,
        effectiveTrackingFps: getEffectiveTrackingFps(),
        detectionRunCount,
        duplicateFrameSkipCount,
        invalidHandFrameCount,
        debugLandmarkReadCount,
        handSlotResetCount,
        trackingHealth: { ...trackingHealth },
        lastProcessedVideoTime,
        gustAge: uniforms.uTime.value - uniforms.uGustTime.value,
        shockAge: uniforms.uTime.value - uniforms.uShockTime.value,
        handText: handDebugHands.textContent,
        gestureText: handDebugGesture.textContent,
        fpsText: perfStatus.textContent,
      };
    },
  };
}

function createSyntheticFaceStateForTest(overrides = {}) {
  return {
    visible: true,
    held: false,
    landmarks: null,
    center: new THREE.Vector3(0, 0, 2.05),
    velocity: new THREE.Vector2(0, 0),
    speed: 0,
    scale: 0.62,
    roll: 0,
    mouth: 0,
    blink: 0,
    yaw: 0,
    eyeAnchors: DEFAULT_FACE_EYE_ANCHORS.clone(),
    shake: new THREE.Vector3(0, 0, 0),
    strength: 1,
    ...overrides,
  };
}

function setupInteractionMarkers() {
  const fingerTexture = createRadialTexture('#82e2ff', 96);
  const palmTexture = createRadialTexture('#00f5a0', 160);

  fingerMarkers = Array.from({ length: MAX_FORCES }, () => {
    const marker = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: fingerTexture,
        color: 0x82e2ff,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    marker.scale.setScalar(0.36);
    marker.visible = false;
    scene.add(marker);
    return marker;
  });

  pinchMarkers = Array.from({ length: 2 }, () => {
    const marker = new THREE.Mesh(
      new THREE.TorusGeometry(0.34, 0.02, 10, 72),
      new THREE.MeshBasicMaterial({
        color: 0xffdf6e,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    marker.visible = false;
    scene.add(marker);
    return marker;
  });

  palmMarker = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: palmTexture,
      color: 0x00f5a0,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  palmMarker.visible = false;
  scene.add(palmMarker);

  shockMarker = new THREE.Mesh(
    new THREE.TorusGeometry(1, 0.012, 8, 96),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
  shockMarker.visible = false;
  scene.add(shockMarker);
}

function createRadialTexture(color, size) {
  const textureCanvas = document.createElement('canvas');
  textureCanvas.width = size;
  textureCanvas.height = size;
  const context = textureCanvas.getContext('2d');
  const gradient = context.createRadialGradient(
    size * 0.5,
    size * 0.5,
    0,
    size * 0.5,
    size * 0.5,
    size * 0.5,
  );
  gradient.addColorStop(0, color);
  gradient.addColorStop(0.2, color);
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(textureCanvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

async function createLogoParticleState(count) {
  const samples = await sampleLogoMask();
  const positions = new Float32Array(count * 3);
  const scatter = new Float32Array(count * 3);
  const seed = new Float32Array(count);
  const size = new Float32Array(count);
  const core = new Float32Array(count);

  const aspect = 60 / 56;
  const logoHeight = 6.35;
  const logoWidth = logoHeight * aspect;
  const depth = 0.42;
  const brightEvery = 4;

  for (let i = 0; i < count; i += 1) {
    const useBright = i % brightEvery === 0 && samples.bright.length > 0;
    const source = useBright ? samples.bright : samples.filled;
    const sample = source[Math.floor(Math.random() * source.length)];
    const jitter = useBright ? 0.007 : 0.018;
    const x = (sample.x - 0.5) * logoWidth + randomSigned(jitter);
    const y = (0.5 - sample.y) * logoHeight + randomSigned(jitter);
    const z = randomSigned(depth) * (useBright ? 0.35 : 1);
    const radius = 8 + Math.random() * 13;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    scatter[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    scatter[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
    scatter[i * 3 + 2] = radius * Math.cos(phi) * 0.72;
    seed[i] = Math.random();
    size[i] = useBright ? 1.55 + Math.random() * 0.95 : 0.9 + Math.random() * 1.05;
    core[i] = useBright ? 1 : sample.brightness;
  }

  return { positions, scatter, seed, size, core };
}

async function sampleLogoMask() {
  const image = new Image();
  image.decoding = 'async';
  image.src = logoUrl;
  await image.decode();

  const width = 900;
  const height = 840;
  const canvas2d = document.createElement('canvas');
  canvas2d.width = width;
  canvas2d.height = height;
  const context = canvas2d.getContext('2d', { willReadFrequently: true });
  context.clearRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);

  const pixels = context.getImageData(0, 0, width, height).data;
  const filled = [];
  const bright = [];
  const stride = 2;

  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const offset = (y * width + x) * 4;
      const r = pixels[offset];
      const g = pixels[offset + 1];
      const b = pixels[offset + 2];
      const a = pixels[offset + 3];
      const brightness = (r + g + b) / 765;
      if (a > 32) {
        filled.push({
          x: (x + Math.random() * stride) / width,
          y: (y + Math.random() * stride) / height,
          brightness,
        });
      }
      if (a > 32 && brightness > 0.62) {
        bright.push({
          x: (x + Math.random() * stride) / width,
          y: (y + Math.random() * stride) / height,
          brightness,
        });
      }
    }
  }

  return { filled, bright };
}

function createParticleSystem(state) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(state.positions, 3));
  geometry.setAttribute('aScatter', new THREE.BufferAttribute(state.scatter, 3));
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(state.seed, 1));
  geometry.setAttribute('aSize', new THREE.BufferAttribute(state.size, 1));
  geometry.setAttribute('aCore', new THREE.BufferAttribute(state.core, 1));
  geometry.computeBoundingSphere();

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const particlePoints = new THREE.Points(geometry, material);
  geometry.setDrawRange(0, activeParticleCount);
  particlePoints.rotation.x = -0.05;
  return particlePoints;
}

function createStarField() {
  const count = 560;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i += 1) {
    const radius = 26 + Math.random() * 44;
    const theta = Math.random() * Math.PI * 2;
    const y = randomSigned(18);
    positions[i * 3] = Math.cos(theta) * radius;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = Math.sin(theta) * radius - 12;
    const cold = 0.46 + Math.random() * 0.46;
    colors[i * 3] = 0.24 * cold;
    colors[i * 3 + 1] = 0.38 * cold;
    colors[i * 3 + 2] = 0.62 * cold;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: 0.014,
      vertexColors: true,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    }),
  );
}

async function setupHands() {
  if (syntheticHandMode) {
    setupSyntheticHandInput();
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraState('POINTER', 'warn');
    return;
  }

  try {
    setCameraState('LOADING', 'warn');
    visionFileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
    );
    await ensureLandmarkerForMode();

    let stream = await openBestCameraStream(await getPreferredCameraDeviceId());
    video.srcObject = stream;
    faceVideoBackdrop.srcObject = stream;
    const [videoTrack] = stream.getVideoTracks();
    updateCameraSettings(videoTrack?.getSettings?.() || {});
    await populateCameraSelect(videoTrack?.getSettings?.().deviceId || '');
    await Promise.all([video.play(), faceVideoBackdrop.play().catch(() => {})]);
    setCameraState('LIVE', '');
    startHandDetectionLoop();
  } catch (error) {
    console.warn('Camera tracking unavailable:', error);
    setCameraState('POINTER', 'warn');
  }
}

function setupSyntheticHandInput() {
  cameraSelect.innerHTML = '';
  const option = document.createElement('option');
  option.value = 'synthetic';
  option.textContent = `Synthetic ${syntheticHandProfile}`;
  cameraSelect.append(option);
  cameraSelect.disabled = true;
  cameraRefreshButton.disabled = true;
  updateCameraSettings({ frameRate: 60, width: 640, height: 360, deviceId: 'synthetic' });
  setCameraState('SYNTH HAND', 'warn');
  syntheticHandStartedAt = 0;

  const tick = (timestamp = performance.now()) => {
    const minTrackingFrameMs = 1000 / getEffectiveTrackingFps();
    if (!lastSyntheticHandAt || timestamp - lastSyntheticHandAt >= minTrackingFrameMs - FRAME_SKIP_EPSILON_MS) {
      lastSyntheticHandAt = timestamp;
      const elapsedSeconds = timestamp * 0.001;
      if (!syntheticHandStartedAt) {
        syntheticHandStartedAt = elapsedSeconds;
      }
      updateSyntheticHandResult(elapsedSeconds - syntheticHandStartedAt);
    }
    window.requestAnimationFrame(tick);
  };
  tick();
}

function updateSyntheticHandResult(elapsedSeconds) {
  if (trackingMode !== TRACKING_MODES.HAND) {
    latestHandResult = null;
    return;
  }

  const now = performance.now();
  const profile =
    syntheticHandProfile === 'pinch' ||
    syntheticHandProfile === 'two' ||
    syntheticHandProfile === 'cross' ||
    syntheticHandProfile === 'drop' ||
    syntheticHandProfile === 'dropout' ||
    syntheticHandProfile === 'stall' ||
    syntheticHandProfile === 'flicker' ||
    syntheticHandProfile === 'glitch' ||
    syntheticHandProfile === 'point' ||
    syntheticHandProfile === 'corrupt' ||
    syntheticHandProfile === 'weak' ||
    syntheticHandProfile === 'none' ||
    syntheticHandProfile === 'sweep'
      ? syntheticHandProfile
      : 'open';
  const driftX = Math.sin(elapsedSeconds * 1.8) * 0.035;
  const driftY = Math.sin(elapsedSeconds * 1.15) * 0.018;
  const handCount =
    profile === 'none' || isSyntheticDropoutFrame(profile, elapsedSeconds)
      ? 0
      : profile === 'two' || profile === 'cross' || (profile === 'drop' && elapsedSeconds < 2.4)
        ? 2
        : 1;

  if (profile === 'none' || handCount === 0) {
    // Intentionally empty: used to verify mouse opt-in and brief tracking dropouts.
  } else if (profile === 'flicker') {
    writeSyntheticFlickerHand(0, 0.5 + driftX, 0.66 + driftY, elapsedSeconds);
  } else if (profile === 'glitch') {
    writeSyntheticGlitchPointHand(0, 0.5 + driftX * 0.45, 0.66 + driftY * 0.35, elapsedSeconds);
  } else if (profile === 'point') {
    writeSyntheticPointHand(0, 0.5 + driftX * 0.45, 0.66 + driftY * 0.35);
  } else if (profile === 'corrupt') {
    writeSyntheticHand(0, 0.5 + driftX, 0.66 + driftY, 0, 1);
    if (elapsedSeconds > 1.2) {
      syntheticHandLandmarks[0][8].x = Number.NaN;
      syntheticHandLandmarks[0][12].y = Number.POSITIVE_INFINITY;
      syntheticWorldHandLandmarks[0][5].z = Number.NaN;
    }
  } else if (profile === 'weak') {
    writeSyntheticHand(0, 0.5 + driftX, 0.66 + driftY, 0, 1);
  } else if (profile === 'cross') {
    const travel = Math.sin(elapsedSeconds * 1.05 - 1.2) * 0.23;
    writeSyntheticHand(0, 0.5 + travel, 0.65 + driftY * 0.5, 0, 1);
    writeSyntheticHand(1, 0.5 - travel, 0.65 - driftY * 0.35, 0, -1);
  } else if (profile === 'sweep') {
    const sweepX = 0.5 + Math.sin(elapsedSeconds * 5.4) * 0.24;
    const sweepY = 0.66 + Math.sin(elapsedSeconds * 2.2) * 0.035;
    writeSyntheticHand(0, sweepX, sweepY, 0, 1);
  } else {
    writeSyntheticHand(0, 0.5 + driftX, 0.66 + driftY, profile === 'pinch' ? 1 : 0, 1);
  }
  if (handCount === 2 && profile !== 'cross') {
    const close = 0.09 + Math.sin(elapsedSeconds * 3.2) * 0.045;
    writeSyntheticHand(0, 0.5 - close, 0.65 + driftY, 0, 1);
    writeSyntheticHand(1, 0.5 + close, 0.65 - driftY * 0.4, 0, -1);
  }

  syntheticActiveHandLandmarks.length = handCount;
  syntheticActiveWorldHandLandmarks.length = handCount;
  syntheticActiveHandedness.length = handCount;
  for (let index = 0; index < handCount; index += 1) {
    syntheticActiveHandLandmarks[index] = syntheticHandLandmarks[index];
    syntheticActiveWorldHandLandmarks[index] = syntheticWorldHandLandmarks[index];
    syntheticActiveHandedness[index] = syntheticHandedness[index];
    syntheticActiveHandedness[index][0].score = profile === 'weak' ? 0.2 : 0.98;
  }
  latestHandResult = syntheticHandResult;
  latestDetectionAt = now;
  const syntheticFps = 1000 / Math.max(1, now - (handDebugStats.lastVideoFrameAt || now - 16.7));
  handDebugStats.fps = THREE.MathUtils.lerp(handDebugStats.fps || syntheticFps, syntheticFps, 0.18);
  handDebugStats.lastVideoFrameAt = now;
  handDebugStats.lastDetectAt = now;
}

function isSyntheticDropoutFrame(profile, elapsedSeconds) {
  if (profile === 'stall') {
    if (elapsedSeconds < 1.2) {
      return false;
    }
    const phase = (elapsedSeconds - 1.2) % 2.1;
    return phase > 0.36 && phase < 1.28;
  }
  if (profile !== 'dropout' || elapsedSeconds < 1.4) {
    return false;
  }
  const phase = elapsedSeconds % 1.15;
  return phase > 0.72 && phase < 0.84;
}

function isSyntheticFingerGlitchFrame(elapsedSeconds) {
  if (elapsedSeconds < 1.25) {
    return false;
  }
  const phase = (elapsedSeconds - 1.25) % 0.68;
  return phase < 0.15;
}

function writeSyntheticHand(handIndex, centerX, wristY, pinchAmount, handednessSign) {
  const landmarks = syntheticHandLandmarks[handIndex];
  const worldLandmarks = syntheticWorldHandLandmarks[handIndex];
  const sign = handednessSign || 1;
  const points = [
    [0, 0, 0.055, 0],
    [1, -0.07 * sign, -0.01, -0.01],
    [2, -0.12 * sign, -0.07, -0.02],
    [3, -0.16 * sign, -0.12, -0.025],
    [4, -0.2 * sign, -0.17, -0.03],
    [5, -0.08 * sign, -0.14, -0.02],
    [6, -0.095 * sign, -0.28, -0.025],
    [7, -0.105 * sign, -0.39, -0.03],
    [8, -0.11 * sign, -0.51, -0.035],
    [9, 0, -0.17, -0.02],
    [10, 0, -0.32, -0.025],
    [11, 0, -0.44, -0.03],
    [12, 0, -0.58, -0.035],
    [13, 0.08 * sign, -0.14, -0.02],
    [14, 0.095 * sign, -0.29, -0.025],
    [15, 0.105 * sign, -0.4, -0.03],
    [16, 0.115 * sign, -0.51, -0.035],
    [17, 0.16 * sign, -0.09, -0.015],
    [18, 0.19 * sign, -0.23, -0.02],
    [19, 0.21 * sign, -0.33, -0.025],
    [20, 0.225 * sign, -0.43, -0.03],
  ];

  for (const [index, xOffset, yOffset, zOffset] of points) {
    setSyntheticLandmark(landmarks[index], worldLandmarks[index], centerX + xOffset, wristY + yOffset, zOffset, centerX, wristY);
  }

  if (pinchAmount > 0) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const pinchX = centerX - 0.13 * sign;
    const pinchY = wristY - 0.38;
    thumbTip.x = THREE.MathUtils.lerp(thumbTip.x, pinchX, pinchAmount);
    thumbTip.y = THREE.MathUtils.lerp(thumbTip.y, pinchY, pinchAmount);
    indexTip.x = THREE.MathUtils.lerp(indexTip.x, pinchX + 0.012 * sign, pinchAmount);
    indexTip.y = THREE.MathUtils.lerp(indexTip.y, pinchY + 0.008, pinchAmount);
    setSyntheticWorldFromScreen(worldLandmarks[4], thumbTip, centerX, wristY);
    setSyntheticWorldFromScreen(worldLandmarks[8], indexTip, centerX, wristY);
  }
}

function writeSyntheticPointHand(handIndex, centerX, wristY) {
  writeSyntheticHand(handIndex, centerX, wristY, 0, 1);
  const landmarks = syntheticHandLandmarks[handIndex];
  const worldLandmarks = syntheticWorldHandLandmarks[handIndex];
  const curled = [
    [1, -0.06, -0.02, -0.01],
    [2, -0.11, -0.08, -0.014],
    [3, -0.16, -0.12, -0.016],
    [4, -0.2, -0.16, -0.018],
    [10, 0.025, -0.25, -0.018],
    [11, 0.038, -0.22, -0.018],
    [12, 0.018, -0.205, -0.018],
    [14, 0.074, -0.235, -0.018],
    [15, 0.066, -0.205, -0.018],
    [16, 0.056, -0.19, -0.018],
    [18, 0.13, -0.195, -0.016],
    [19, 0.12, -0.175, -0.016],
    [20, 0.1, -0.16, -0.016],
  ];
  for (const [index, xOffset, yOffset, zOffset] of curled) {
    setSyntheticLandmark(landmarks[index], worldLandmarks[index], centerX + xOffset, wristY + yOffset, zOffset, centerX, wristY);
  }
}

function writeSyntheticGlitchPointHand(handIndex, centerX, wristY, elapsedSeconds) {
  writeSyntheticPointHand(handIndex, centerX, wristY);
  if (!isSyntheticFingerGlitchFrame(elapsedSeconds)) {
    return;
  }

  const landmarks = syntheticHandLandmarks[handIndex];
  const worldLandmarks = syntheticWorldHandLandmarks[handIndex];
  setSyntheticLandmark(landmarks[8], worldLandmarks[8], centerX + 0.26, wristY - 0.54, -0.05, centerX, wristY);
}

function writeSyntheticFlickerHand(handIndex, centerX, wristY, elapsedSeconds) {
  writeSyntheticHand(handIndex, centerX, wristY, 0, 1);
  const landmarks = syntheticHandLandmarks[handIndex];
  const worldLandmarks = syntheticWorldHandLandmarks[handIndex];
  const curled = [
    [1, -0.07, -0.01, -0.01],
    [2, -0.11, -0.06, -0.014],
    [3, -0.2, -0.07, -0.016],
    [4, -0.25, -0.13, -0.015],
    [6, -0.05, -0.22, -0.018],
    [7, -0.03, -0.2, -0.018],
    [8, -0.055, -0.2, -0.018],
    [10, 0.02, -0.25, -0.018],
    [11, 0.03, -0.22, -0.018],
    [12, 0, -0.21, -0.018],
    [14, 0.07, -0.24, -0.018],
    [15, 0.06, -0.21, -0.018],
    [16, 0.055, -0.2, -0.018],
    [18, 0.13, -0.2, -0.016],
    [19, 0.12, -0.18, -0.016],
    [20, 0.1, -0.17, -0.016],
  ];
  for (const [index, xOffset, yOffset, zOffset] of curled) {
    setSyntheticLandmark(landmarks[index], worldLandmarks[index], centerX + xOffset, wristY + yOffset, zOffset, centerX, wristY);
  }

  const flickerPhase = elapsedSeconds % 0.52;
  if (flickerPhase < 0.034) {
    setSyntheticLandmark(landmarks[8], worldLandmarks[8], centerX - 0.11, wristY - 0.51, -0.035, centerX, wristY);
  }
}

function setSyntheticLandmark(landmark, worldLandmark, x, y, z, centerX, wristY) {
  landmark.x = THREE.MathUtils.clamp(x, 0.05, 0.95);
  landmark.y = THREE.MathUtils.clamp(y, 0.05, 0.95);
  landmark.z = z;
  setSyntheticWorldFromScreen(worldLandmark, landmark, centerX, wristY);
}

function setSyntheticWorldFromScreen(worldLandmark, landmark, centerX, wristY) {
  worldLandmark.x = (landmark.x - centerX) * 0.38;
  worldLandmark.y = (landmark.y - wristY) * 0.38;
  worldLandmark.z = landmark.z * 0.16;
}

async function openBestCameraStream(deviceId = '', useBrowserDefault = false) {
  const withDevice = (video) => ({
    ...video,
    ...(deviceId ? { deviceId: { exact: deviceId } } : useBrowserDefault ? {} : { facingMode: 'user' }),
  });
  const candidates = [
    {
      video: withDevice({
        width: { exact: 960 },
        height: { exact: 540 },
        frameRate: { exact: 60 },
      }),
      audio: false,
    },
    {
      video: withDevice({
        width: { exact: 640 },
        height: { exact: 360 },
        frameRate: { exact: 60 },
      }),
      audio: false,
    },
    {
      video: withDevice({
        width: { exact: 640 },
        height: { exact: 480 },
        frameRate: { exact: 60 },
      }),
      audio: false,
    },
    {
      video: withDevice({
        width: { ideal: 960 },
        height: { ideal: 540 },
        frameRate: { ideal: 60, max: 60 },
      }),
      audio: false,
    },
    {
      video: withDevice({
        width: { ideal: 640 },
        height: { ideal: 360 },
        frameRate: { ideal: 60, max: 60 },
      }),
      audio: false,
    },
  ];

  let lastError;
  for (const constraints of candidates) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError;
}

async function getPreferredCameraDeviceId() {
  const devices = await getVideoInputDevices();
  const savedDeviceId = getSavedCameraDeviceId();
  if (savedDeviceId && devices.some((device) => device.deviceId === savedDeviceId)) {
    return savedDeviceId;
  }

  const nonPhoneDevices = devices.filter((device) => !isPhoneCamera(device.label));
  const preferred =
    nonPhoneDevices.find((device) => /facetime|built-in|integrated|usb|webcam|camera/i.test(device.label)) ||
    nonPhoneDevices[0] ||
    devices[0];
  return preferred?.deviceId || '';
}

async function refreshCameraDevices() {
  const selectedDeviceId = activeCameraDeviceId || cameraSettings.deviceId || cameraSelect.value;
  try {
    cameraRefreshButton.disabled = true;
    setCameraState('SCANNING', 'warn');
    await populateCameraSelect(selectedDeviceId);
    setCameraState(video.srcObject ? 'LIVE' : 'CAMERA', '');
  } finally {
    cameraRefreshButton.disabled = false;
  }
}

async function populateCameraSelect(selectedDeviceId = '') {
  const devices = await getVideoInputDevices();
  cameraSelect.innerHTML = '';

  const defaultOption = document.createElement('option');
  defaultOption.value = DEFAULT_CAMERA_VALUE;
  defaultOption.textContent = 'Default camera';
  defaultOption.title = 'Ask the browser for its current default camera';
  cameraSelect.append(defaultOption);

  devices.forEach((device, index) => {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.textContent = formatCameraLabel(device.label, index + 1);
    option.title = device.label || option.textContent;
    cameraSelect.append(option);
  });

  if (selectedDeviceId && devices.some((device) => device.deviceId === selectedDeviceId)) {
    cameraSelect.value = selectedDeviceId;
    activeCameraDeviceId = selectedDeviceId;
  } else {
    cameraSelect.value = DEFAULT_CAMERA_VALUE;
    activeCameraDeviceId = '';
  }
}

async function getVideoInputDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === 'videoinput');
}

async function handleCameraSelection() {
  const deviceId = cameraSelect.value;
  const useBrowserDefault = deviceId === DEFAULT_CAMERA_VALUE;
  if (!deviceId || (!useBrowserDefault && deviceId === activeCameraDeviceId)) {
    return;
  }

  try {
    setCameraState('SWITCHING', 'warn');
    if (useBrowserDefault) {
      clearSavedCameraDeviceId();
    }
    const stream = await openBestCameraStream(useBrowserDefault ? '' : deviceId, useBrowserDefault);
    stopStream(video.srcObject);
    video.srcObject = stream;
    faceVideoBackdrop.srcObject = stream;
    await Promise.all([video.play(), faceVideoBackdrop.play().catch(() => {})]);
    const [track] = stream.getVideoTracks();
    updateCameraSettings(track?.getSettings?.() || {});
    const actualDeviceId = track?.getSettings?.().deviceId || (useBrowserDefault ? '' : deviceId);
    await populateCameraSelect(actualDeviceId);
    if (actualDeviceId) {
      saveCameraDeviceId(actualDeviceId);
    }
    resetTrackingForStreamChange();
    startHandDetectionLoop();
    setCameraState('LIVE', '');
  } catch (error) {
    console.warn('Camera switch failed:', error);
    setCameraState('CAMERA ERR', 'offline');
  }
}

function stopStream(stream) {
  if (!stream) {
    return;
  }

  for (const track of stream.getTracks()) {
    track.stop();
  }
}

function isPhoneCamera(label = '') {
  return PHONE_CAMERA_PATTERN.test(label);
}

function formatCameraLabel(label = '', index = 1) {
  const cleaned = label
    .replace(/\s*\([0-9a-f:._-]{4,}\)\s*$/i, '')
    .replace(/\s+/g, ' ')
    .trim();
  const name = cleaned || `Camera ${index}`;
  return isPhoneCamera(name) && !/^iphone/i.test(name) ? `iPhone ${name}` : name;
}

function getSavedCameraDeviceId() {
  try {
    return window.localStorage.getItem(CAMERA_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

function saveCameraDeviceId(deviceId) {
  if (!deviceId || deviceId === DEFAULT_CAMERA_VALUE) {
    return;
  }

  try {
    window.localStorage.setItem(CAMERA_STORAGE_KEY, deviceId);
  } catch {
    // Storage can be blocked in private contexts; camera selection still works for this session.
  }
}

function clearSavedCameraDeviceId() {
  try {
    window.localStorage.removeItem(CAMERA_STORAGE_KEY);
  } catch {
    // Storage can be blocked in private contexts.
  }
}

function updateCameraSettings(settings) {
  cameraSettings = {
    frameRate: Number(settings.frameRate) || 0,
    width: Number(settings.width) || 0,
    height: Number(settings.height) || 0,
    deviceId: settings.deviceId || '',
  };
  const summary = cameraSettings.frameRate
    ? `${Math.round(cameraSettings.frameRate)}fps ${cameraSettings.width}x${cameraSettings.height}`
    : 'camera fps unavailable';
  cameraStatus.title = summary;
  handDebugMeta.title = summary;
  if (cameraSettings.frameRate > 0 && cameraSettings.frameRate < 50) {
    cameraStatus.classList.add('warn');
  }
}

async function createHandLandmarker(vision) {
  const options = {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.44,
    minHandPresenceConfidence: 0.46,
    minTrackingConfidence: 0.5,
  };

  try {
    return await HandLandmarker.createFromOptions(vision, options);
  } catch (error) {
    console.warn('GPU hand landmarker unavailable, falling back to CPU:', error);
    return HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: {
        ...options.baseOptions,
        delegate: 'CPU',
      },
    });
  }
}

async function createFaceLandmarker(vision) {
  const options = {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.48,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.52,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  };

  try {
    return await FaceLandmarker.createFromOptions(vision, options);
  } catch (error) {
    console.warn('GPU face landmarker unavailable, falling back to CPU:', error);
    return FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: {
        ...options.baseOptions,
        delegate: 'CPU',
      },
    });
  }
}

function ensureLandmarkerForMode() {
  return trackingMode === TRACKING_MODES.FACE ? ensureFaceLandmarker() : ensureHandLandmarker();
}

function ensureHandLandmarker() {
  if (handLandmarker) {
    return Promise.resolve(handLandmarker);
  }
  if (!visionFileset) {
    return Promise.resolve(null);
  }
  if (!handLandmarkerPromise) {
    setCameraState('HAND MODEL', 'warn');
    handLandmarkerPromise = createHandLandmarker(visionFileset)
      .then((landmarker) => {
        handLandmarker = landmarker;
        return landmarker;
      })
      .catch((error) => {
        handLandmarkerPromise = null;
        throw error;
      });
  }
  return handLandmarkerPromise;
}

function ensureFaceLandmarker() {
  if (faceLandmarker) {
    return Promise.resolve(faceLandmarker);
  }
  if (!visionFileset) {
    return Promise.resolve(null);
  }
  if (!faceLandmarkerPromise) {
    setCameraState('FACE MODEL', 'warn');
    faceLandmarkerPromise = createFaceLandmarker(visionFileset)
      .then((landmarker) => {
        faceLandmarker = landmarker;
        return landmarker;
      })
      .catch((error) => {
        faceLandmarkerPromise = null;
        throw error;
      });
  }
  return faceLandmarkerPromise;
}

function requestLandmarkerForMode() {
  ensureLandmarkerForMode().catch((error) => {
    console.warn('Tracking model unavailable:', error);
    setCameraState('TRACK ERR', 'offline');
  });
}

function startHandDetectionLoop() {
  cancelHandDetectionLoop();
  const generation = ++trackingLoopGeneration;
  lastTrackingAt = 0;
  lastProcessedVideoTime = -1;

  if (typeof video.requestVideoFrameCallback === 'function') {
    const onVideoFrame = (now, metadata) => {
      if (generation !== trackingLoopGeneration) {
        return;
      }
      runHandDetection(now, metadata?.mediaTime);
      if (generation === trackingLoopGeneration && video.srcObject) {
        trackingVideoFrameHandle = video.requestVideoFrameCallback(onVideoFrame);
      }
    };
    trackingVideoFrameHandle = video.requestVideoFrameCallback(onVideoFrame);
    return;
  }

  trackingIntervalId = window.setInterval(() => {
    if (generation === trackingLoopGeneration) {
      runHandDetection(performance.now(), video.currentTime);
    }
  }, 1000 / 60);
}

function cancelHandDetectionLoop() {
  trackingLoopGeneration += 1;
  if (trackingIntervalId) {
    window.clearInterval(trackingIntervalId);
    trackingIntervalId = 0;
  }
  if (
    trackingVideoFrameHandle &&
    typeof video.cancelVideoFrameCallback === 'function'
  ) {
    video.cancelVideoFrameCallback(trackingVideoFrameHandle);
  }
  trackingVideoFrameHandle = 0;
}

function runHandDetection(timestamp, mediaTime = video.currentTime) {
  if (document.hidden || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  const startedAt = performance.now();
  if (isDuplicateVideoFrame(mediaTime)) {
    duplicateFrameSkipCount += 1;
    return;
  }
  const minTrackingFrameMs = 1000 / getEffectiveTrackingFps();
  if (lastTrackingAt && startedAt - lastTrackingAt < minTrackingFrameMs - FRAME_SKIP_EPSILON_MS) {
    return;
  }
  lastTrackingAt = startedAt;
  rememberProcessedVideoFrame(mediaTime);

  if (trackingMode === TRACKING_MODES.FACE) {
    if (!faceLandmarker) {
      requestLandmarkerForMode();
      return;
    }
    const inferenceStartedAt = performance.now();
    try {
      latestFaceResult = faceLandmarker.detectForVideo(video, timestamp);
    } catch (error) {
      handleTrackingError(error);
      return;
    }
    updateTrackingPerformance(performance.now() - inferenceStartedAt);
    if (hasUsableFaceResult(latestFaceResult)) {
      latestFaceDetectionAt = startedAt;
    } else {
      latestFaceResult = null;
    }
    latestHandResult = null;
  } else {
    if (!handLandmarker) {
      requestLandmarkerForMode();
      return;
    }
    const inferenceStartedAt = performance.now();
    try {
      latestHandResult = handLandmarker.detectForVideo(video, timestamp);
    } catch (error) {
      handleTrackingError(error);
      return;
    }
    updateTrackingPerformance(performance.now() - inferenceStartedAt);
    latestDetectionAt = startedAt;
    latestFaceResult = null;
  }
  detectionRunCount += 1;

  if (handDebugStats.lastVideoFrameAt > 0) {
    const videoFps = 1000 / Math.max(1, startedAt - handDebugStats.lastVideoFrameAt);
    handDebugStats.fps = THREE.MathUtils.lerp(handDebugStats.fps || videoFps, videoFps, 0.18);
  }
  handDebugStats.lastVideoFrameAt = startedAt;
  handDebugStats.lastDetectAt = startedAt;
}

function handleTrackingError(error) {
  const now = performance.now();
  if (now - lastTrackingErrorAt > 1000) {
    console.warn('Tracking frame failed:', error);
    lastTrackingErrorAt = now;
  }
  if (trackingMode === TRACKING_MODES.FACE) {
    latestFaceResult = null;
    updateTrackingHealth(TRACKING_MODES.FACE, 0, 0);
  } else {
    latestHandResult = null;
    updateTrackingHealth(TRACKING_MODES.HAND, 0, 0);
  }
  trackingStats.pressure = Math.max(trackingStats.pressure, 0.65);
  setCameraState('TRACK ERR', 'offline');
}

function updateTrackingPerformance(inferenceMs) {
  if (!Number.isFinite(inferenceMs)) {
    return;
  }
  handDebugStats.inferenceMs = THREE.MathUtils.lerp(handDebugStats.inferenceMs || inferenceMs, inferenceMs, 0.18);
  const configuredFps = Math.max(1, particleSettings.trackingFps);
  const frameBudgetMs = 1000 / configuredFps;
  const pressureTarget = smoothstep(0.52, 0.92, handDebugStats.inferenceMs / Math.max(1, frameBudgetMs));
  trackingStats.pressure = THREE.MathUtils.lerp(
    trackingStats.pressure,
    pressureTarget,
    pressureTarget > trackingStats.pressure ? 0.24 : 0.08,
  );
}

function updateTrackingHealth(mode, liveCount, heldCount = 0) {
  const now = performance.now();
  if (trackingHealth.mode !== mode) {
    resetTrackingHealth(mode);
  }
  trackingHealth.mode = mode;
  trackingHealth.liveCount = liveCount;
  trackingHealth.heldCount = heldCount;
  trackingHealth.lastUpdateAt = now;
  if (liveCount > 0) {
    trackingHealth.lastLiveAt = now;
    trackingHealth.missCount = 0;
    trackingHealth.staleMs = 0;
    return;
  }

  trackingHealth.missCount += 1;
  trackingHealth.staleMs = trackingHealth.lastLiveAt > 0 ? now - trackingHealth.lastLiveAt : 0;
}

function resetTrackingHealth(mode = trackingMode) {
  trackingHealth.mode = mode;
  trackingHealth.liveCount = 0;
  trackingHealth.heldCount = 0;
  trackingHealth.missCount = 0;
  trackingHealth.lastLiveAt = 0;
  trackingHealth.lastUpdateAt = performance.now();
  trackingHealth.staleMs = 0;
  handDebugStats.fps = 0;
  handDebugStats.inferenceMs = 0;
  handDebugStats.lastDetectAt = 0;
  handDebugStats.lastVideoFrameAt = 0;
  handDebugStats.openPalm = 0;
  handDebugStats.pinch = 0;
  handDebugStats.forces = 0;
  handDebugStats.quality = 0;
  handDebugStats.activeHands = 0;
}

function getTrackingHealthSuffix() {
  if (trackingHealth.liveCount > 0 || trackingHealth.missCount < 2) {
    return '';
  }
  const staleMs = Math.min(999, Math.round(trackingHealth.staleMs));
  if (trackingHealth.heldCount > 0) {
    return ` H${staleMs}`;
  }
  return ` M${staleMs}`;
}

function isDuplicateVideoFrame(mediaTime) {
  return (
    Number.isFinite(mediaTime) &&
    mediaTime > 0 &&
    lastProcessedVideoTime >= 0 &&
    mediaTime <= lastProcessedVideoTime + 0.00001
  );
}

function rememberProcessedVideoFrame(mediaTime) {
  if (Number.isFinite(mediaTime) && mediaTime > 0) {
    lastProcessedVideoTime = mediaTime;
  }
}

function createHandState() {
  return {
    forces: Array.from({ length: MAX_FORCES }, () => null),
    forcePool: Array.from({ length: MAX_FORCES }, () => ({
      position: new THREE.Vector3(),
      velocity: new THREE.Vector2(),
      strength: 0,
      speed: 0,
      radius: 0,
    })),
    pinches: Array.from({ length: 2 }, () => null),
    pinchPool: Array.from({ length: 2 }, () => ({
      position: new THREE.Vector3(),
      velocity: new THREE.Vector2(),
      speed: 0,
      strength: 0,
    })),
    activeForceCount: 0,
    heldForceCount: 0,
    unstableFingerCount: 0,
    activeHands: 0,
    quality: 0,
    palm: {
      position: new THREE.Vector3(0, 0, 0),
      velocity: new THREE.Vector2(0, 0),
      strength: 0,
      speed: 0,
      depth: 0,
    },
    centers: [],
    liveHands: 0,
    heldHands: 0,
  };
}

function resetHandState(state) {
  state.forces.fill(null);
  state.pinches.fill(null);
  state.activeForceCount = 0;
  state.heldForceCount = 0;
  state.unstableFingerCount = 0;
  state.activeHands = 0;
  state.quality = 0;
  state.liveHands = 0;
  state.heldHands = 0;
  state.palm.position.set(0, 0, 0);
  state.palm.velocity.set(0, 0);
  state.palm.strength = 0;
  state.palm.speed = 0;
  state.palm.depth = 0;
  state.centers.length = 0;
  return state;
}

function readHandFrame(elapsedSeconds, deltaSeconds) {
  const handState = resetHandState(reusableHandState);
  handScratchSeenSlots[0] = false;
  handScratchSeenSlots[1] = false;

  const handEntryCount = getFreshHands();

  if (handEntryCount > 0) {
    const assignmentCount = getStableHandAssignments(handEntryCount);
    for (let assignmentIndex = 0; assignmentIndex < assignmentCount; assignmentIndex += 1) {
      const assignment = handScratchAssignments[assignmentIndex];
      const { landmarks, worldLandmarks, confidence, slot: handSlot, rawPalm } = assignment;
      const metrics = getHandMetrics(landmarks, worldLandmarks, confidence);
      if (confidence < HAND_MIN_TRACKING_CONFIDENCE || metrics.quality < HAND_MIN_FRAME_QUALITY) {
        fadeHandGestureState(handSlot, deltaSeconds);
        continue;
      }

      handState.activeHands += 1;
      handState.liveHands += 1;
      handState.quality = Math.max(handState.quality, metrics.quality);
      handScratchSeenSlots[handSlot] = true;
      previousHandGestures[handSlot].lastSeenAt = performance.now();
      previousHandGestures[handSlot].handedness = assignment.handedness;

      const palm = withVelocity(`palm:${handSlot}`, rawPalm, previousPalmCenters, deltaSeconds, HAND_FILTER);
      const fingerScores = smoothFingerScores(
        getFingerExtensionScores(landmarks, metrics, handScratchFingerScores[handSlot]),
        handSlot,
        deltaSeconds,
        landmarks,
        metrics,
      );
      for (const finger of fingerScores) {
        if (finger.strength > 0.42 && finger.continuity < 0.45) {
          handState.unstableFingerCount += 1;
        }
      }
      const pinch = getPinch(landmarks, handSlot, deltaSeconds, metrics, fingerScores);
      const rawOpenPalm = getOpenPalmStrength(landmarks, fingerScores, metrics) * (1 - pinch.strength * 0.58);
      const gesture = stabilizeHandGesture(handSlot, rawOpenPalm, pinch.strength, deltaSeconds);
      pinch.strength = gesture.pinch;
      const openPalm = gesture.pinchActive ? gesture.openPalm * 0.34 : gesture.openPalm;

      handState.centers.push(palm.position);
      if (openPalm > handState.palm.strength) {
        handState.palm.position.copy(palm.position);
        handState.palm.velocity.copy(palm.velocity);
        handState.palm.strength = openPalm * metrics.quality * (0.74 + metrics.depth * 0.32);
        handState.palm.speed = palm.speed;
        handState.palm.depth = metrics.depth;
      }

      if (pinch.strength > 0.08) {
        const pinchState = handState.pinchPool[handSlot];
        pinchState.position.copy(pinch.position);
        pinchState.velocity.copy(pinch.velocity);
        pinchState.speed = pinch.speed;
        pinchState.strength = pinch.strength * metrics.quality * (0.9 + metrics.depth * 0.28);
        handState.pinches[handSlot] = pinchState;
      }

      for (const finger of getInteractiveFingers(
        fingerScores,
        openPalm,
        pinch.strength,
        handScratchInteractiveFingers[handSlot],
      )) {
        const forceSlot = getFingerForceSlot(handSlot, finger.tip);
        const tip = landmarkToSceneInto(landmarks[finger.tip], handScratchTipPoints[forceSlot]);
        const key = `tip:${handSlot}:${finger.tip}`;
        const point = withVelocity(key, tip, previousTipPoints, deltaSeconds, HAND_FILTER);
        const forceState = handState.forcePool[forceSlot];
        forceState.position.copy(point.position);
        forceState.velocity.copy(point.velocity);
        forceState.strength =
          finger.strength *
          finger.stability *
          finger.stability *
          metrics.quality *
          (openPalm > 0.58 ? 0.22 : 0.86) *
          (1 - pinch.strength * 0.52) *
          (0.88 + metrics.depth * 0.34);
        forceState.speed = point.speed;
        forceState.radius = finger.radius * (0.92 + metrics.depth * 0.22);
        handState.forces[forceSlot] = forceState;
        handState.activeForceCount += 1;
      }
    }
    cleanupMissingHandSlots(handScratchSeenSlots, deltaSeconds);
  } else {
    holdRecentHandState(handState);
    cleanupMissingHandSlots(handScratchSeenSlots, deltaSeconds);
  }

  if (handState.activeForceCount === 0 && pointer.strength > 0.01) {
    const forceState = handState.forcePool[0];
    forceState.position.set(pointer.x, pointer.y, 1);
    forceState.velocity.set(0, 0);
    forceState.strength = pointer.strength;
    forceState.speed = 0;
    forceState.radius = 0.58;
    handState.forces[0] = forceState;
    handState.activeForceCount = 1;
  }

  updateGustTrigger(handState, elapsedSeconds);
  updateShockTrigger(handState, elapsedSeconds);
  updateTrackingHealth(TRACKING_MODES.HAND, handState.liveHands, handState.heldHands);
  return handState;
}

function getStableHandAssignments(entryCount) {
  for (let index = 0; index < entryCount; index += 1) {
    getPalmCenterInto(handScratchEntries[index].landmarks, handScratchEntries[index].rawPalm);
  }

  if (entryCount === 2) {
    const first = handScratchEntries[0];
    const second = handScratchEntries[1];
    const hasPrevious = previousPalmCenters.has('palm:0') || previousPalmCenters.has('palm:1');
    if (!hasPrevious) {
      if (first.rawPalm.x <= second.rawPalm.x) {
        setHandAssignment(0, first, 0);
        setHandAssignment(1, second, 1);
      } else {
        setHandAssignment(0, second, 0);
        setHandAssignment(1, first, 1);
      }
      return 2;
    }

    const handednessLockedAssignment = getHandednessLockedAssignment(first, second);
    if (handednessLockedAssignment) {
      setHandAssignment(0, first, handednessLockedAssignment.firstSlot);
      setHandAssignment(1, second, handednessLockedAssignment.secondSlot);
      return 2;
    }

    const swappedCost = getHandAssignmentCost(first, 1) + getHandAssignmentCost(second, 0);
    const sameAssignmentCost = getHandAssignmentCost(first, 0) + getHandAssignmentCost(second, 1);
    if (swappedCost + HAND_ASSIGNMENT_SWITCH_MARGIN < sameAssignmentCost) {
      setHandAssignment(0, first, 1);
      setHandAssignment(1, second, 0);
    } else {
      setHandAssignment(0, first, 0);
      setHandAssignment(1, second, 1);
    }
    return 2;
  }

  if (entryCount === 1) {
    const entry = handScratchEntries[0];
    const hasSlot0 = previousPalmCenters.has('palm:0');
    const hasSlot1 = previousPalmCenters.has('palm:1');
    const slot =
      hasSlot1 && (!hasSlot0 || getHandAssignmentCost(entry, 1) + 0.18 < getHandAssignmentCost(entry, 0))
        ? 1
        : 0;
    setHandAssignment(0, entry, slot);
    return 1;
  }

  return 0;
}

function getHandednessLockedAssignment(first, second) {
  if (
    !first.handedness ||
    !second.handedness ||
    first.handedness === second.handedness ||
    first.confidence < 0.62 ||
    second.confidence < 0.62
  ) {
    return null;
  }

  const now = performance.now();
  const slot0 = previousHandGestures[0];
  const slot1 = previousHandGestures[1];
  if (
    now - slot0.lastSeenAt > HAND_ASSIGNMENT_HANDEDNESS_LOCK_MS ||
    now - slot1.lastSeenAt > HAND_ASSIGNMENT_HANDEDNESS_LOCK_MS ||
    !slot0.handedness ||
    !slot1.handedness
  ) {
    return null;
  }

  const sameSlots = first.handedness === slot0.handedness && second.handedness === slot1.handedness;
  const swappedSlots = first.handedness === slot1.handedness && second.handedness === slot0.handedness;
  if (sameSlots && !swappedSlots) {
    return { firstSlot: 0, secondSlot: 1 };
  }
  if (swappedSlots && !sameSlots) {
    return { firstSlot: 1, secondSlot: 0 };
  }
  return null;
}

function setHandAssignment(index, entry, slot) {
  const assignment = handScratchAssignments[index];
  assignment.landmarks = entry.landmarks;
  assignment.worldLandmarks = entry.worldLandmarks;
  assignment.confidence = entry.confidence;
  assignment.handedness = entry.handedness;
  assignment.rawPalm.copy(entry.rawPalm);
  assignment.slot = slot;
}

function getHandSlotCost(point, slot) {
  const previous = previousPalmCenters.get(`palm:${slot}`);
  return previous ? point.distanceTo(previous.position) : 0.75;
}

function getHandAssignmentCost(entry, slot) {
  const gesture = previousHandGestures[slot];
  let cost = getHandSlotCost(entry.rawPalm, slot);
  if (
    entry.handedness &&
    gesture.handedness &&
    entry.handedness !== gesture.handedness &&
    performance.now() - gesture.lastSeenAt < HAND_ASSIGNMENT_HANDEDNESS_LOCK_MS
  ) {
    cost += HAND_ASSIGNMENT_HANDEDNESS_PENALTY;
  }
  return cost;
}

function getFingerForceSlot(handSlot, tipIndex) {
  return handSlot * HAND_TIP_INDICES.length + (HAND_TIP_SLOTS.get(tipIndex) ?? 0);
}

function holdRecentHandState(handState) {
  const now = performance.now();
  for (let slot = 0; slot < previousHandGestures.length; slot += 1) {
    const gesture = previousHandGestures[slot];
    const ageMs = now - gesture.lastSeenAt;
    if (ageMs < 0 || ageMs > HAND_DROPOUT_HOLD_MS || !hasHandSlotCache(slot)) {
      continue;
    }

    const hold = 1 - smoothstep(180, HAND_DROPOUT_HOLD_MS, ageMs);
    if (hold <= 0.01) {
      continue;
    }

    let heldSlot = false;
    const palm = previousPalmCenters.get(`palm:${slot}`);
    if (palm && gesture.openPalm > handState.palm.strength) {
      handState.palm.position.copy(palm.position);
      handState.palm.velocity.copy(palm.velocity).multiplyScalar(hold);
      handState.palm.strength = gesture.openPalm * hold * 0.86;
      handState.palm.speed = palm.speed * hold;
      handState.palm.depth = 0.74;
      handState.quality = Math.max(handState.quality, 0.72 * hold);
      heldSlot = true;
    }

    for (const tip of HAND_TIP_INDICES) {
      const forceSlot = getFingerForceSlot(slot, tip);
      const point = previousTipPoints.get(`tip:${slot}:${tip}`);
      if (!point) {
        continue;
      }
      const score = previousFingerScores.get(`finger:${slot}:${tip}`) || 0;
      const stability = previousFingerStability.get(`finger:${slot}:${tip}`) || 0;
      const strength = score * stability * stability * hold * (gesture.openPalm > 0.58 ? 0.28 : 0.72);
      if (strength < 0.025) {
        continue;
      }
      const forceState = handState.forcePool[forceSlot];
      forceState.position.copy(point.position);
      forceState.velocity.copy(point.velocity).multiplyScalar(hold);
      forceState.strength = strength;
      forceState.speed = point.speed * hold;
      forceState.radius = 0.36 + score * 0.2;
      handState.forces[forceSlot] = forceState;
      handState.activeForceCount += 1;
      handState.heldForceCount += 1;
      heldSlot = true;
    }

    if (heldSlot) {
      handState.activeHands += 1;
      handState.heldHands += 1;
      handState.quality = Math.max(handState.quality, 0.62 * hold);
    }
  }
}

function cleanupMissingHandSlots(seenSlots, deltaSeconds) {
  const now = performance.now();
  for (let slot = 0; slot < previousHandGestures.length; slot += 1) {
    if (seenSlots[slot]) {
      continue;
    }
    if (!hasHandSlotCache(slot)) {
      continue;
    }
    fadeHandGestureState(slot, deltaSeconds, now - previousHandGestures[slot].lastSeenAt);
    if (now - previousHandGestures[slot].lastSeenAt > HAND_SLOT_RESET_GRACE_MS) {
      clearHandSlotCache(slot);
    }
  }
}

function hasHandSlotCache(slot) {
  if (previousHandGestures[slot].lastSeenAt > -9999) {
    return true;
  }
  if (previousPalmCenters.has(`palm:${slot}`) || previousPinches.has(`pinch:${slot}`)) {
    return true;
  }
  for (const tip of HAND_TIP_INDICES) {
    if (previousTipPoints.has(`tip:${slot}:${tip}`) || previousFingerScores.has(`finger:${slot}:${tip}`)) {
      return true;
    }
  }
  return false;
}

function clearHandSlotCache(slot) {
  handSlotResetCount += 1;
  previousPalmCenters.delete(`palm:${slot}`);
  previousPinches.delete(`pinch:${slot}`);
  for (const tip of HAND_TIP_INDICES) {
    previousTipPoints.delete(`tip:${slot}:${tip}`);
      previousFingerScores.delete(`finger:${slot}:${tip}`);
      previousFingerStability.delete(`finger:${slot}:${tip}`);
      previousFingerHold.delete(`finger:${slot}:${tip}`);
      previousFingerContinuity.delete(`finger:${slot}:${tip}`);
      previousFingerLocal.delete(`finger:${slot}:${tip}`);
    }
  const state = previousHandGestures[slot];
  state.openPalm = 0;
  state.pinch = 0;
  state.palmActive = false;
  state.pinchActive = false;
  state.lastSeenAt = -10000;
  state.handedness = '';
}

function stabilizeHandGesture(handSlot, rawOpenPalm, rawPinch, deltaSeconds) {
  const state = previousHandGestures[handSlot];
  const safeDelta = Math.max(deltaSeconds, 1 / 90);
  const response = THREE.MathUtils.lerp(19, 8.5, particleSettings.handSmoothing);
  const riseAlpha = THREE.MathUtils.clamp(safeDelta * response, 0.16, 0.42);
  const fallAlpha = THREE.MathUtils.clamp(safeDelta * response * 0.42, 0.04, 0.22);

  state.openPalm = THREE.MathUtils.lerp(
    state.openPalm,
    rawOpenPalm,
    rawOpenPalm > state.openPalm ? riseAlpha : fallAlpha,
  );
  state.pinch = THREE.MathUtils.lerp(
    state.pinch,
    rawPinch,
    rawPinch > state.pinch ? Math.max(riseAlpha, 0.24) : fallAlpha,
  );

  state.pinchActive = state.pinchActive ? state.pinch > 0.2 : state.pinch > 0.32;
  state.palmActive = state.palmActive ? state.openPalm > 0.3 && !state.pinchActive : state.openPalm > 0.42 && !state.pinchActive;

  return {
    openPalm: state.palmActive ? Math.max(state.openPalm, 0.48) : state.openPalm,
    pinch: state.pinchActive ? Math.max(state.pinch, 0.12) : state.pinch,
    palmActive: state.palmActive,
    pinchActive: state.pinchActive,
  };
}

function fadeHandGestureState(handSlot, deltaSeconds, missedForMs = HAND_SLOT_RESET_GRACE_MS) {
  const state = previousHandGestures[handSlot];
  const fadeBase = missedForMs < HAND_DROPOUT_HOLD_MS ? 0.62 : 0.2;
  const fade = Math.pow(fadeBase, Math.max(deltaSeconds, 1 / 90));
  state.openPalm *= fade;
  state.pinch *= fade;
  if (state.openPalm < 0.08) {
    state.palmActive = false;
  }
  if (state.pinch < 0.08) {
    state.pinchActive = false;
  }
}

function resetHandGestureStates() {
  for (const state of previousHandGestures) {
    state.openPalm = 0;
    state.pinch = 0;
    state.palmActive = false;
    state.pinchActive = false;
    state.lastSeenAt = -10000;
    state.handedness = '';
  }
}

function resetHandTrackingCaches({ resetCounters = false } = {}) {
  latestHandResult = null;
  latestDetectionAt = 0;
  previousPalmCenters.clear();
  previousTipPoints.clear();
  previousPinches.clear();
  previousFingerScores.clear();
  previousFingerStability.clear();
  previousFingerHold.clear();
  previousFingerContinuity.clear();
  previousFingerLocal.clear();
  resetHandGestureStates();
  if (resetCounters) {
    detectionRunCount = 0;
    duplicateFrameSkipCount = 0;
    invalidHandFrameCount = 0;
    debugLandmarkReadCount = 0;
    handSlotResetCount = 0;
  }
}

function resetFaceTrackingCaches() {
  latestFaceResult = null;
  latestFaceDetectionAt = 0;
  previousFaceCenters.clear();
}

function resetTrackingForStreamChange() {
  lastTrackingAt = 0;
  lastTrackingErrorAt = 0;
  lastProcessedVideoTime = -1;
  resetHandTrackingCaches();
  resetFaceTrackingCaches();
  resetHandUniforms();
  resetFaceUniforms();
  resetTrackingHealth();
}

function smoothFingerScores(fingerScores, handSlot, deltaSeconds, landmarks = null, metrics = null) {
  const safeDelta = Math.max(deltaSeconds, 1 / 90);
  const response = THREE.MathUtils.lerp(20, 7.5, particleSettings.handSmoothing);
  const baseAlpha = THREE.MathUtils.clamp(safeDelta * response, 0.08, 0.36);

  for (const finger of fingerScores) {
    const key = `finger:${handSlot}:${finger.tip}`;
    const continuity = landmarks && metrics
      ? updateFingerContinuity(key, landmarks, finger.tip, metrics.palmWidth, safeDelta)
      : 1;
    const previous = previousFingerScores.get(key) ?? finger.strength;
    const alpha = finger.strength > previous ? Math.max(baseAlpha, 0.2) : baseAlpha * 0.72;
    const smoothed = THREE.MathUtils.lerp(previous, finger.strength, alpha);
    finger.strength = smoothed < 0.035 ? 0 : smoothed;
    finger.continuity = continuity;
    previousFingerScores.set(key, finger.strength);
    const previousHold = previousFingerHold.get(key) ?? 0;
    const hold = finger.strength > 0.48
      ? Math.min(0.35, previousHold + safeDelta)
      : Math.max(0, previousHold - safeDelta * 5);
    previousFingerHold.set(key, hold);
    const previousStability = previousFingerStability.get(key) ?? 0;
    const targetStability =
      smoothstep(0.075, 0.18, hold) *
      smoothstep(0.18, 0.54, finger.strength) *
      THREE.MathUtils.lerp(0.55, 1, smoothstep(0.24, 0.68, continuity));
    const stabilityAlpha = targetStability > previousStability ? Math.max(baseAlpha * 0.72, 0.12) : Math.max(baseAlpha, 0.24);
    const stability = THREE.MathUtils.lerp(previousStability, targetStability, stabilityAlpha);
    finger.stability = stability < 0.035 ? 0 : stability;
    previousFingerStability.set(key, finger.stability);
  }

  return fingerScores;
}

function updateFingerContinuity(key, landmarks, tipIndex, palmWidth, deltaSeconds) {
  const palmScale = Math.max(palmWidth, 0.001);
  const palmX = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) * 0.2;
  const palmY = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) * 0.2;
  const tip = landmarks[tipIndex];
  const localX = (tip.x - palmX) / palmScale;
  const localY = (tip.y - palmY) / palmScale;
  const previousLocal = previousFingerLocal.get(key);

  if (!previousLocal) {
    previousFingerLocal.set(key, { x: localX, y: localY });
    previousFingerContinuity.set(key, 1);
    return 1;
  }

  const localJump = Math.hypot(localX - previousLocal.x, localY - previousLocal.y);
  previousLocal.x = localX;
  previousLocal.y = localY;

  const jumpTrust = smoothstep(1.08, 0.38, localJump);
  const previousTrust = previousFingerContinuity.get(key) ?? 1;
  const alpha =
    jumpTrust < previousTrust
      ? 0.76
      : THREE.MathUtils.clamp(deltaSeconds * 5.2, 0.08, 0.34);
  const continuity = THREE.MathUtils.lerp(previousTrust, jumpTrust, alpha);
  previousFingerContinuity.set(key, continuity);
  return continuity;
}

function getFreshHands() {
  if (!latestHandResult?.landmarks?.length || performance.now() - latestDetectionAt > HAND_RESULT_FRESH_MS) {
    return 0;
  }

  const count = Math.min(2, latestHandResult.landmarks.length);
  let validCount = 0;
  for (let index = 0; index < count; index += 1) {
    const landmarks = latestHandResult.landmarks[index];
    const worldLandmarks = latestHandResult.worldLandmarks?.[index] || null;
    if (!hasValidLandmarkArray(landmarks)) {
      invalidHandFrameCount += 1;
      continue;
    }
    const handedness = latestHandResult.handedness?.[index]?.[0] || latestHandResult.handednesses?.[index]?.[0];
    const entry = handScratchEntries[validCount];
    entry.landmarks = landmarks;
    entry.worldLandmarks = hasValidLandmarkArray(worldLandmarks, true) ? worldLandmarks : null;
    entry.confidence = handedness?.score ?? 1;
    entry.handedness = handedness?.categoryName || handedness?.displayName || '';
    validCount += 1;
  }
  return validCount;
}

function getFreshLandmarks() {
  debugLandmarkReadCount += 1;
  handScratchDebugLandmarks.length = 0;
  if (!latestHandResult?.landmarks?.length || performance.now() - latestDetectionAt > HAND_RESULT_FRESH_MS) {
    return handScratchDebugLandmarks;
  }

  const count = Math.min(2, latestHandResult.landmarks.length);
  for (let index = 0; index < count; index += 1) {
    const landmarks = latestHandResult.landmarks[index];
    if (hasValidLandmarkArray(landmarks)) {
      handScratchDebugLandmarks.push(landmarks);
    }
  }
  return handScratchDebugLandmarks;
}

function hasValidLandmarkArray(landmarks, requireZ = false) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) {
    return false;
  }
  for (let index = 0; index < 21; index += 1) {
    if (!hasFiniteLandmark(landmarks[index], requireZ)) {
      return false;
    }
  }
  return true;
}

function hasFiniteLandmark(landmark, requireZ = false) {
  return (
    landmark &&
    Number.isFinite(landmark.x) &&
    Number.isFinite(landmark.y) &&
    (!requireZ || Number.isFinite(landmark.z))
  );
}

function readFaceFrame(deltaSeconds) {
  const faceResult = getFreshFaceResult();
  const landmarks = faceResult?.faceLandmarks?.[0];
  if (!landmarks) {
    const missedForMs = performance.now() - latestFaceDetectionAt;
    const heldCenter = previousFaceCenters.get('face:center');
    if (heldCenter && missedForMs <= FACE_DROPOUT_HOLD_MS) {
      const hold = 1 - smoothstep(120, FACE_DROPOUT_HOLD_MS, missedForMs);
      smoothedFace.shake *= 0.9;
      smoothedFace.shakeX *= 0.9;
      smoothedFace.shakeY *= 0.9;
      updateTrackingHealth(TRACKING_MODES.FACE, 0, 1);
      return {
        visible: true,
        held: true,
        landmarks: null,
        center: heldCenter.position.clone(),
        velocity: heldCenter.velocity.clone().multiplyScalar(hold),
        speed: heldCenter.speed * hold,
        scale: smoothedFace.scale,
        roll: smoothedFace.roll,
        mouth: smoothedFace.mouth * hold,
        blink: smoothedFace.blink * hold,
        yaw: smoothedFace.yaw,
        eyeAnchors: smoothedFace.eyeAnchors.clone(),
        shake: new THREE.Vector3(smoothedFace.shakeX, smoothedFace.shakeY, smoothedFace.shake),
        strength: Math.max(smoothedFace.strength * (0.78 + hold * 0.22), 0.18),
      };
    }

    if (missedForMs > FACE_CACHE_RESET_GRACE_MS) {
      previousFaceCenters.clear();
    }
    smoothedFace.shake *= 0.82;
    smoothedFace.shakeX *= 0.82;
    smoothedFace.shakeY *= 0.82;
    updateTrackingHealth(TRACKING_MODES.FACE, 0, 0);
    return {
      visible: false,
      landmarks: null,
      center: new THREE.Vector3(0, 0, 0),
      velocity: new THREE.Vector2(0, 0),
      speed: 0,
      scale: smoothedFace.scale,
      roll: smoothedFace.roll,
      mouth: 0,
      blink: 0,
      yaw: smoothedFace.yaw,
      eyeAnchors: smoothedFace.eyeAnchors.clone(),
      shake: new THREE.Vector3(smoothedFace.shakeX, smoothedFace.shakeY, smoothedFace.shake),
      strength: 0,
    };
  }

  const top = landmarks[FACE_LANDMARKS.top];
  const chin = landmarks[FACE_LANDMARKS.chin];
  const leftCheek = landmarks[FACE_LANDMARKS.leftCheek];
  const rightCheek = landmarks[FACE_LANDMARKS.rightCheek];
  const leftTemple = landmarks[FACE_LANDMARKS.leftTemple];
  const rightTemple = landmarks[FACE_LANDMARKS.rightTemple];
  const leftJaw = landmarks[FACE_LANDMARKS.leftJaw];
  const rightJaw = landmarks[FACE_LANDMARKS.rightJaw];
  const leftEyeOuter = landmarks[FACE_LANDMARKS.leftEye];
  const rightEyeOuter = landmarks[FACE_LANDMARKS.rightEye];
  const leftEyeInner = landmarks[FACE_LANDMARKS.leftEyeInner];
  const rightEyeInner = landmarks[FACE_LANDMARKS.rightEyeInner];
  const nose = landmarks[FACE_LANDMARKS.nose];
  const mouthTop = landmarks[FACE_LANDMARKS.mouthTop];
  const mouthBottom = landmarks[FACE_LANDMARKS.mouthBottom];
  const mouthLeft = landmarks[FACE_LANDMARKS.mouthLeft];
  const mouthRight = landmarks[FACE_LANDMARKS.mouthRight];

  const cheekWidth = Math.max(0.001, distance2d(leftCheek, rightCheek));
  const eyeLeft = faceLandmarkToScene(getMidLandmark(leftEyeOuter, leftEyeInner));
  const eyeRight = faceLandmarkToScene(getMidLandmark(rightEyeOuter, rightEyeInner));
  const eyeMid = eyeLeft.clone().add(eyeRight).multiplyScalar(0.5);
  const noseTip = faceLandmarkToScene(nose);
  const mouthMid = faceLandmarkToScene({
    x: (mouthTop.x + mouthBottom.x) * 0.5,
    y: (mouthTop.y + mouthBottom.y) * 0.5,
    z: (mouthTop.z + mouthBottom.z) * 0.5,
  });
  const topScene = faceLandmarkToScene(top);
  const chinScene = faceLandmarkToScene(chin);
  const cheekWidthScene = faceLandmarkToScene(leftCheek).distanceTo(faceLandmarkToScene(rightCheek));
  const templeWidthScene = faceLandmarkToScene(leftTemple).distanceTo(faceLandmarkToScene(rightTemple));
  const jawWidthScene = faceLandmarkToScene(leftJaw).distanceTo(faceLandmarkToScene(rightJaw));
  const faceWidthScene = cheekWidthScene * 0.58 + templeWidthScene * 0.27 + jawWidthScene * 0.15;
  const faceBlendshapes = getFaceBlendshapeSignals(faceResult);
  const faceMatrix = getFaceMatrixMetrics(faceResult);
  const landmarkRoll = Math.atan2(eyeLeft.y - eyeRight.y, eyeLeft.x - eyeRight.x);
  const rawRoll = faceMatrix ? lerpAngle(landmarkRoll, faceMatrix.roll, 0.34) : landmarkRoll;
  const verticalFeatureScale = eyeMid.distanceTo(mouthMid) / LOGO_FACE_ANCHORS.eyeToMouth;
  const horizontalScale = faceWidthScene / LOGO_FACE_ANCHORS.cheekWidth;
  const heightScale = topScene.distanceTo(chinScene) / 4.85;
  const rawScale = THREE.MathUtils.clamp(
    (verticalFeatureScale * 0.58 + horizontalScale * 0.28 + heightScale * 0.14) * faceCalibration.maskScale,
    0.18,
    0.94,
  );
  const originFromEyes = getLogoOriginForAnchor(eyeMid, LOGO_FACE_ANCHORS.eyeMid, rawScale, rawRoll);
  const originFromNose = getLogoOriginForAnchor(noseTip, LOGO_FACE_ANCHORS.noseTip, rawScale, rawRoll);
  const originFromMouth = getLogoOriginForAnchor(mouthMid, LOGO_FACE_ANCHORS.mouth, rawScale, rawRoll);
  const rawOrigin = originFromEyes
    .multiplyScalar(0.58)
    .add(originFromNose.multiplyScalar(0.26))
    .add(originFromMouth.multiplyScalar(0.16));
  rawOrigin.x += faceCalibration.maskX;
  rawOrigin.y += faceCalibration.maskY;
  const center = withVelocity('face:center', rawOrigin, previousFaceCenters, deltaSeconds, FACE_FILTER);
  const mouthGap = distance2d(mouthTop, mouthBottom);
  const mouthWidth = Math.max(0.001, distance2d(mouthLeft, mouthRight));
  const landmarkMouth = smoothstep(0.24, 0.58, mouthGap / mouthWidth);
  const rawMouth = getMouthStrength(landmarkMouth, faceBlendshapes);
  const rawBlink = getBlinkStrength(landmarks, faceBlendshapes);
  const landmarkYaw = getFaceYaw(landmarks);
  const rawYaw = faceMatrix
    ? THREE.MathUtils.clamp(THREE.MathUtils.lerp(landmarkYaw, faceMatrix.yaw, 0.58), -1, 1)
    : landmarkYaw;
  const depth = smoothstep(0.16, 0.32, cheekWidth);
  const safeDelta = Math.max(deltaSeconds, 1 / 90);
  const faceFollow = particleSettings.faceFollow;
  const rawMotion = previousFaceCenters.get('face:center')?.delta?.length?.() || 0;
  const motionBoost = smoothstep(0.035, 0.18, rawMotion);
  const alpha =
    faceFollow > 0.98
      ? THREE.MathUtils.lerp(0.42, 0.68, motionBoost)
      : THREE.MathUtils.clamp(
          safeDelta * THREE.MathUtils.lerp(6, 34, faceFollow) * (1 + motionBoost * 0.9),
          THREE.MathUtils.lerp(0.04, 0.18, faceFollow),
          THREE.MathUtils.lerp(0.18, 0.46, faceFollow) + motionBoost * 0.18,
        );

  applyFacePoseSmoothing({
    rawScale,
    rawRoll,
    rawMouth,
    rawBlink,
    rawYaw,
    rawEyeAnchors: null,
    alpha,
    motionBoost,
  });
  const intentionalSpeed = Math.max(0, center.velocity.length() - 0.28);
  const rawShake = THREE.MathUtils.clamp(intentionalSpeed / 4.5, 0, 1);
  smoothedFace.shake = THREE.MathUtils.lerp(smoothedFace.shake, rawShake, rawShake > smoothedFace.shake ? 0.82 : 0.26);
  const shakeVelocity = rawShake > 0.015 ? center.velocity : new THREE.Vector2(0, 0);
  smoothedFace.shakeX = THREE.MathUtils.lerp(smoothedFace.shakeX, shakeVelocity.x / 6, 0.62);
  smoothedFace.shakeY = THREE.MathUtils.lerp(smoothedFace.shakeY, shakeVelocity.y / 6, 0.62);
  smoothedFace.strength = THREE.MathUtils.lerp(smoothedFace.strength, 1, alpha);
  const rawEyeAnchors = getDetectedEyeAnchors(eyeLeft, eyeRight, center.position, smoothedFace.scale, smoothedFace.roll);
  applyFaceEyeAnchorSmoothing(rawEyeAnchors, alpha, motionBoost);
  updateTrackingHealth(TRACKING_MODES.FACE, 1, 0);

  return {
    visible: true,
    landmarks,
    center: center.position,
    velocity: center.velocity,
    speed: center.speed,
    scale: smoothedFace.scale,
    roll: smoothedFace.roll,
    mouth: smoothedFace.mouth,
    blink: smoothedFace.blink,
    yaw: smoothedFace.yaw,
    eyeAnchors: smoothedFace.eyeAnchors.clone(),
    shake: new THREE.Vector3(smoothedFace.shakeX, smoothedFace.shakeY, smoothedFace.shake),
    strength: smoothedFace.strength * (0.88 + depth * 0.22),
  };
}

function applyFacePoseSmoothing({ rawScale, rawRoll, rawMouth, rawBlink, rawYaw, rawEyeAnchors, alpha, motionBoost }) {
  const poseBoost = THREE.MathUtils.clamp(motionBoost || 0, 0, 1);
  smoothedFace.scale = dampStableScalarLimited(
    smoothedFace.scale,
    rawScale,
    alpha,
    FACE_SCALE_DEADZONE,
    THREE.MathUtils.lerp(FACE_SCALE_MAX_STEP, FACE_SCALE_MAX_STEP * 1.6, poseBoost),
  );
  smoothedFace.roll = dampStableAngleLimited(
    smoothedFace.roll,
    rawRoll,
    alpha,
    FACE_ROLL_DEADZONE,
    THREE.MathUtils.lerp(FACE_ROLL_MAX_STEP, FACE_ROLL_MAX_STEP * 1.65, poseBoost),
  );
  smoothedFace.mouth = dampStableScalar(smoothedFace.mouth, rawMouth, alpha, FACE_EXPRESSION_DEADZONE);
  smoothedFace.blink = THREE.MathUtils.lerp(smoothedFace.blink, rawBlink, rawBlink > smoothedFace.blink ? 0.58 : 0.22);
  smoothedFace.yaw = dampStableScalarLimited(
    smoothedFace.yaw,
    rawYaw,
    THREE.MathUtils.lerp(0.16, 0.36, poseBoost),
    FACE_EXPRESSION_DEADZONE,
    THREE.MathUtils.lerp(FACE_YAW_MAX_STEP, FACE_YAW_MAX_STEP * 1.55, poseBoost),
  );
  if (rawEyeAnchors) {
    applyFaceEyeAnchorSmoothing(rawEyeAnchors, alpha, poseBoost);
  }
}

function applyFaceEyeAnchorSmoothing(rawEyeAnchors, alpha, motionBoost) {
  dampStableVector4Limited(
    smoothedFace.eyeAnchors,
    rawEyeAnchors,
    alpha,
    FACE_EYE_ANCHOR_DEADZONE,
    THREE.MathUtils.lerp(FACE_EYE_ANCHOR_MAX_STEP, FACE_EYE_ANCHOR_MAX_STEP * 1.5, THREE.MathUtils.clamp(motionBoost || 0, 0, 1)),
  );
}

function getMidLandmark(first, second) {
  return {
    x: (first.x + second.x) * 0.5,
    y: (first.y + second.y) * 0.5,
    z: ((first.z || 0) + (second.z || 0)) * 0.5,
  };
}

function getDetectedEyeAnchors(leftEye, rightEye, origin, scale, roll) {
  const left = scenePointToLogoLocal(leftEye, origin, scale, roll);
  const right = scenePointToLogoLocal(rightEye, origin, scale, roll);
  return new THREE.Vector4(left.x, left.y, right.x, right.y);
}

function scenePointToLogoLocal(scenePoint, origin, scale, roll) {
  const dx = (scenePoint.x - origin.x) / Math.max(scale, 0.001);
  const dy = (scenePoint.y - origin.y) / Math.max(scale, 0.001);
  const c = Math.cos(-roll);
  const s = Math.sin(-roll);
  return new THREE.Vector2(c * dx - s * dy, s * dx + c * dy);
}

function getFreshFaceResult() {
  if (!hasUsableFaceResult(latestFaceResult) || performance.now() - latestFaceDetectionAt > FACE_RESULT_FRESH_MS) {
    return null;
  }

  return latestFaceResult;
}

function hasUsableFaceResult(faceResult) {
  const landmarks = faceResult?.faceLandmarks?.[0];
  if (!Array.isArray(landmarks)) {
    return false;
  }

  const requiredIndices = Object.values(FACE_LANDMARKS);
  for (const index of requiredIndices) {
    if (!hasFiniteLandmark(landmarks[index])) {
      return false;
    }
  }

  const cheekWidth = distance2d(landmarks[FACE_LANDMARKS.leftCheek], landmarks[FACE_LANDMARKS.rightCheek]);
  const faceHeight = distance2d(landmarks[FACE_LANDMARKS.top], landmarks[FACE_LANDMARKS.chin]);
  const eyeSpan = distance2d(landmarks[FACE_LANDMARKS.leftEye], landmarks[FACE_LANDMARKS.rightEye]);
  const mouthWidth = distance2d(landmarks[FACE_LANDMARKS.mouthLeft], landmarks[FACE_LANDMARKS.mouthRight]);
  return (
    cheekWidth > 0.025 &&
    cheekWidth < 0.95 &&
    faceHeight > 0.045 &&
    faceHeight < 1.25 &&
    eyeSpan > 0.02 &&
    eyeSpan < 0.95 &&
    mouthWidth > 0.006
  );
}

function getFreshFaceLandmarks() {
  debugLandmarkReadCount += 1;
  return getFreshFaceResult()?.faceLandmarks?.slice(0, 1) || [];
}

function getLogoOriginForAnchor(scenePoint, logoAnchor, scale, roll) {
  const rotatedAnchor = rotateLogoPoint(logoAnchor, scale, roll);
  return new THREE.Vector3(scenePoint.x - rotatedAnchor.x, scenePoint.y - rotatedAnchor.y, scenePoint.z);
}

function rotateLogoPoint(point, scale, roll) {
  const x = point.x * scale;
  const y = point.y * scale;
  const c = Math.cos(roll);
  const s = Math.sin(roll);
  return new THREE.Vector2(c * x - s * y, s * x + c * y);
}

function getBlinkStrength(landmarks, blendshapes = null) {
  const leftRatio = getEyeOpenRatio(
    landmarks[FACE_LANDMARKS.leftEye],
    landmarks[FACE_LANDMARKS.leftEyeInner],
    landmarks[FACE_LANDMARKS.leftEyeTop],
    landmarks[FACE_LANDMARKS.leftEyeBottom],
  );
  const rightRatio = getEyeOpenRatio(
    landmarks[FACE_LANDMARKS.rightEye],
    landmarks[FACE_LANDMARKS.rightEyeInner],
    landmarks[FACE_LANDMARKS.rightEyeTop],
    landmarks[FACE_LANDMARKS.rightEyeBottom],
  );
  const landmarkBlink = smoothstep(0.135, 0.055, (leftRatio + rightRatio) * 0.5);
  if (!blendshapes) {
    return smoothstep(0.68, 0.94, landmarkBlink);
  }

  const blink = Math.max(getBlendshapeScore(blendshapes, 'eyeBlinkLeft'), getBlendshapeScore(blendshapes, 'eyeBlinkRight'));
  const blinkGate = smoothstep(0.54, 0.78, blink);
  const squint = Math.max(getBlendshapeScore(blendshapes, 'eyeSquintLeft'), getBlendshapeScore(blendshapes, 'eyeSquintRight'));
  const squintGate = smoothstep(0.44, 0.72, squint) * 0.95;
  const landmarkGate = smoothstep(0.48, 0.86, landmarkBlink) * 0.82;
  const wide =
    Math.max(getBlendshapeScore(blendshapes, 'eyeWideLeft'), getBlendshapeScore(blendshapes, 'eyeWideRight')) * 0.28;
  return THREE.MathUtils.clamp(Math.max(landmarkGate, blinkGate, squintGate) - wide, 0, 1);
}

function getEyeIgnitionAmount(blinkStrength, blinkResponse = faceCalibration.blinkResponse) {
  return smoothstep(EYE_IGNITION_START, EYE_IGNITION_END, blinkStrength * blinkResponse);
}

function createTestFaceLandmarks(eyeOpenRatio) {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const setEye = (outerIndex, innerIndex, topIndex, bottomIndex, outerX, innerX, y) => {
    const width = Math.abs(innerX - outerX);
    const halfGap = width * eyeOpenRatio * 0.5;
    landmarks[outerIndex] = { x: outerX, y, z: 0 };
    landmarks[innerIndex] = { x: innerX, y, z: 0 };
    landmarks[topIndex] = { x: (outerX + innerX) * 0.5, y: y - halfGap, z: 0 };
    landmarks[bottomIndex] = { x: (outerX + innerX) * 0.5, y: y + halfGap, z: 0 };
  };
  setEye(FACE_LANDMARKS.leftEye, FACE_LANDMARKS.leftEyeInner, FACE_LANDMARKS.leftEyeTop, FACE_LANDMARKS.leftEyeBottom, 0.36, 0.46, 0.42);
  setEye(FACE_LANDMARKS.rightEye, FACE_LANDMARKS.rightEyeInner, FACE_LANDMARKS.rightEyeTop, FACE_LANDMARKS.rightEyeBottom, 0.64, 0.54, 0.42);
  return landmarks;
}

function getEyeOpenRatio(outer, inner, top, bottom) {
  const width = Math.max(0.001, distance2d(outer, inner));
  return distance2d(top, bottom) / width;
}

function getFaceYaw(landmarks) {
  const leftCheek = faceLandmarkToScene(landmarks[FACE_LANDMARKS.leftCheek]);
  const rightCheek = faceLandmarkToScene(landmarks[FACE_LANDMARKS.rightCheek]);
  const nose = faceLandmarkToScene(landmarks[FACE_LANDMARKS.nose]);
  const cheekMid = leftCheek.clone().add(rightCheek).multiplyScalar(0.5);
  const width = Math.max(0.001, leftCheek.distanceTo(rightCheek));
  return THREE.MathUtils.clamp(((nose.x - cheekMid.x) / width) * 3.4, -1, 1);
}

function getMouthStrength(landmarkMouth, blendshapes = null) {
  if (!blendshapes) {
    return landmarkMouth;
  }

  const jawOpen = getBlendshapeScore(blendshapes, 'jawOpen');
  const mouthFunnel = getBlendshapeScore(blendshapes, 'mouthFunnel');
  const mouthPucker = getBlendshapeScore(blendshapes, 'mouthPucker');
  const lowerDown =
    (getBlendshapeScore(blendshapes, 'mouthLowerDownLeft') + getBlendshapeScore(blendshapes, 'mouthLowerDownRight')) *
    0.5;
  return THREE.MathUtils.clamp(Math.max(landmarkMouth * 0.42, jawOpen * 1.08 + lowerDown * 0.22, mouthFunnel * 0.52 + mouthPucker * 0.28), 0, 1);
}

function getFaceBlendshapeSignals(faceResult) {
  const categories = faceResult?.faceBlendshapes?.[0]?.categories;
  if (!categories?.length) {
    return null;
  }

  const signals = {
    eyeBlinkLeft: 0,
    eyeBlinkRight: 0,
    eyeSquintLeft: 0,
    eyeSquintRight: 0,
    eyeWideLeft: 0,
    eyeWideRight: 0,
    jawOpen: 0,
    mouthFunnel: 0,
    mouthPucker: 0,
    mouthLowerDownLeft: 0,
    mouthLowerDownRight: 0,
  };
  for (const category of categories) {
    const name = category.categoryName || category.displayName;
    if (Object.hasOwn(signals, name)) {
      signals[name] = category.score || 0;
    }
  }
  return signals;
}

function getBlendshapeScore(blendshapes, name) {
  return blendshapes?.[name] || 0;
}

function getFaceMatrixMetrics(faceResult) {
  const matrix = faceResult?.facialTransformationMatrixes?.[0];
  const data = matrix?.data;
  if (!data || data.length < 16) {
    return null;
  }

  const roll = Math.atan2(data[4], data[0]);
  const yaw = THREE.MathUtils.clamp(Math.atan2(-data[8], Math.hypot(data[0], data[4])) / 0.75, -1, 1);
  return { roll, yaw };
}

function withVelocity(key, position, cache, deltaSeconds, filter = null) {
  const safeDelta = Math.max(deltaSeconds, 1 / 90);
  const previous = cache.get(key);
  if (!previous) {
    const state = {
      position: position.clone(),
      velocity: new THREE.Vector2(),
      target: new THREE.Vector3(),
      delta: new THREE.Vector3(),
      rawVelocity: new THREE.Vector2(),
      speed: 0,
    };
    cache.set(key, state);
    return state;
  }

  const isHandFilter = filter?.kind === 'hand';
  const isFaceFilter = filter?.kind === 'face';
  const faceFollow = particleSettings.faceFollow;
  const smoothing = isHandFilter ? particleSettings.handSmoothing : 0.42;
  const maxStepBase = isHandFilter
    ? particleSettings.handMaxStep
    : isFaceFilter
      ? THREE.MathUtils.lerp(0.1, 1.25, faceFollow)
      : 0.28;
  const rawMaxStep = maxStepBase + safeDelta * (isHandFilter ? 5.2 : isFaceFilter ? 14 : 8);
  const previousX = previous.position.x;
  const previousY = previous.position.y;
  previous.delta.copy(position).sub(previous.position);
  const targetDelta = previous.delta;
  const targetDistance = targetDelta.length();
  const jumpContainment = isFaceFilter ? smoothstep(FACE_JUMP_CONTAIN_START, FACE_JUMP_CONTAIN_END, targetDistance) : 0;
  const containedFaceStep =
    isFaceFilter && jumpContainment > 0
      ? THREE.MathUtils.lerp(0.2, 0.62, faceFollow) + safeDelta * 4.5
      : rawMaxStep;
  const maxStep = isFaceFilter
    ? THREE.MathUtils.lerp(rawMaxStep, Math.min(rawMaxStep, containedFaceStep), jumpContainment)
    : rawMaxStep;
  const deadzone = isHandFilter ? particleSettings.handDeadzone : isFaceFilter ? FACE_POSITION_DEADZONE : 0;
  if (deadzone > 0 && targetDistance < deadzone) {
    targetDelta.multiplyScalar(isHandFilter ? 0.14 : 0.04);
  } else {
    targetDelta.clampLength(0, maxStep);
  }
  previous.target.copy(previous.position).add(targetDelta);
  const response = isHandFilter
    ? THREE.MathUtils.lerp(24, 7.5, smoothing)
    : isFaceFilter
      ? THREE.MathUtils.lerp(7, 38, faceFollow)
      : 18;
  const minAlpha = isHandFilter ? 0.06 : isFaceFilter ? THREE.MathUtils.lerp(0.05, 0.18, faceFollow) : 0.18;
  const maxAlpha = isHandFilter ? 0.46 : isFaceFilter ? THREE.MathUtils.lerp(0.18, 0.54, faceFollow) : 0.58;
  const alpha =
    THREE.MathUtils.clamp(safeDelta * response, minAlpha, maxAlpha) *
    (isFaceFilter ? THREE.MathUtils.lerp(1, 0.68, jumpContainment) : 1);
  previous.position.lerp(previous.target, alpha);
  const velocityLimit = isHandFilter
    ? THREE.MathUtils.lerp(8, 4.2, smoothing)
    : isFaceFilter
      ? THREE.MathUtils.lerp(6, 15, faceFollow)
      : 8;
  previous.rawVelocity
    .set((previous.position.x - previousX) / safeDelta, (previous.position.y - previousY) / safeDelta)
    .clampLength(0, velocityLimit);
  const velocityAlpha = isHandFilter
    ? THREE.MathUtils.lerp(0.34, 0.16, smoothing)
    : isFaceFilter
      ? THREE.MathUtils.lerp(0.12, 0.34, faceFollow)
      : 0.28;
  previous.velocity.lerp(previous.rawVelocity, velocityAlpha);
  previous.speed = THREE.MathUtils.clamp(previous.velocity.length() / Math.max(3.5, velocityLimit * 0.86), 0, 1);
  return previous;
}

function updateGustTrigger(handState, elapsedSeconds) {
  if (
    handState.palm.strength > 0.38 &&
    handState.palm.speed > 0.62 &&
    elapsedSeconds - lastGustTime > 0.55
  ) {
    lastGustTime = elapsedSeconds;
    uniforms.uGustTime.value = elapsedSeconds;
    uniforms.uGustOrigin.value.set(handState.palm.position.x, handState.palm.position.y);
    uniforms.uGustVelocity.value.copy(handState.palm.velocity).clampLength(0, 8);
  }
}

function updateShockTrigger(handState, elapsedSeconds) {
  if (handState.centers.length === 2) {
    const distance = handState.centers[0].distanceTo(handState.centers[1]);
    const closingSpeed = lastTwoHandDistance - distance;
    if (
      distance < 1.12 &&
      (closingSpeed > 0.045 || distance < 0.72) &&
      elapsedSeconds - lastClapTime > 1.55
    ) {
      lastClapTime = elapsedSeconds;
      uniforms.uShockTime.value = elapsedSeconds;
      uniforms.uShockCenter.value
        .copy(handState.centers[0])
        .add(handState.centers[1])
        .multiplyScalar(0.5);
    }
    lastTwoHandDistance = distance;
  } else {
    lastTwoHandDistance = 10;
  }
}

function applyHandState(handState) {
  uniforms.uMode.value = 0;
  uniforms.uFace.value.w *= 0.74;
  uniforms.uFaceExpression.value.multiplyScalar(0.78);

  let uploadedForceCount = 0;
  for (let sourceIndex = 0; sourceIndex < MAX_FORCES; sourceIndex += 1) {
    const force = handState.forces[sourceIndex];
    if (force) {
      const current = uniforms.uForce.value[uploadedForceCount];
      const currentVelocity = uniforms.uForceVelocity.value[uploadedForceCount];
      const strength = force.strength * particleSettings.fingerStrength * (reducedMotion ? 0.42 : 1);
      const nextForce = tmpForceUniform.set(force.position.x, force.position.y, strength, force.radius || 0.42);
      if (uploadedForceSourceIds[uploadedForceCount] !== sourceIndex) {
        current.copy(nextForce);
        currentVelocity.copy(force.velocity);
      } else {
        current.lerp(nextForce, 0.52);
        currentVelocity.lerp(force.velocity, 0.42);
      }
      uploadedForceSourceIds[uploadedForceCount] = sourceIndex;
      uploadedForceCount += 1;
    }
  }
  for (let i = uploadedForceCount; i < MAX_FORCES; i += 1) {
    const current = uniforms.uForce.value[i];
    const currentVelocity = uniforms.uForceVelocity.value[i];
    current.z *= 0.74;
    current.w *= 0.72;
    currentVelocity.multiplyScalar(0.7);
    uploadedForceSourceIds[i] = -1;
  }
  uniforms.uForceCount.value = getForceLoopCount(uploadedForceCount);

  const palmStrength =
    handState.palm.strength * particleSettings.palmStrength * (reducedMotion ? 0.35 : 1);
  uniforms.uPalm.value.lerp(
    tmpPalmUniform.set(
      handState.palm.position.x,
      handState.palm.position.y,
      palmStrength,
      handState.palm.depth,
    ),
    0.42,
  );
  uniforms.uPalmVelocity.value.lerp(handState.palm.velocity, 0.36);

  for (let i = 0; i < 2; i += 1) {
    const pinch = handState.pinches[i];
    const current = uniforms.uPinch.value[i];
    if (pinch) {
      const strength = pinch.strength * particleSettings.pinchStrength * (reducedMotion ? 0.42 : 1);
      current.lerp(tmpPinchUniform.set(pinch.position.x, pinch.position.y, strength, pinch.speed), 0.36);
    } else {
      current.z *= 0.74;
      current.w *= 0.72;
    }
  }

  updateInteractionMarkers(handState);
}

function getForceLoopCount(activeCount = 0) {
  let count = activeCount;
  for (let i = activeCount; i < MAX_FORCES; i += 1) {
    if (uniforms.uForce.value[i].z > FORCE_LOOP_EPSILON) {
      count = i + 1;
    }
  }
  return count;
}

function applyFaceState(faceState) {
  const faceFollow = particleSettings.faceFollow;
  const followCurve = Math.pow(faceFollow, 0.55);
  const motionBoost = faceState?.visible ? smoothstep(0.08, 0.62, faceState.speed || 0) : 0;
  const followAlpha =
    faceFollow > 0.98
      ? THREE.MathUtils.lerp(0.56, 0.82, motionBoost)
      : THREE.MathUtils.lerp(0.05, 0.7, followCurve) + motionBoost * 0.16;
  const velocityAlpha =
    faceFollow > 0.98
      ? THREE.MathUtils.lerp(0.56, 0.84, motionBoost)
      : THREE.MathUtils.lerp(0.22, 0.66, followCurve) + motionBoost * 0.14;
  const expressionAlpha =
    faceFollow > 0.98
      ? 0.74
      : THREE.MathUtils.lerp(0.28, 0.68, followCurve) + motionBoost * 0.08;
  uniforms.uMode.value = 1;
  uniforms.uForceCount.value = 0;
  uniforms.uPalm.value.z *= 0.72;
  uniforms.uPalm.value.w *= 0.72;
  uniforms.uPalmVelocity.value.multiplyScalar(0.7);
  for (const force of uniforms.uForce.value) {
    force.z *= 0.72;
    force.w *= 0.72;
  }
  for (const velocity of uniforms.uForceVelocity.value) {
    velocity.multiplyScalar(0.7);
  }
  for (const pinch of uniforms.uPinch.value) {
    pinch.z *= 0.72;
    pinch.w *= 0.72;
  }

  if (faceState.visible) {
    uniforms.uFaceEyeAnchors.value.lerp(faceState.eyeAnchors, followAlpha);
    uniforms.uFace.value.lerp(
      new THREE.Vector4(faceState.center.x, faceState.center.y, faceState.scale, faceState.strength),
      followAlpha,
    );
    uniforms.uFaceVelocity.value.lerp(faceState.velocity, velocityAlpha);
    uniforms.uFaceRotation.value = lerpAngle(uniforms.uFaceRotation.value, faceState.roll, followAlpha);
    uniforms.uFaceExpression.value.lerp(
      new THREE.Vector4(faceState.mouth, faceState.shake.z, faceState.blink, faceState.yaw),
      expressionAlpha,
    );
  } else {
    smoothedFace.strength *= 0.88;
    uniforms.uFace.value.w *= 0.78;
    uniforms.uFaceVelocity.value.multiplyScalar(0.78);
    uniforms.uFaceExpression.value.multiplyScalar(0.78);
  }

  fadeInteractionMarkers();
}

function fadeInteractionMarkers() {
  for (const marker of fingerMarkers) {
    marker.material.opacity *= 0.7;
    marker.visible = marker.material.opacity > 0.02;
  }
  for (const marker of pinchMarkers) {
    marker.material.opacity *= 0.7;
    marker.visible = marker.material.opacity > 0.02;
  }
  palmMarker.material.opacity *= 0.7;
  palmMarker.visible = palmMarker.material.opacity > 0.01;
  shockMarker.material.opacity *= 0.7;
  shockMarker.visible = shockMarker.material.opacity > 0.01;
}

function updateInteractionMarkers(handState) {
  for (let i = 0; i < fingerMarkers.length; i += 1) {
    const marker = fingerMarkers[i];
    const force = handState.forces[i];
    if (force) {
      marker.visible = true;
      marker.position.set(force.position.x, force.position.y, 1.45 + i * 0.006);
      marker.material.opacity = THREE.MathUtils.lerp(marker.material.opacity, 0.5 + force.speed * 0.25, 0.28);
      marker.scale.setScalar((force.radius || 0.4) * (0.82 + force.speed * 0.42));
    } else {
      marker.material.opacity *= 0.74;
      marker.visible = marker.material.opacity > 0.02;
    }
  }

  for (let i = 0; i < pinchMarkers.length; i += 1) {
    const marker = pinchMarkers[i];
    const pinch = handState.pinches[i];
    const strength = uniforms.uPinch.value[i].z;
    if (pinch || strength > 0.02) {
      const position = pinch?.position || uniforms.uPinch.value[i];
      marker.visible = true;
      marker.position.set(position.x, position.y, 1.8);
      marker.rotation.z -= 0.06 + strength * 0.14;
      marker.scale.setScalar(0.55 + strength * 1.15);
      marker.material.opacity = THREE.MathUtils.lerp(marker.material.opacity, strength * 0.82, 0.26);
    } else {
      marker.material.opacity *= 0.74;
      marker.visible = marker.material.opacity > 0.02;
    }
  }

  const palmStrength = uniforms.uPalm.value.z;
  if (palmStrength > 0.02) {
    palmMarker.visible = true;
    palmMarker.position.set(uniforms.uPalm.value.x, uniforms.uPalm.value.y, 1.2);
    palmMarker.material.opacity = Math.min(0.48, palmStrength * 0.24);
    palmMarker.scale.setScalar(2.3 + palmStrength * 3.1);
  } else {
    palmMarker.material.opacity *= 0.72;
    palmMarker.visible = palmMarker.material.opacity > 0.01;
  }

  const shockAge = clock.elapsedTime - uniforms.uShockTime.value;
  if (shockAge > 0 && shockAge < 1.5) {
    shockMarker.visible = true;
    shockMarker.position.set(uniforms.uShockCenter.value.x, uniforms.uShockCenter.value.y, 2.1);
    shockMarker.scale.setScalar(0.35 + shockAge * 4.8);
    shockMarker.material.opacity = Math.pow(1 - shockAge / 1.5, 1.4) * 0.82;
  } else {
    shockMarker.material.opacity *= 0.7;
    shockMarker.visible = shockMarker.material.opacity > 0.01;
  }
}

function landmarkToScene(landmark) {
  return landmarkToSceneInto(landmark, new THREE.Vector3());
}

function landmarkToSceneInto(landmark, out) {
  return out.set((0.5 - landmark.x) * 8.2, (0.5 - landmark.y) * 6.2, 1);
}

function landmarkPairMidpointToScene(first, second) {
  return landmarkPairMidpointToSceneInto(first, second, new THREE.Vector3());
}

function landmarkPairMidpointToSceneInto(first, second, out) {
  return out.set((0.5 - (first.x + second.x) * 0.5) * 8.2, (0.5 - (first.y + second.y) * 0.5) * 6.2, 1);
}

function faceLandmarkToScene(landmark, planeZ = 2.05) {
  const videoWidth = video.videoWidth || cameraSettings.width || 16;
  const videoHeight = video.videoHeight || cameraSettings.height || 9;
  const videoAspect = videoWidth / Math.max(1, videoHeight);
  const viewportAspect = window.innerWidth / Math.max(1, window.innerHeight);
  const mirroredX = 1 - landmark.x;
  let screenX = mirroredX;
  let screenY = landmark.y;

  if (videoAspect > viewportAspect) {
    const coveredWidth = videoAspect / viewportAspect;
    screenX = mirroredX * coveredWidth - (coveredWidth - 1) * 0.5;
  } else {
    const coveredHeight = viewportAspect / videoAspect;
    screenY = landmark.y * coveredHeight - (coveredHeight - 1) * 0.5;
  }

  const distance = Math.max(0.1, camera.position.z - planeZ);
  const visibleHeight = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5) * distance;
  const visibleWidth = visibleHeight * camera.aspect;
  return new THREE.Vector3((screenX - 0.5) * visibleWidth, (0.5 - screenY) * visibleHeight, planeZ);
}

function getPalmCenter(landmarks) {
  return getPalmCenterInto(landmarks, new THREE.Vector3());
}

function getPalmCenterInto(landmarks, out) {
  out.set(0, 0, 0);
  addWeightedLandmarkToScene(out, landmarks[0], 0.9);
  addWeightedLandmarkToScene(out, landmarks[5], 1);
  addWeightedLandmarkToScene(out, landmarks[9], 1.12);
  addWeightedLandmarkToScene(out, landmarks[13], 1);
  addWeightedLandmarkToScene(out, landmarks[17], 1);
  return out.multiplyScalar(1 / 5.02);
}

function addWeightedLandmarkToScene(out, landmark, weight) {
  out.x += (0.5 - landmark.x) * 8.2 * weight;
  out.y += (0.5 - landmark.y) * 6.2 * weight;
  out.z += weight;
}

function getHandMetrics(landmarks, worldLandmarks = null, confidence = 1) {
  const palmWidth = Math.max(0.001, distance2d(landmarks[5], landmarks[17]));
  const palmLength = Math.max(0.001, distance2d(landmarks[0], landmarks[9]));
  const middleReach = Math.max(0.001, distance2d(landmarks[0], landmarks[12]));
  const palmAspect = palmWidth / palmLength;
  const faceOn = smoothstep(0.58, 1.02, palmAspect);
  const screenDepth = smoothstep(0.075, 0.205, palmWidth);
  const worldPalmWidth = worldLandmarks ? distance3d(worldLandmarks[5], worldLandmarks[17]) : 0;
  const worldDepth = worldPalmWidth > 0 ? smoothstep(0.055, 0.115, worldPalmWidth) : screenDepth;
  const fingerReach = smoothstep(1.62, 2.78, middleReach / palmWidth);
  const quality =
    THREE.MathUtils.clamp(confidence, 0, 1) *
    THREE.MathUtils.clamp(0.5 + faceOn * 0.36 + fingerReach * 0.22, 0, 1);

  return {
    palmWidth,
    palmLength,
    faceOn,
    depth: screenDepth * 0.72 + worldDepth * 0.28,
    quality,
  };
}

function getPinch(
  landmarks,
  handIndex,
  deltaSeconds,
  metrics = getHandMetrics(landmarks),
  fingerScores = getFingerExtensionScores(landmarks, metrics),
) {
  const thumb = landmarks[4];
  const index = landmarks[8];
  const distance = Math.hypot(thumb.x - index.x, thumb.y - index.y, (thumb.z - index.z) * 0.4);
  const normalizedDistance = distance / Math.max(metrics.palmWidth, 0.001);
  const indexExtension = fingerScores[1]?.strength ?? 0;
  const strength = smoothstep(0.62, 0.28, normalizedDistance) * (0.58 + indexExtension * 0.42);
  const scenePoint =
    handIndex === null
      ? landmarkPairMidpointToScene(thumb, index)
      : landmarkPairMidpointToSceneInto(thumb, index, handScratchPinchPoints[handIndex]);
  const point =
    handIndex === null
      ? { position: scenePoint, velocity: new THREE.Vector2(0, 0), speed: 0 }
      : withVelocity(`pinch:${handIndex}`, scenePoint, previousPinches, deltaSeconds, HAND_FILTER);
  return {
    position: point.position,
    velocity: point.velocity,
    speed: point.speed,
    strength,
  };
}

function getHandDepth(landmarks) {
  const palmWidth = distance2d(landmarks[5], landmarks[17]);
  return smoothstep(0.08, 0.2, palmWidth);
}

function getOpenPalmStrength(landmarks, fingerScores = getFingerExtensionScores(landmarks), metrics = getHandMetrics(landmarks)) {
  let longFingerScore = 0;
  let averageLongTipReach = 0;
  const wrist = landmarks[0];
  const palmWidth = metrics.palmWidth + 0.001;
  for (const finger of fingerScores) {
    if (finger.tip !== 4) {
      longFingerScore += finger.strength;
      averageLongTipReach += distance2d(wrist, landmarks[finger.tip]);
    }
  }
  longFingerScore /= LONG_FINGER_TIPS.length;
  averageLongTipReach /= LONG_FINGER_TIPS.length;
  const spread = distance2d(landmarks[8], landmarks[20]) / palmWidth;
  const reachScore = smoothstep(1.52, 2.62, averageLongTipReach / palmWidth);
  const spreadScore = smoothstep(1.02, 2.18, spread);
  const fingerScore = smoothstep(0.48, 0.82, longFingerScore);
  return THREE.MathUtils.clamp((fingerScore * 0.62 + reachScore * 0.2 + spreadScore * 0.18) * (0.72 + metrics.faceOn * 0.28), 0, 1);
}

function getInteractiveFingers(fingerScores, openPalm, pinchStrength = 0, out = []) {
  out.length = 0;
  if (openPalm > 0.68) {
    return out;
  }
  if (openPalm > 0.58) {
    for (const finger of fingerScores) {
      if (finger.tip !== 4 && finger.strength > 0.48 && finger.stability > 0.42) {
        finger.radius = 0.44 + finger.strength * 0.24;
        out.push(finger);
        if (out.length >= 2) {
          break;
        }
      }
    }
    return out;
  }

  const indexFinger = fingerScores[1];
  if (
    indexFinger?.strength > 0.28 &&
    indexFinger.stability > 0.34 &&
    pinchStrength < 0.72
  ) {
    indexFinger.radius = 0.34 + indexFinger.strength * 0.16;
    out.push(indexFinger);
    return out;
  }

  if (pinchStrength < 0.42) {
    for (const finger of fingerScores) {
      if (finger.tip === 4) {
        continue;
      }
      if (finger.strength > 0.42 && finger.stability > 0.36) {
        finger.radius = 0.32 + finger.strength * 0.14;
        out.push(finger);
        if (out.length >= 2) {
          break;
        }
      }
    }
  }
  return out;
}

function getFingerExtensionScores(landmarks, metrics = getHandMetrics(landmarks), out = null) {
  const wrist = landmarks[0];
  const scores = out || FINGER_CHAINS.map(({ tip }) => ({ tip, strength: 0, stability: 0, radius: 0 }));
  for (let index = 0; index < FINGER_CHAINS.length; index += 1) {
    const { tip, dip, pip, mcp } = FINGER_CHAINS[index];
    const tipReach = distance2d(wrist, landmarks[tip]);
    const dipReach = distance2d(wrist, landmarks[dip]);
    const pipReach = distance2d(wrist, landmarks[pip]);
    const dipRatio = tipReach / Math.max(dipReach, 0.001);
    const pipRatio = tipReach / Math.max(pipReach, 0.001);
    const straightness = getJointStraightness(landmarks[mcp], landmarks[pip], landmarks[tip]);
    const reach = smoothstep(tip === 4 ? 1.06 : 1.24, tip === 4 ? 1.72 : 2.22, tipReach / metrics.palmWidth);
    const score = scores[index];
    score.tip = tip;
    score.strength =
      smoothstep(1.0, 1.1, dipRatio) *
      smoothstep(1.04, 1.18, pipRatio) *
      (0.54 + straightness * 0.3 + reach * 0.16);
  }
  return scores;
}

function animate(frameTime = 0) {
  requestAnimationFrame(animate);
  const renderTimestamp = frameTime || performance.now();
  if (lastRenderAt && renderTimestamp - lastRenderAt < MIN_RENDER_FRAME_MS - FRAME_SKIP_EPSILON_MS) {
    return;
  }

  lastRenderAt = renderTimestamp;
  const delta = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;
  updateRenderPerformance(delta);
  uniforms.uTime.value = elapsed;
  uniforms.uIntro.value = Math.min(1, uniforms.uIntro.value + delta * 0.36);

  if (trackingMode === TRACKING_MODES.FACE) {
    const faceState = readFaceFrame(delta);
    applyFaceState(faceState);
    drawTrackingDebug(null, faceState);
  } else {
    const handState = readHandFrame(elapsed, delta);
    applyHandState(handState);
    drawTrackingDebug(handState, null);
  }

  if (points) {
    if (trackingMode === TRACKING_MODES.FACE) {
      points.rotation.x = THREE.MathUtils.lerp(points.rotation.x, 0, 0.1);
      points.rotation.y = THREE.MathUtils.lerp(points.rotation.y, 0, 0.1);
      points.rotation.z = THREE.MathUtils.lerp(points.rotation.z, 0, 0.1);
    } else {
      points.rotation.x = THREE.MathUtils.lerp(points.rotation.x, -0.05, 0.08);
      points.rotation.y = Math.sin(elapsed * 0.16) * 0.07;
      points.rotation.z = Math.sin(elapsed * 0.1) * 0.016;
    }
  }

  if (starField) {
    starField.rotation.y += delta * 0.008;
    starField.rotation.x = Math.sin(elapsed * 0.04) * 0.02;
  }

  composer.render();
}

function handleResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const distanceForLogoWidth = 8.6 / (2 * Math.tan(fov * 0.5) * camera.aspect);
  camera.position.z = Math.max(12, distanceForLogoWidth);
  camera.updateProjectionMatrix();
  applyRendererSize();
}

function updateRenderPerformance(deltaSeconds) {
  const instantFps = Math.min(FPS_DISPLAY_CAP, 1 / Math.max(deltaSeconds, 1 / 120));
  renderStats.fps = THREE.MathUtils.lerp(renderStats.fps || instantFps, instantFps, 0.08);
  const pressureTarget =
    renderStats.fps < 48
      ? 1
      : renderStats.fps < 54
        ? 0.64
        : renderStats.fps < 58
          ? 0.26
          : 0;
  renderStats.pressure = THREE.MathUtils.lerp(
    renderStats.pressure,
    pressureTarget,
    pressureTarget > renderStats.pressure ? 0.22 : 0.08,
  );
  renderStats.frameCount += 1;

  const now = performance.now();
  if (now - renderStats.lastSecondAt > 500) {
    perfStatus.textContent = `${Math.min(FPS_DISPLAY_CAP, Math.round(renderStats.fps))} FPS`;
    renderStats.lastSecondAt = now;
  }

  if (qualityMode !== 'auto' || now - renderStats.lastQualityAt < 900) {
    return;
  }

  renderStats.lastQualityAt = now;
  const maxRatio = Math.min(window.devicePixelRatio || 1, 1.65);
  const maxParticleCount = qualityMode === 'high' ? PARTICLE_COUNT : getParticleBudget();
  const previousRatio = renderPixelRatio;
  const previousParticleCount = activeParticleCount;
  const adaptivePressure = getAdaptivePressure();
  if (adaptivePressure > 0.52) {
    const dropScale = adaptivePressure > 0.85 ? 1.25 : 1;
    renderPixelRatio = Math.max(0.68, renderPixelRatio - 0.16 * dropScale);
    activeParticleCount = Math.max(MIN_PARTICLE_COUNT, activeParticleCount - Math.round(10_000 * dropScale));
  } else if (renderStats.fps > 59 && adaptivePressure < 0.08) {
    if (renderPixelRatio < maxRatio) {
      renderPixelRatio = Math.min(maxRatio, renderPixelRatio + 0.06);
    }
    activeParticleCount = Math.min(maxParticleCount, activeParticleCount + 3_000);
  }

  if (Math.abs(previousRatio - renderPixelRatio) > 0.01) {
    applyRendererSize();
  }
  if (points && previousParticleCount !== activeParticleCount) {
    points.geometry.setDrawRange(0, activeParticleCount);
  }
}

function getEffectiveTrackingFps() {
  const configuredFps = Math.max(1, particleSettings.trackingFps);
  if (qualityMode === 'high') {
    return configuredFps;
  }
  const pressure = getAdaptivePressure();
  return Math.round(THREE.MathUtils.lerp(configuredFps, Math.min(configuredFps, MIN_ADAPTIVE_TRACKING_FPS), pressure));
}

function getAdaptivePressure() {
  return THREE.MathUtils.clamp(Math.max(renderStats.pressure, trackingStats.pressure), 0, 1);
}

function getInitialPixelRatio() {
  const deviceRatio = window.devicePixelRatio || 1;
  return qualityMode === 'high' ? Math.min(deviceRatio, 2) : Math.min(deviceRatio, 1.25);
}

function getParticleBudget() {
  const density = particleSettings?.particleDensity ?? DEFAULT_PARTICLE_COUNT / PARTICLE_COUNT;
  return Math.round(THREE.MathUtils.clamp(density, 0.18, 1) * PARTICLE_COUNT);
}

function applyRendererSize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setPixelRatio(renderPixelRatio);
  renderer.setSize(width, height);
  if (typeof composer.setPixelRatio === 'function') {
    composer.setPixelRatio(renderPixelRatio);
  }
  composer.setSize(width, height);
  uniforms.uPixelRatio.value = renderPixelRatio;
  if (bloomPass) {
    bloomPass.strength = getBloomStrength();
    bloomPass.radius = getBloomRadius();
    bloomPass.threshold = getBloomThreshold();
    bloomPass.enabled = particleSettings.bloom > 0.001;
  }
}

function handlePointerMove(event) {
  if (!pointerEnabled || trackingMode !== TRACKING_MODES.HAND) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  pointer.active = true;
  pointer.x = ((event.clientX - rect.left) / rect.width - 0.5) * 8.2;
  pointer.y = (0.5 - (event.clientY - rect.top) / rect.height) * 6.2;
  pointer.strength = event.buttons ? 1 : 0.66;
}

function handlePointerLeave() {
  pointer.active = false;
  pointer.strength = 0;
}

function togglePointerInput() {
  pointerEnabled = !pointerEnabled;
  pointer.strength = 0;
  pointer.active = false;
  updatePointerToggle();
}

function updatePointerToggle() {
  const handMode = trackingMode === TRACKING_MODES.HAND;
  pointerToggle.hidden = !handMode;
  pointerToggle.setAttribute('aria-hidden', String(!handMode));
  if (!handMode) {
    pointer.active = false;
    pointer.strength = 0;
  }
  pointerToggle.setAttribute('aria-pressed', String(pointerEnabled));
  pointerToggle.textContent = pointerEnabled ? 'MOUSE ON' : 'MOUSE OFF';
}

function toggleDebugPanel() {
  debugEnabled = !debugEnabled;
  updateDebugToggle();
}

function updateDebugToggle() {
  debugToggle.setAttribute('aria-pressed', String(debugEnabled));
  debugToggle.textContent = debugEnabled ? 'DEBUG ON' : 'DEBUG OFF';
  handDebugPanel.hidden = !debugEnabled;
}

function toggleCalibrationPanel() {
  calibrationEnabled = !calibrationEnabled;
  if (calibrationEnabled) {
    setTrackingMode(TRACKING_MODES.FACE);
  }
  updateCalibrationToggle();
}

function updateCalibrationToggle() {
  const faceMode = trackingMode === TRACKING_MODES.FACE;
  const active = calibrationEnabled && faceMode;
  calibrationToggle.setAttribute('aria-pressed', String(active));
  calibrationToggle.setAttribute('aria-disabled', String(!faceMode));
  calibrationToggle.textContent = active ? 'TUNE ON' : 'TUNE OFF';
  calibrationPanel.hidden = !active;
  for (const input of calibrationControls.querySelectorAll('input')) {
    input.disabled = !active;
  }
}

function toggleSettingsPanel() {
  settingsEnabled = !settingsEnabled;
  updateSettingsToggle();
}

function updateSettingsToggle() {
  settingsToggle.setAttribute('aria-pressed', String(settingsEnabled));
  settingsToggle.textContent = settingsEnabled ? 'SETTINGS ON' : 'SETTINGS OFF';
  settingsPanel.hidden = !settingsEnabled;
}

function setupCalibrationControls() {
  calibrationControls.innerHTML = '';
  for (const control of CALIBRATION_CONTROLS) {
    const row = document.createElement('label');
    row.className = 'calibration-row';
    row.htmlFor = `calibration-${control.key}`;

    const name = document.createElement('span');
    name.textContent = control.label;

    const input = document.createElement('input');
    input.id = `calibration-${control.key}`;
    input.type = 'range';
    input.min = String(control.min);
    input.max = String(control.max);
    input.step = String(control.step);
    input.value = String(faceCalibration[control.key]);
    input.dataset.key = control.key;

    const value = document.createElement('output');
    value.htmlFor = input.id;
    value.textContent = formatCalibrationValue(faceCalibration[control.key]);
    value.dataset.key = `${control.key}:value`;

    input.addEventListener('input', () => {
      faceCalibration[control.key] = Number(input.value);
      value.textContent = formatCalibrationValue(faceCalibration[control.key]);
      applyFaceCalibration();
      saveFaceCalibration();
      updateCalibrationValues();
    });

    row.append(name, input, value);
    calibrationControls.append(row);
  }

  calibrationValues.addEventListener('focus', () => calibrationValues.select());
  updateCalibrationValues();
}

function applyFaceCalibration() {
  uniforms.uEyeCalib.value.set(
    faceCalibration.eyeOriginX,
    faceCalibration.eyeOriginY,
    faceCalibration.eyeScale,
    faceCalibration.eyeSpread,
  );
  uniforms.uEyeFineTune.value.set(
    faceCalibration.eyeHeight,
    faceCalibration.blinkResponse,
    faceCalibration.eyeShape,
    faceCalibration.eyeIntensity,
  );
}

function resetFaceCalibration() {
  faceCalibration = { ...DEFAULT_FACE_CALIBRATION };
  saveFaceCalibration();
  applyFaceCalibration();
  for (const control of CALIBRATION_CONTROLS) {
    const input = document.querySelector(`#calibration-${control.key}`);
    const output = calibrationControls.querySelector(`[data-key="${control.key}:value"]`);
    if (input) {
      input.value = String(faceCalibration[control.key]);
    }
    if (output) {
      output.textContent = formatCalibrationValue(faceCalibration[control.key]);
    }
  }
  updateCalibrationValues();
}

function loadFaceCalibration() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(CALIBRATION_STORAGE_KEY) || '{}');
    return normalizeFaceCalibration(saved);
  } catch {
    return { ...DEFAULT_FACE_CALIBRATION };
  }
}

function normalizeFaceCalibration(values = {}) {
  const normalized = { ...DEFAULT_FACE_CALIBRATION };
  for (const control of CALIBRATION_CONTROLS) {
    const value = Number(values[control.key]);
    if (Number.isFinite(value)) {
      normalized[control.key] = THREE.MathUtils.clamp(value, control.min, control.max);
    }
  }
  return normalized;
}

function saveFaceCalibration() {
  try {
    window.localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(getRoundedFaceCalibration()));
  } catch {
    // Storage can be blocked in private contexts; sliders still work for this session.
  }
}

function updateCalibrationValues() {
  calibrationValues.value = JSON.stringify(getRoundedFaceCalibration(), null, 2);
}

function getRoundedFaceCalibration() {
  return Object.fromEntries(
    CALIBRATION_CONTROLS.map(({ key }) => [key, Number(faceCalibration[key].toFixed(3))]),
  );
}

function formatCalibrationValue(value) {
  return Number(value).toFixed(2);
}

function setupParticleSettingsControls() {
  settingsControls.innerHTML = '';
  for (const sectionName of SETTING_SECTIONS) {
    const section = document.createElement('section');
    section.className = 'settings-section';
    section.dataset.section = sectionName.toLowerCase();

    const heading = document.createElement('div');
    heading.className = 'settings-section-title';
    heading.textContent = sectionName;
    section.append(heading);

    for (const control of PARTICLE_SETTING_CONTROLS.filter((item) => item.section === sectionName)) {
      section.append(createParticleSettingRow(control));
    }

    settingsControls.append(section);
  }

  settingsValues.addEventListener('focus', () => settingsValues.select());
  updateSettingsModeState();
  updateParticleSettingsValues();
}

function createParticleSettingRow(control) {
  const row = document.createElement('div');
  row.className = 'settings-row';
  row.dataset.section = control.section.toLowerCase();

  const name = document.createElement('label');
  name.textContent = control.label;
  name.htmlFor = `setting-${control.key}`;

  const input = document.createElement('input');
  input.id = `setting-${control.key}`;
  input.type = 'range';
  input.min = String(control.min);
  input.max = String(control.max);
  input.step = String(control.step);
  input.value = String(particleSettings[control.key]);
  input.dataset.key = control.key;

  const value = document.createElement('output');
  value.htmlFor = input.id;
  value.textContent = formatParticleSettingValue(particleSettings[control.key], control);
  value.dataset.key = `${control.key}:value`;

  const help = document.createElement('button');
  help.className = 'settings-help';
  help.type = 'button';
  help.textContent = '?';
  const tooltipId = `setting-${control.key}-tip`;
  help.setAttribute('aria-label', `${control.label}: ${control.hint}`);
  help.setAttribute('aria-describedby', tooltipId);
  help.addEventListener('click', (event) => {
    event.preventDefault();
    help.focus();
  });

  const tooltip = document.createElement('div');
  tooltip.id = tooltipId;
  tooltip.className = 'settings-tooltip';
  tooltip.textContent = control.hint;

  input.addEventListener('input', () => {
    particleSettings[control.key] = Number(input.value);
    value.textContent = formatParticleSettingValue(particleSettings[control.key], control);
    applyParticleSettings();
    saveParticleSettings();
    updateParticleSettingsValues();
  });

  row.append(name, input, value, help, tooltip);
  return row;
}

function applyParticleSettings() {
  uniforms.uMotionSettings.value.set(
    particleSettings.particleForce,
    particleSettings.particleCurl,
    particleSettings.particleDepth,
    particleSettings.idleMotion,
  );
  uniforms.uHandTuning.value.set(
    particleSettings.handContactRadius,
    particleSettings.handPalmRadius,
    particleSettings.handPinchReach,
    particleSettings.handWake,
  );
  uniforms.uVisualSettings.value.set(
    particleSettings.bloom,
    particleSettings.faceFollow,
    particleSettings.faceMotion,
    0,
  );
  app.style.setProperty('--face-camera-opacity', particleSettings.faceCameraOpacity.toFixed(3));

  if (bloomPass) {
    bloomPass.strength = getBloomStrength();
    bloomPass.radius = getBloomRadius();
    bloomPass.threshold = getBloomThreshold();
    bloomPass.enabled = particleSettings.bloom > 0.001;
  }

  if (qualityMode !== 'high') {
    activeParticleCount = Math.min(activeParticleCount, getParticleBudget());
  }
  if (points) {
    points.geometry.setDrawRange(0, activeParticleCount);
  }
}

function updateSettingsModeState() {
  const faceMode = trackingMode === TRACKING_MODES.FACE;
  for (const section of settingsControls.querySelectorAll('.settings-section')) {
    const sectionName = section.dataset.section;
    const inactive = (sectionName === 'hand' && faceMode) || (sectionName === 'face' && !faceMode);
    section.classList.toggle('is-inactive', inactive);
    section.setAttribute('aria-disabled', String(inactive));
    for (const input of section.querySelectorAll('input')) {
      input.disabled = inactive;
    }
  }
}

function resetParticleSettings() {
  particleSettings = { ...DEFAULT_PARTICLE_SETTINGS };
  saveParticleSettings();
  applyParticleSettings();
  for (const control of PARTICLE_SETTING_CONTROLS) {
    const input = document.querySelector(`#setting-${control.key}`);
    const output = settingsControls.querySelector(`[data-key="${control.key}:value"]`);
    if (input) {
      input.value = String(particleSettings[control.key]);
    }
    if (output) {
      output.textContent = formatParticleSettingValue(particleSettings[control.key], control);
    }
  }
  updateParticleSettingsValues();
}

function loadParticleSettings() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(PARTICLE_SETTINGS_STORAGE_KEY) || '{}');
    return normalizeParticleSettings(saved);
  } catch {
    return { ...DEFAULT_PARTICLE_SETTINGS };
  }
}

function normalizeParticleSettings(values = {}) {
  const normalized = { ...DEFAULT_PARTICLE_SETTINGS };
  for (const control of PARTICLE_SETTING_CONTROLS) {
    const value = Number(values[control.key]);
    if (Number.isFinite(value)) {
      normalized[control.key] = THREE.MathUtils.clamp(value, control.min, control.max);
    }
  }
  return normalized;
}

function saveParticleSettings() {
  try {
    window.localStorage.setItem(PARTICLE_SETTINGS_STORAGE_KEY, JSON.stringify(getRoundedParticleSettings()));
  } catch {
    // Storage can be blocked in private contexts; sliders still work for this session.
  }
}

function updateParticleSettingsValues() {
  settingsValues.value = JSON.stringify(getRoundedParticleSettings(), null, 2);
}

function getRoundedParticleSettings() {
  return Object.fromEntries(
    PARTICLE_SETTING_CONTROLS.map((control) => [
      control.key,
      Number(particleSettings[control.key].toFixed(control.precision ?? 2)),
    ]),
  );
}

function formatParticleSettingValue(value, control) {
  return Number(value).toFixed(control.precision ?? 2);
}

function getBloomStrength() {
  const baseStrength = renderPixelRatio < 1 ? 0.65 : 0.9;
  const normalized = THREE.MathUtils.clamp(particleSettings.bloom / 1.4, 0, 1);
  return particleSettings.bloom > 0.001 ? baseStrength * (0.18 + Math.pow(normalized, 1.25) * 2.15) : 0;
}

function getBloomRadius() {
  const normalized = THREE.MathUtils.clamp(particleSettings.bloom / 1.4, 0, 1);
  return THREE.MathUtils.lerp(0.18, renderPixelRatio < 1 ? 0.58 : 0.82, normalized);
}

function getBloomThreshold() {
  const normalized = THREE.MathUtils.clamp(particleSettings.bloom / 1.4, 0, 1);
  return THREE.MathUtils.lerp(0.54, 0.0, normalized);
}

function setTrackingMode(mode) {
  if (mode === trackingMode) {
    return;
  }

  trackingMode = mode;
  if (mode !== TRACKING_MODES.FACE) {
    calibrationEnabled = false;
  }
  lastTrackingAt = 0;
  lastTrackingErrorAt = 0;
  lastProcessedVideoTime = -1;
  resetHandTrackingCaches();
  resetFaceTrackingCaches();
  resetTrackingHealth(mode);
  resetHandUniforms();
  resetFaceUniforms();
  updateModeControls();
  updateCalibrationToggle();
  updateSettingsModeState();
  setCameraState('LIVE', '');
  requestLandmarkerForMode();
}

function updateModeControls() {
  const faceMode = trackingMode === TRACKING_MODES.FACE;
  app.dataset.mode = trackingMode;
  uniforms.uMode.value = faceMode ? 1 : 0;
  handModeButton.setAttribute('aria-pressed', String(!faceMode));
  faceModeButton.setAttribute('aria-pressed', String(faceMode));
  handDebugPanel.setAttribute('aria-label', faceMode ? 'Camera face tracking debug' : 'Camera hand tracking debug');
  updatePointerToggle();
  updateSettingsModeState();
}

function resetHandUniforms() {
  uniforms.uForceCount.value = 0;
  uniforms.uPalm.value.set(0, 0, 0, 0);
  uniforms.uPalmVelocity.value.set(0, 0);
  for (const force of uniforms.uForce.value) {
    force.set(0, 0, 0, 0);
  }
  for (const velocity of uniforms.uForceVelocity.value) {
    velocity.set(0, 0);
  }
  uploadedForceSourceIds.fill(-1);
  for (const pinch of uniforms.uPinch.value) {
    pinch.set(0, 0, 0, 0);
  }
}

function resetFaceUniforms() {
  uniforms.uFace.value.set(0, 0, 0.36, 0);
  uniforms.uFaceVelocity.value.set(0, 0);
  uniforms.uFaceRotation.value = 0;
  uniforms.uFaceExpression.value.set(0, 0, 0, 0);
  uniforms.uFaceEyeAnchors.value.copy(DEFAULT_FACE_EYE_ANCHORS);
  smoothedFace.scale = 0.36;
  smoothedFace.roll = 0;
  smoothedFace.mouth = 0;
  smoothedFace.blink = 0;
  smoothedFace.yaw = 0;
  smoothedFace.shakeX = 0;
  smoothedFace.shakeY = 0;
  smoothedFace.shake = 0;
  smoothedFace.eyeAnchors.copy(DEFAULT_FACE_EYE_ANCHORS);
  smoothedFace.strength = 0;
}

function setCameraState(text, tone) {
  cameraMode = text;
  cameraStatus.textContent = text;
  const negotiatedFpsWarning = cameraSettings.frameRate > 0 && cameraSettings.frameRate < 50;
  const statusTone = tone || (negotiatedFpsWarning ? 'warn' : '');
  cameraStatus.className = `status ${statusTone}`.trim();
  handDebugMeta.textContent = text;
}

function drawTrackingDebug(handState, faceState) {
  const handMode = trackingMode === TRACKING_MODES.HAND;
  const handCount = handMode ? handState?.activeHands || 0 : 0;
  const liveHandCount = handMode ? handState?.liveHands || 0 : 0;
  const heldHandCount = handMode ? handState?.heldHands || 0 : 0;
  const faceCount = !handMode && faceState?.visible ? 1 : 0;

  if (handMode && liveHandCount) {
    setCameraState(`${liveHandCount} HAND${liveHandCount > 1 ? 'S' : ''}`, '');
  } else if (handMode && heldHandCount) {
    setCameraState('HELD HAND', 'warn');
  } else if (!handMode && faceCount) {
    setCameraState(faceState.held ? 'HELD FACE' : '1 FACE', faceState.held ? 'warn' : '');
  } else if (cameraMode !== 'POINTER' && cameraMode !== 'LOADING') {
    setCameraState('SEARCH', 'warn');
  }

  if (handMode && handState) {
    let strongestPinch = 0;
    for (const pinch of handState.pinches) {
      strongestPinch = Math.max(strongestPinch, pinch?.strength || 0);
    }
    handDebugStats.openPalm = THREE.MathUtils.lerp(handDebugStats.openPalm, handState.palm.strength, 0.18);
    handDebugStats.pinch = THREE.MathUtils.lerp(handDebugStats.pinch, strongestPinch, 0.22);
    handDebugStats.forces = THREE.MathUtils.lerp(handDebugStats.forces, handState.activeForceCount, 0.2);
    handDebugStats.quality = THREE.MathUtils.lerp(handDebugStats.quality, handState.quality, 0.22);
    handDebugStats.activeHands = THREE.MathUtils.lerp(handDebugStats.activeHands, handState.activeHands, 0.22);
    handDebugHands.textContent = `${handCount} HAND${handCount === 1 ? '' : 'S'} Q${handDebugStats.quality.toFixed(2)}`;
    handDebugGesture.textContent =
      handDebugStats.pinch > 0.08
        ? `PINCH ${handDebugStats.pinch.toFixed(2)}`
        : handDebugStats.openPalm > 0.58
          ? `PALM ${handDebugStats.openPalm.toFixed(2)}`
          : handDebugStats.forces > 0.2
            ? `FORCE ${Math.round(handDebugStats.forces)}`
            : `TRACK ${handDebugStats.activeHands.toFixed(1)}`;
  } else {
    handDebugStats.openPalm = THREE.MathUtils.lerp(handDebugStats.openPalm, faceState?.strength || 0, 0.18);
    handDebugStats.pinch = THREE.MathUtils.lerp(handDebugStats.pinch, faceState?.mouth || 0, 0.2);
    handDebugHands.textContent = `${faceCount} FACE${faceCount === 1 ? '' : 'S'}`;
    handDebugGesture.textContent =
      faceState?.visible && faceState.blink > 0.28
        ? `BLINK ${faceState.blink.toFixed(2)}`
        : faceState?.visible && faceState.mouth > 0.1
        ? `MOUTH ${faceState.mouth.toFixed(2)}`
        : `LOCK ${handDebugStats.openPalm.toFixed(2)}`;
  }

  const detectionFps = Math.min(FPS_DISPLAY_CAP, Math.round(handDebugStats.fps));
  const inferenceMs = handDebugStats.inferenceMs > 0 ? ` ${handDebugStats.inferenceMs.toFixed(1)}MS` : '';
  const healthSuffix = getTrackingHealthSuffix();
  handDebugFps.textContent = cameraSettings.frameRate
    ? `${detectionFps}D ${Math.round(cameraSettings.frameRate)}C${inferenceMs}${healthSuffix}`
    : `${detectionFps} FPS${inferenceMs}${healthSuffix}`;

  if (!debugEnabled) {
    return;
  }

  const handLandmarks = handMode ? getFreshLandmarks() : [];
  const faceLandmarks = handMode ? [] : getFreshFaceLandmarks();
  resizeHandDebugCanvas();
  const width = handOverlay.width;
  const height = handOverlay.height;
  handDebugContext.clearRect(0, 0, width, height);
  handDebugContext.lineCap = 'round';
  handDebugContext.lineJoin = 'round';
  const videoRect = getDebugVideoRect(width, height);

  if (handMode) {
    for (const landmarks of handLandmarks.slice(0, 2)) {
      drawHandConnections(landmarks, videoRect);
      drawHandPoints(landmarks, videoRect);
    }
  } else if (faceLandmarks[0]) {
    drawFaceDebug(faceLandmarks[0], videoRect);
  }
}

function resizeHandDebugCanvas() {
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(handOverlay.clientWidth * pixelRatio));
  const height = Math.max(1, Math.round(handOverlay.clientHeight * pixelRatio));
  if (handOverlay.width !== width || handOverlay.height !== height) {
    handOverlay.width = width;
    handOverlay.height = height;
  }
}

function getDebugVideoRect(width, height) {
  const videoWidth = video.videoWidth || width;
  const videoHeight = video.videoHeight || height;
  const videoAspect = videoWidth / videoHeight;
  const panelAspect = width / height;

  if (videoAspect > panelAspect) {
    const fittedHeight = width / videoAspect;
    return {
      x: 0,
      y: (height - fittedHeight) * 0.5,
      width,
      height: fittedHeight,
    };
  }

  const fittedWidth = height * videoAspect;
  return {
    x: (width - fittedWidth) * 0.5,
    y: 0,
    width: fittedWidth,
    height,
  };
}

function drawHandConnections(landmarks, videoRect) {
  handDebugContext.strokeStyle = 'rgba(0, 245, 160, 0.86)';
  handDebugContext.lineWidth = Math.max(1.4, videoRect.width * 0.006);

  for (const [start, end] of HAND_CONNECTIONS) {
    const from = landmarkToDebugPoint(landmarks[start], videoRect);
    const to = landmarkToDebugPoint(landmarks[end], videoRect);
    handDebugContext.beginPath();
    handDebugContext.moveTo(from.x, from.y);
    handDebugContext.lineTo(to.x, to.y);
    handDebugContext.stroke();
  }
}

function drawHandPoints(landmarks, videoRect) {
  for (let index = 0; index < landmarks.length; index += 1) {
    const point = landmarkToDebugPoint(landmarks[index], videoRect);
    const isTip = HAND_TIP_SET.has(index);
    handDebugContext.beginPath();
    handDebugContext.arc(
      point.x,
      point.y,
      isTip ? videoRect.width * 0.017 : videoRect.width * 0.011,
      0,
      Math.PI * 2,
    );
    handDebugContext.fillStyle = isTip ? 'rgba(130, 226, 255, 0.96)' : 'rgba(250, 251, 252, 0.82)';
    handDebugContext.fill();
  }

  const pinch = getPinch(landmarks, null, 1 / 30);
  if (pinch.strength > 0.08) {
    const thumb = landmarkToDebugPoint(landmarks[4], videoRect);
    const index = landmarkToDebugPoint(landmarks[8], videoRect);
    handDebugContext.strokeStyle = `rgba(255, 222, 110, ${0.35 + pinch.strength * 0.65})`;
    handDebugContext.lineWidth = Math.max(2, videoRect.width * 0.01);
    handDebugContext.beginPath();
    handDebugContext.moveTo(thumb.x, thumb.y);
    handDebugContext.lineTo(index.x, index.y);
    handDebugContext.stroke();
  }
}

function drawFaceDebug(landmarks, videoRect) {
  const top = landmarkToDebugPoint(landmarks[FACE_LANDMARKS.top], videoRect);
  const chin = landmarkToDebugPoint(landmarks[FACE_LANDMARKS.chin], videoRect);
  const leftCheek = landmarkToDebugPoint(landmarks[FACE_LANDMARKS.leftCheek], videoRect);
  const rightCheek = landmarkToDebugPoint(landmarks[FACE_LANDMARKS.rightCheek], videoRect);
  const center = landmarkToDebugPoint(landmarks[FACE_LANDMARKS.nose], videoRect);
  const width = Math.max(8, Math.abs(rightCheek.x - leftCheek.x));
  const height = Math.max(8, Math.abs(chin.y - top.y));

  handDebugContext.strokeStyle = 'rgba(130, 226, 255, 0.72)';
  handDebugContext.lineWidth = Math.max(1.5, videoRect.width * 0.006);
  handDebugContext.beginPath();
  handDebugContext.ellipse(center.x, center.y + height * 0.06, width * 0.58, height * 0.52, 0, 0, Math.PI * 2);
  handDebugContext.stroke();

  handDebugContext.strokeStyle = 'rgba(0, 245, 160, 0.74)';
  handDebugContext.beginPath();
  handDebugContext.moveTo(leftCheek.x, leftCheek.y);
  handDebugContext.lineTo(rightCheek.x, rightCheek.y);
  handDebugContext.moveTo(top.x, top.y);
  handDebugContext.lineTo(chin.x, chin.y);
  handDebugContext.stroke();

  for (const index of FACE_DEBUG_POINTS) {
    const landmark = landmarks[index];
    if (!landmark) {
      continue;
    }
    const point = landmarkToDebugPoint(landmark, videoRect);
    handDebugContext.beginPath();
    handDebugContext.arc(point.x, point.y, videoRect.width * 0.0065, 0, Math.PI * 2);
    handDebugContext.fillStyle = 'rgba(250, 251, 252, 0.78)';
    handDebugContext.fill();
  }
}

function landmarkToDebugPoint(landmark, videoRect) {
  return {
    x: videoRect.x + (1 - landmark.x) * videoRect.width,
    y: videoRect.y + landmark.y * videoRect.height,
  };
}

function distance2d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function distance3d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function getJointStraightness(base, joint, tip) {
  const baseVector = { x: base.x - joint.x, y: base.y - joint.y };
  const tipVector = { x: tip.x - joint.x, y: tip.y - joint.y };
  const baseLength = Math.hypot(baseVector.x, baseVector.y);
  const tipLength = Math.hypot(tipVector.x, tipVector.y);
  if (baseLength < 0.0001 || tipLength < 0.0001) {
    return 0;
  }

  const dot = (baseVector.x * tipVector.x + baseVector.y * tipVector.y) / (baseLength * tipLength);
  return smoothstep(0.32, 0.92, (-dot + 1) * 0.5);
}

function smoothstep(edge0, edge1, value) {
  const t = THREE.MathUtils.clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function dampStableScalar(current, target, alpha, deadzone) {
  const delta = target - current;
  const distance = Math.abs(delta);
  if (distance < deadzone) {
    return current + delta * alpha * 0.08;
  }
  const weightedAlpha = alpha * THREE.MathUtils.clamp((distance - deadzone) / Math.max(deadzone * 3, 0.001), 0.18, 1);
  return THREE.MathUtils.lerp(current, target, weightedAlpha);
}

function dampStableScalarLimited(current, target, alpha, deadzone, maxStep) {
  const next = dampStableScalar(current, target, alpha, deadzone);
  return current + THREE.MathUtils.clamp(next - current, -maxStep, maxStep);
}

function dampStableAngle(current, target, alpha, deadzone) {
  const delta = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  const distance = Math.abs(delta);
  if (distance < deadzone) {
    return current + delta * alpha * 0.08;
  }
  const weightedAlpha = alpha * THREE.MathUtils.clamp((distance - deadzone) / Math.max(deadzone * 3, 0.001), 0.18, 1);
  return current + delta * weightedAlpha;
}

function dampStableAngleLimited(current, target, alpha, deadzone, maxStep) {
  const delta = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  const next = dampStableAngle(current, target, alpha, deadzone);
  const limitedDelta = THREE.MathUtils.clamp(
    Math.atan2(Math.sin(next - current), Math.cos(next - current)),
    -Math.min(maxStep, Math.abs(delta)),
    Math.min(maxStep, Math.abs(delta)),
  );
  return current + limitedDelta;
}

function dampStableVector4(current, target, alpha, deadzone) {
  const deltas = [target.x - current.x, target.y - current.y, target.z - current.z, target.w - current.w];
  const distance = Math.hypot(...deltas);
  const stableAlpha =
    distance < deadzone
      ? alpha * 0.08
      : alpha * THREE.MathUtils.clamp((distance - deadzone) / Math.max(deadzone * 3, 0.001), 0.18, 1);
  current.x += deltas[0] * stableAlpha;
  current.y += deltas[1] * stableAlpha;
  current.z += deltas[2] * stableAlpha;
  current.w += deltas[3] * stableAlpha;
  return current;
}

function dampStableVector4Limited(current, target, alpha, deadzone, maxStep) {
  tmpVector4.copy(current);
  dampStableVector4(tmpVector4, target, alpha, deadzone);
  const deltaX = tmpVector4.x - current.x;
  const deltaY = tmpVector4.y - current.y;
  const deltaZ = tmpVector4.z - current.z;
  const deltaW = tmpVector4.w - current.w;
  const distance = Math.hypot(deltaX, deltaY, deltaZ, deltaW);
  const scale = distance > maxStep ? maxStep / Math.max(distance, 0.0001) : 1;
  current.x += deltaX * scale;
  current.y += deltaY * scale;
  current.z += deltaZ * scale;
  current.w += deltaW * scale;
  return current;
}

function lerpAngle(from, to, amount) {
  return from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * amount;
}

function randomSigned(scale) {
  return (Math.random() - 0.5) * 2 * scale;
}

const vertexShader = /* glsl */ `
attribute vec3 aScatter;
attribute float aSeed;
attribute float aSize;
attribute float aCore;

uniform float uTime;
uniform float uPixelRatio;
uniform float uIntro;
uniform float uMode;
uniform vec4 uPalm;
uniform vec2 uPalmVelocity;
uniform vec4 uFace;
uniform vec2 uFaceVelocity;
uniform float uFaceRotation;
uniform vec4 uFaceExpression;
uniform vec4 uEyeCalib;
uniform vec4 uEyeFineTune;
uniform vec4 uFaceEyeAnchors;
uniform vec4 uMotionSettings;
uniform vec4 uHandTuning;
uniform vec4 uVisualSettings;
uniform int uForceCount;
uniform vec4 uForce[10];
uniform vec2 uForceVelocity[10];
uniform vec4 uPinch[2];
uniform float uGustTime;
uniform vec2 uGustOrigin;
uniform vec2 uGustVelocity;
uniform float uShockTime;
uniform vec2 uShockCenter;

varying vec3 vColor;
varying float vAlpha;

float hash(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec2 safeNormalize(vec2 value) {
  return value / max(length(value), 0.0001);
}

void main() {
  vec3 target = position;
  float motionForce = uMotionSettings.x;
  float motionCurl = uMotionSettings.y;
  float motionDepth = uMotionSettings.z;
  float idleMotion = uMotionSettings.w;
  float handContactScale = max(uHandTuning.x, 0.05);
  float handPalmScale = max(uHandTuning.y, 0.05);
  float handPinchScale = max(uHandTuning.z, 0.05);
  float handWakeScale = uHandTuning.w;
  float bloomLift = uVisualSettings.x;
  float faceFollow = uVisualSettings.y;
  float faceMotion = uVisualSettings.z;
  float idle = sin(uTime * 0.8 + aSeed * 21.0) * 0.028 * idleMotion;
  vec3 radial = normalize(vec3(target.xy, 0.38));
  target += radial * idle;
  target.z += sin(uTime * 0.54 + aSeed * 17.0) * 0.035 * idleMotion;

  float intro = smoothstep(0.0, 1.0, uIntro);
  vec3 transformed = mix(aScatter, target, intro);

  float fingerGlow = 0.0;
  float palmGlow = 0.0;
  float pinchGlow = 0.0;
  float gustGlow = 0.0;
  float faceGlow = 0.0;
  float eyeGlow = 0.0;
  float handGlow = 0.0;
  float faceVisibility = 1.0;

  if (uMode > 0.5) {
    float faceStrength = smoothstep(0.0, 0.72, uFace.w);
    faceVisibility = smoothstep(0.05, 0.44, uFace.w);
    float mouth = uFaceExpression.x;
    float faceShake = uFaceExpression.y;
    float blink = uFaceExpression.z;
    float yaw = -clamp(uFaceExpression.w, -1.0, 1.0);
    float sway = clamp(length(uFaceVelocity) * 0.2, 0.0, 1.6);
    float c = cos(uFaceRotation);
    float s = sin(uFaceRotation);
    mat2 faceRotation = mat2(c, s, -s, c);
    vec2 localPoint = target.xy;
    vec2 eyeOrigin = uEyeCalib.xy;
    float eyeScale = max(uEyeCalib.z, 0.05);
    float eyeSpread = max(uEyeCalib.w, 0.05);
    float eyeHeight = max(uEyeFineTune.x, 0.05);
    float blinkResponse = uEyeFineTune.y;
    float eyeSharpness = max(uEyeFineTune.z, 0.05);
    float eyeIntensity = uEyeFineTune.w;
    vec2 detectedEyeMid = (uFaceEyeAnchors.xy + uFaceEyeAnchors.zw) * 0.5;
    vec2 leftEyeAnchor = detectedEyeMid + (uFaceEyeAnchors.xy - detectedEyeMid) * vec2(eyeSpread, eyeHeight) + eyeOrigin;
    vec2 rightEyeAnchor = detectedEyeMid + (uFaceEyeAnchors.zw - detectedEyeMid) * vec2(eyeSpread, eyeHeight) + eyeOrigin;
    vec2 noseAnchor = vec2(0.08, -0.34);
    float yawAmount = abs(yaw);
    vec2 headLag = clamp(uFaceVelocity, vec2(-18.0), vec2(18.0));
    float motionKick = clamp(length(headLag) * 0.2, 0.0, 3.2) * faceMotion;
    vec2 motionDir = safeNormalize(headLag + vec2(0.0001, 0.0));
    vec2 motionNormal = vec2(-motionDir.y, motionDir.x);
    localPoint.x += (localPoint.y + 0.18) * yaw * 0.24;
    localPoint.x *= 1.0 - yawAmount * 0.16;
    localPoint.y += yawAmount * sin(aSeed * 13.0 + localPoint.x * 2.0) * 0.035;
    localPoint += motionNormal * sin(aSeed * 41.0 + uTime * 18.0 + localPoint.y * 3.5) * motionKick * 0.2;
    localPoint += motionDir * sin(aSeed * 17.0 + uTime * 22.0 + localPoint.x * 2.8) * motionKick * 0.13;
    vec2 leftEyeVector = localPoint - leftEyeAnchor;
    vec2 rightEyeVector = localPoint - rightEyeAnchor;
    leftEyeVector.y /= eyeHeight;
    rightEyeVector.y /= eyeHeight;
    float eyeFalloff = 1.55 * eyeSharpness / (eyeScale * eyeScale);
    float leftEyeZone = exp(-dot(leftEyeVector, leftEyeVector) * eyeFalloff);
    float rightEyeZone = exp(-dot(rightEyeVector, rightEyeVector) * eyeFalloff);
    float eyeZone = clamp(leftEyeZone + rightEyeZone, 0.0, 1.0);
    float noseZone = exp(-dot(localPoint - noseAnchor, localPoint - noseAnchor) * 1.15);
    float mouthPulse = smoothstep(0.08, 0.78, mouth);
    float eyeIgnition = smoothstep(${EYE_IGNITION_START.toFixed(2)}, ${EYE_IGNITION_END.toFixed(2)}, blink * blinkResponse);
    float blinkFold = eyeIgnition * eyeZone;
    localPoint.y = mix(localPoint.y, 0.58 + (localPoint.y - 0.58) * 0.48, blinkFold);
    localPoint.x += sin(aSeed * 31.0 + uTime * 18.0) * blinkFold * 0.055;
    vec2 breathDir = safeNormalize(localPoint + vec2(0.0, 0.18));
    float breathRipple = sin(uTime * 13.0 + aSeed * 24.0 + length(localPoint) * 4.0) * 0.5 + 0.5;
    localPoint += breathDir * mouthPulse * (0.12 + breathRipple * 0.16);
    localPoint += safeNormalize(localPoint - noseAnchor) * noseZone * sway * 0.08;
    vec2 logoPoint = faceRotation * (localPoint * uFace.z);
    vec2 outward = safeNormalize(logoPoint);
    float sparkle = sin(uTime * 7.0 + aSeed * 32.0 + length(target.xy) * 3.0);
    logoPoint += outward * faceStrength * (0.025 + mouth * 0.18) * sparkle;
    float leash = 0.052 + length(target.xy) * 0.022 + (1.0 - aCore) * 0.024;
    float dragAmount = pow(1.0 - faceFollow, 0.75);
    logoPoint -= headLag * leash * (1.0 + sway * 2.0) * dragAmount;
    logoPoint += vec2(-headLag.y, headLag.x) * sin(aSeed * 23.0 + uTime * 4.5) * leash * (0.38 + motionKick * 0.82);
    vec3 faceMounted = vec3(
      uFace.xy + logoPoint,
      2.05 + target.z * uFace.z * 0.55 + yaw * localPoint.x * 0.18 + mouthPulse * (0.16 + breathRipple * 0.18) + eyeZone * eyeIgnition * 0.2 + faceShake * faceMotion * 1.65 + motionKick * sin(aSeed * 29.0 + uTime * 24.0) * 0.45
    );
    transformed = mix(transformed, faceMounted, faceStrength);
    eyeGlow = faceStrength * eyeZone * eyeIntensity * eyeIgnition * (0.22 + eyeIgnition * 1.45);
    faceGlow = faceStrength * (
      0.55 + aCore * 0.65 + mouthPulse * (0.55 + breathRipple * 0.45) + yawAmount * 0.28 + sway * 0.5 + motionKick * 0.9
    );
  }

  if (uMode < 0.5) {
    handGlow = 0.22;
    float palmStrength = uPalm.z;
    if (palmStrength > 0.001) {
      vec2 fromPalm = target.xy - uPalm.xy;
      float d = length(fromPalm);
      vec2 dir = safeNormalize(fromPalm);
      float depth = 0.75 + uPalm.w * 0.42;
      float palmRadius = (1.18 + uPalm.w * 0.78) * handPalmScale;
      float pressure = exp(-pow(d / palmRadius, 2.0));
      float rim = exp(-pow((d - palmRadius * 0.72) / max(0.18, palmRadius * 0.34), 2.0));
      float broad = exp(-pow(d / max(palmRadius * 2.65, 0.001), 1.65));
      float pulse = 0.82 + 0.18 * sin(uTime * 4.6 - d * 2.2 + aSeed * 4.0);
      float field = palmStrength * depth * (pressure * 0.84 + rim * 0.36) * pulse * motionForce * 1.62;
      float bloomPush = palmStrength * depth * broad * motionForce * 1.18;
      vec2 wake = clamp(uPalmVelocity * 0.038 * handWakeScale, vec2(-0.42), vec2(0.42));
      float handMass = palmStrength * depth * exp(-pow(d / max(palmRadius * 1.72, 0.001), 2.0)) * motionForce;
      vec2 shear = clamp(uPalmVelocity * 0.026 * handWakeScale, vec2(-0.34), vec2(0.34));
      vec2 orbital = vec2(-dir.y, dir.x) * sin(uTime * 5.2 + aSeed * 17.0 + d * 2.7) * handMass * 0.045 * motionCurl;
      transformed.xy += dir * field * (0.66 + pressure * 0.54) + wake * field * (0.38 + rim * 0.58);
      transformed.xy += shear * (handMass * (0.22 + pressure * 0.36) + bloomPush * 0.24) + orbital;
      transformed.xy += dir * bloomPush * (0.3 + aCore * 0.16);
      transformed.z += (field * (0.64 + pressure * 0.56 + aCore * 0.4) + handMass * 0.2 + bloomPush * (0.72 + aCore * 0.46)) * motionDepth;
      palmGlow += field * (1.08 + rim * 0.44) + handMass * 0.25 + bloomPush * 0.66;
    }

    for (int i = 0; i < 10; i++) {
      if (i < uForceCount) {
        vec4 force = uForce[i];
        vec2 fromFinger = target.xy - force.xy;
        float d = length(fromFinger);
        float radius = max(force.w * handContactScale, 0.08);
        vec2 dir = safeNormalize(fromFinger);
        float contact = exp(-pow(d / radius, 2.35));
        float shell = exp(-pow(d / max(radius * 2.05, 0.001), 1.8));
        float skin = max(shell - contact * 0.22, 0.0);
        float speed = clamp(length(uForceVelocity[i]) * 0.16, 0.0, 1.6);
        float field = force.z * (contact * 0.9 + skin * 0.16) * motionForce * 1.32;
        vec2 wake = clamp(uForceVelocity[i] * 0.035 * handWakeScale, vec2(-0.34), vec2(0.34));
        vec2 curl = vec2(-dir.y, dir.x) * sin(uTime * 3.5 + aSeed * 19.0 + d * 6.2);
        transformed.xy +=
          dir * field * (0.18 + contact * 0.42) +
          wake * field * (0.2 + speed * 0.38) +
          curl * field * (0.075 + speed * 0.12) * motionCurl;
        transformed.z += field * (0.28 + contact * 0.52 + speed * 0.22) * motionDepth;
        fingerGlow += force.z * (contact * 0.88 + skin * 0.2 + speed * contact * 0.28) * motionForce;
      }
    }

    for (int i = 0; i < 2; i++) {
      vec4 pinch = uPinch[i];
      if (pinch.z > 0.001) {
        vec2 toPinch = pinch.xy - target.xy;
        float d = length(toPinch) + 0.08;
        vec2 dir = toPinch / d;
        vec2 vortex = vec2(-dir.y, dir.x);
        float core = exp(-pow(d / max(0.46 * handPinchScale, 0.001), 2.0));
        float pull = exp(-pow(d / max(1.45 * handPinchScale, 0.001), 1.55));
        float speed = clamp(pinch.w, 0.0, 1.0);
        float field = pinch.z * (core * 0.96 + pull * 0.28) * motionForce * 1.18;
        float spin = 0.82 + 0.24 * sin(uTime * 9.0 + d * 4.0 + aSeed * 11.0);
        transformed.xy += dir * field * (0.58 + core * 0.42) + vortex * field * spin * (0.48 + speed * 0.5) * motionCurl;
        transformed.z += field * (0.54 + core * 0.42 + speed * 0.26) * motionDepth;
        pinchGlow += pinch.z * (core * 0.9 + pull * 0.2) * motionForce;
      }
    }

    float gustAge = uTime - uGustTime;
    if (gustAge > 0.0 && gustAge < 1.2) {
      vec2 dir = safeNormalize(uGustVelocity);
      vec2 normal = vec2(-dir.y, dir.x);
      vec2 rel = target.xy - uGustOrigin;
      float along = dot(rel, dir);
      float across = abs(dot(rel, normal));
      float front = 1.0 - smoothstep(0.0, 0.52, abs(along - gustAge * 5.4));
      float width = exp(-across * 0.42);
      float decay = pow(1.0 - gustAge / 1.2, 1.5);
      float gust = front * width * decay;
      transformed.xy +=
        dir * gust * 1.15 * motionForce +
        normal * sin(aSeed * 9.0 + uTime * 6.0) * gust * 0.16 * motionCurl;
      transformed.z += gust * 0.48 * motionDepth;
      gustGlow += gust;
    }

    float shockAge = uTime - uShockTime;
    if (shockAge > 0.0 && shockAge < 1.85) {
      vec2 fromShock = target.xy - uShockCenter;
      float shockDistance = length(fromShock);
      float wave = 1.0 - smoothstep(0.0, 0.32, abs(shockDistance - shockAge * 4.7));
      float decay = pow(1.0 - shockAge / 1.85, 1.55);
      vec3 blast = normalize(vec3(fromShock, 0.34 + hash(aSeed) * 0.7));
      transformed += blast * decay * (1.0 + wave * 3.8) * vec3(motionForce, motionForce, motionDepth);
      fingerGlow += decay * (0.34 + wave * 0.5);
    }
  }

  vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
  float perspectiveScale = 24.0 / max(2.0, -mvPosition.z);
  float energy = clamp(fingerGlow + palmGlow * 0.9 + pinchGlow + gustGlow * 0.8 + faceGlow + eyeGlow + handGlow, 0.0, 3.4);
  gl_PointSize = aSize * uPixelRatio * perspectiveScale * (0.84 + energy * 0.42);
  gl_Position = projectionMatrix * mvPosition;

  vec3 base = mix(vec3(0.42, 0.55, 0.68), vec3(0.9, 0.97, 1.0), 0.24 + aCore * 0.68);
  vec3 shimmer = vec3(0.04, 0.08, 0.1) * sin(aSeed * 20.0 + uTime * 0.7);
  vec3 forceColor = fingerGlow * vec3(0.24, 0.56, 1.0);
  vec3 palmColor = palmGlow * vec3(0.0, 0.72, 0.38);
  vec3 pinchColor = pinchGlow * vec3(0.78, 0.5, 0.08);
  vec3 gustColor = gustGlow * vec3(0.36, 0.66, 0.82);
  vec3 faceColor = faceGlow * mix(vec3(0.08, 0.5, 0.42), vec3(0.72, 0.93, 1.0), aCore);
  vec3 eyeColor = eyeGlow * vec3(1.0, 0.3, 0.03);
  float emission = 0.95 + bloomLift * 0.78;
  vColor = (base * 0.58 + shimmer + forceColor + palmColor + pinchColor + gustColor + faceColor + eyeColor) * emission;
  vAlpha = (0.04 + aCore * 0.065 + energy * 0.12 + (faceGlow + eyeGlow) * 0.018) * faceVisibility * (0.9 + bloomLift * 0.34);
}
`;

const fragmentShader = /* glsl */ `
precision highp float;

varying vec3 vColor;
varying float vAlpha;

void main() {
  vec2 uv = gl_PointCoord - 0.5;
  float d = length(uv);
  float core = smoothstep(0.5, 0.0, d);
  float spark = smoothstep(0.17, 0.0, d);
  float alpha = (core * 0.48 + spark * 0.24) * vAlpha;

  if (alpha < 0.012) {
    discard;
  }

  gl_FragColor = vec4(vColor * (0.48 + spark * 0.78), alpha);
}
`;
