import * as THREE from 'three';

export const PARTICLE_COUNT = 100_000;
export const MIN_PARTICLE_COUNT = 18_000;
export const DEFAULT_PARTICLE_COUNT = 54_000;
export const MAX_FORCES = 10;
export const TARGET_RENDER_FPS = 60;
export const FPS_DISPLAY_CAP = 60;
export const MIN_RENDER_FRAME_MS = 1000 / TARGET_RENDER_FPS;
export const FRAME_SKIP_EPSILON_MS = 0.1;
export const MIN_ADAPTIVE_TRACKING_FPS = 30;
export const FORCE_LOOP_EPSILON = 0.02;
export const TRACKING_MODES = {
  HAND: 'hand',
  FACE: 'face',
};
export const CAMERA_STORAGE_KEY = 'particle-demo-camera-device-id';
export const CALIBRATION_STORAGE_KEY = 'particle-demo-face-calibration';
export const PARTICLE_SETTINGS_STORAGE_KEY = 'particle-demo-particle-settings';
export const DEFAULT_CAMERA_VALUE = '__particle_demo_browser_default_camera__';
export const PHONE_CAMERA_PATTERN = /iphone|continuity|ipad|phone/i;
export const HAND_TIP_INDICES = [4, 8, 12, 16, 20];
export const HAND_TIP_SET = new Set(HAND_TIP_INDICES);
export const HAND_TIP_SLOTS = new Map(HAND_TIP_INDICES.map((tip, index) => [tip, index]));
export const FINGER_CHAINS = [
  { tip: 4, dip: 3, pip: 2, mcp: 1 },
  { tip: 8, dip: 7, pip: 6, mcp: 5 },
  { tip: 12, dip: 11, pip: 10, mcp: 9 },
  { tip: 16, dip: 15, pip: 14, mcp: 13 },
  { tip: 20, dip: 19, pip: 18, mcp: 17 },
];
export const LONG_FINGER_TIPS = [8, 12, 16, 20];
export const HAND_CONNECTIONS = [
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
export const FACE_LANDMARKS = {
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
export const FACE_DEBUG_POINTS = [
  10, 21, 54, 67, 103, 109, 127, 132, 136, 148, 152, 162, 172, 176, 234, 251, 284, 297,
  323, 332, 356, 361, 365, 377, 389, 397, 454, 1, 4, 33, 61, 70, 105, 133, 145, 159, 263,
  291, 300, 334, 362, 374, 386,
];
export const LOGO_FACE_ANCHORS = {
  eyeMid: new THREE.Vector2(0, 0.58),
  noseTip: new THREE.Vector2(0.08, -0.34),
  mouth: new THREE.Vector2(-0.02, -1.3),
  eyeToMouth: 1.82,
  cheekWidth: 5.55,
};
export const DEFAULT_FACE_EYE_ANCHORS = new THREE.Vector4(-1.08, 0.58, 1.08, 0.58);
export const DEFAULT_FACE_CALIBRATION = {
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
export const CALIBRATION_CONTROLS = [
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
export const DEFAULT_PARTICLE_SETTINGS = {
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
export const PARTICLE_SETTING_CONTROLS = [
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
export const HAND_FILTER = { kind: 'hand' };
export const FACE_FILTER = { kind: 'face' };
// Camera states the per-frame SEARCH fallback must never clobber: setup progress,
// in-flight camera work, and terminal failure states that carry recovery guidance.
export const STICKY_CAMERA_STATES = new Set([
  'CAMERA',
  'LOADING',
  'SCANNING',
  'SWITCHING',
  'POINTER',
  'CAM BLOCKED',
  'NO CAMERA',
  'CAM BUSY',
  'CAM ERROR',
  'NO TRACKING',
  'GPU LOST',
]);
export const SETTING_SECTIONS = ['Shared', 'Hand', 'Face'];
export const FACE_POSITION_DEADZONE = 0.018;
export const FACE_SCALE_DEADZONE = 0.007;
export const FACE_ROLL_DEADZONE = 0.01;
export const FACE_EXPRESSION_DEADZONE = 0.025;
export const FACE_EYE_ANCHOR_DEADZONE = 0.035;
export const FACE_JUMP_CONTAIN_START = 0.42;
export const FACE_JUMP_CONTAIN_END = 1.12;
export const FACE_SCALE_MAX_STEP = 0.045;
export const FACE_ROLL_MAX_STEP = 0.075;
export const FACE_YAW_MAX_STEP = 0.12;
export const FACE_EYE_ANCHOR_MAX_STEP = 0.24;
export const FACE_RESULT_FRESH_MS = 420;
export const FACE_DROPOUT_HOLD_MS = 420;
export const FACE_CACHE_RESET_GRACE_MS = 1100;
export const HAND_RESULT_FRESH_MS = 560;
export const HAND_DROPOUT_HOLD_MS = 1050;
export const HAND_SLOT_RESET_GRACE_MS = 1700;
export const HAND_MIN_TRACKING_CONFIDENCE = 0.38;
export const HAND_MIN_FRAME_QUALITY = 0.28;
export const HAND_ASSIGNMENT_SWITCH_MARGIN = 0.24;
export const HAND_ASSIGNMENT_HANDEDNESS_LOCK_MS = 1600;
export const HAND_ASSIGNMENT_HANDEDNESS_PENALTY = 0.78;
export const EYE_IGNITION_START = 0.46;
export const EYE_IGNITION_END = 0.86;
export const HINT_MESSAGES = {
  [TRACKING_MODES.HAND]:
    'Show your hand to the camera — point to stir the particles, open your palm to push, pinch to pull. Clap for a shockwave.',
  [TRACKING_MODES.FACE]:
    'Look into the camera — the particles form a mask that follows your head. Blink, smile, and shake your head.',
};
export const HINT_FADE_MS = 450;
