import { EYE_IGNITION_START, EYE_IGNITION_END } from './constants.js';

export const vertexShader = /* glsl */ `
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
    logoPoint += vec2(-headLag.y, headLag.x) * sin(aSeed * 23.0 + uTime * 4.5) * leash * (0.38 + motionKick * 0.32);
    vec3 faceMounted = vec3(
      uFace.xy + logoPoint,
      2.05 + target.z * uFace.z * 0.55 + yaw * localPoint.x * 0.18 + mouthPulse * (0.16 + breathRipple * 0.18) + eyeZone * eyeIgnition * 0.2 + faceShake * faceMotion * 0.42 + motionKick * sin(aSeed * 29.0 + uTime * 24.0) * 0.18
    );
    transformed = mix(transformed, faceMounted, faceStrength);
    eyeGlow = faceStrength * eyeZone * eyeIntensity * eyeIgnition * (0.22 + eyeIgnition * 1.45);
    faceGlow = faceStrength * (
      0.34 + aCore * 0.44 + mouthPulse * (0.1 + breathRipple * 0.12) + yawAmount * 0.12 + sway * 0.08 + motionKick * 0.07
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

export const fragmentShader = /* glsl */ `
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
