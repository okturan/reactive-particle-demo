import * as THREE from 'three';

const tmpVector4 = new THREE.Vector4();

export function distance2d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function distance3d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

export function getJointStraightness(base, joint, tip) {
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

export function smoothstep(edge0, edge1, value) {
  const t = THREE.MathUtils.clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

export function dampStableScalar(current, target, alpha, deadzone) {
  const delta = target - current;
  const distance = Math.abs(delta);
  if (distance < deadzone) {
    return current + delta * alpha * 0.08;
  }
  const weightedAlpha = alpha * THREE.MathUtils.clamp((distance - deadzone) / Math.max(deadzone * 3, 0.001), 0.18, 1);
  return THREE.MathUtils.lerp(current, target, weightedAlpha);
}

export function dampStableScalarLimited(current, target, alpha, deadzone, maxStep) {
  const next = dampStableScalar(current, target, alpha, deadzone);
  return current + THREE.MathUtils.clamp(next - current, -maxStep, maxStep);
}

export function dampStableAngle(current, target, alpha, deadzone) {
  const delta = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  const distance = Math.abs(delta);
  if (distance < deadzone) {
    return current + delta * alpha * 0.08;
  }
  const weightedAlpha = alpha * THREE.MathUtils.clamp((distance - deadzone) / Math.max(deadzone * 3, 0.001), 0.18, 1);
  return current + delta * weightedAlpha;
}

export function dampStableAngleLimited(current, target, alpha, deadzone, maxStep) {
  const delta = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  const next = dampStableAngle(current, target, alpha, deadzone);
  const limitedDelta = THREE.MathUtils.clamp(
    Math.atan2(Math.sin(next - current), Math.cos(next - current)),
    -Math.min(maxStep, Math.abs(delta)),
    Math.min(maxStep, Math.abs(delta)),
  );
  return current + limitedDelta;
}

export function dampStableVector4(current, target, alpha, deadzone) {
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

export function dampStableVector4Limited(current, target, alpha, deadzone, maxStep) {
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

export function lerpAngle(from, to, amount) {
  return from + Math.atan2(Math.sin(to - from), Math.cos(to - from)) * amount;
}

export function randomSigned(scale) {
  return (Math.random() - 0.5) * 2 * scale;
}
