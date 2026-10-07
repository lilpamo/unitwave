// Where the 3D views' camera goes: looking at what they show from a fixed angle, at the
// distance where it just fills the canvas (its width and its height), with a small
// margin. Only the camera is placed here; every position comes from the server.
import * as THREE from 'three';

// The views' fixed viewing angle: from anterior-left and above, as before.
const VIEW = new THREE.Vector3(-0.55, -0.75, 0.95).normalize();
const MAX_POINTS = 20000;  // enough vertices to bound a mesh; more only costs time

// (n,) world positions of an object's vertices, at most MAX_POINTS of them, spread evenly.
export function vertices(object) {
  object.updateMatrixWorld(true);
  const attrs = [];
  object.traverse((o) => { if (o.geometry?.attributes?.position) attrs.push([o.geometry.attributes.position, o.matrixWorld]); });
  const total = attrs.reduce((n, [p]) => n + p.count, 0);
  const step = Math.max(1, Math.ceil(total / MAX_POINTS));
  const out = [];
  for (const [p, matrix] of attrs) {
    for (let i = 0; i < p.count; i += step) out.push(new THREE.Vector3().fromBufferAttribute(p, i).applyMatrix4(matrix));
  }
  return out;
}

// The 8 corners of a box, for fitting before any mesh has loaded.
export function corners(box) {
  const out = [];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) out.push(new THREE.Vector3(x, y, z));
  return out;
}

// Aim the camera at the points' middle (as seen from VIEW) and move it back until every
// point is inside the view, leaving `margin` of the canvas free at the tightest edge.
export function fit(camera, controls, points, margin = 0.04) {
  if (!points.length) return;
  const forward = VIEW.clone().negate();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  // The middle across the view (right and up), not the box's centre: equal margins.
  const box = new THREE.Box3().setFromPoints(points);
  let centre = box.getCenter(new THREE.Vector3());
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  const rel = new THREE.Vector3();
  for (const p of points) {
    rel.subVectors(p, centre);
    const x = rel.dot(right), y = rel.dot(up);
    lo = [Math.min(lo[0], x), Math.min(lo[1], y)];
    hi = [Math.max(hi[0], x), Math.max(hi[1], y)];
  }
  centre = centre.clone().addScaledVector(right, (lo[0] + hi[0]) / 2).addScaledVector(up, (lo[1] + hi[1]) / 2);
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * (1 - margin);
  const tanH = tanV * camera.aspect;
  let distance = 0;
  for (const p of points) {
    rel.subVectors(p, centre);
    const depth = rel.dot(forward);  // nearer the camera when negative
    distance = Math.max(distance, Math.abs(rel.dot(right)) / tanH - depth, Math.abs(rel.dot(up)) / tanV - depth);
  }
  controls.target.copy(centre);
  camera.position.copy(centre).addScaledVector(VIEW, distance);
  controls.update();
}
