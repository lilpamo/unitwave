// The homepage's 3D overview: the whole brain and every matching probe as a line from
// its tip to its top, in CCF µm as the server sends them (the page computes none).
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';
import { OBJLoader } from '/static/vendor/three/OBJLoader.js';
import { fit, vertices } from '/static/fit3d.js';

export function createOverview(el, { onHover, onPick }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  el.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 10, 200000);
  camera.up.set(0, -1, 0);  // CCF dorsoventral grows ventrally, so dorsal is up
  const controls = new OrbitControls(camera, renderer.domElement);
  const render = () => renderer.render(scene, camera);
  controls.addEventListener('change', render);
  const light = new THREE.DirectionalLight(0xffffff, 1.6);
  light.position.set(-0.5, -1, 0.6);
  camera.add(light);
  scene.add(camera, new THREE.AmbientLight(0xffffff, 1.1));
  const brainMaterial = new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.1, depthWrite: false });
  let lines = null, rows = [], hovered = -1;
  // The brain fills the canvas; refitted when the canvas changes shape, until the user
  // rotates or zooms it themselves.
  let brainPoints = [], probePoints = [], moved = false;
  // The brain and the probes' ends (they stick out above it) both stay in view.
  const refit = () => { if (!moved) fit(camera, controls, [...brainPoints, ...probePoints]); };
  controls.addEventListener('start', () => { moved = true; });

  new ResizeObserver(() => {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    refit();
    render();
  }).observe(el);

  async function loadBrain(id, muted) {
    brainMaterial.color.set(muted);
    const brain = await new OBJLoader().loadAsync(`/mesh/${id}.obj`);
    brain.traverse((o) => { if (o.isMesh) o.material = brainMaterial; });
    scene.add(brain);
    brainPoints = vertices(brain);
    refit();
    render();
  }

  function pickAt(event) {
    if (!lines || !rows.length) return -1;
    const rect = renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.params.Line.threshold = 120;  // µm
    ray.setFromCamera(new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1), camera);
    const hit = ray.intersectObject(lines)[0];
    return hit ? Math.floor(hit.index / 2) : -1;  // two vertices per probe
  }
  renderer.domElement.addEventListener('mousemove', (e) => {
    const i = pickAt(e);
    if (i !== hovered) { hovered = i; onHover(i < 0 ? null : rows[i], e); }
    else if (i >= 0) onHover(rows[i], e);
  });
  renderer.domElement.addEventListener('mouseleave', () => { hovered = -1; onHover(null); });
  // A drag to rotate ends in a click; only a click that barely moved opens a session.
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('click', (e) => {
    const moved = down ? Math.hypot(e.clientX - down[0], e.clientY - down[1]) : Infinity;
    down = null;
    if (moved > 4) return;
    const i = pickAt(e);
    if (i >= 0) onPick(rows[i]);
  });

  function update(probes, colours) {
    if (lines) scene.remove(lines);
    rows = probes.lines;
    const positions = [], vertexColours = [];
    for (const r of rows) {
      const c = new THREE.Color(colours[r.lab]);
      positions.push(...r.tip, ...r.top);
      vertexColours.push(...c.toArray(), ...c.toArray());
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.Float32BufferAttribute(vertexColours, 3));
    lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ vertexColors: true }));
    scene.add(lines);
    probePoints = rows.flatMap((r) => [new THREE.Vector3(...r.tip), new THREE.Vector3(...r.top)]);
    if (brainPoints.length) refit();  // before the brain loads, it decides the view
    render();
  }

  return { loadBrain, update, setMuted: (c) => { brainMaterial.color.set(c); render(); } };
}
