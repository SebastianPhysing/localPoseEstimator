// script for visualization, uses three.js for movable three 3D scene
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const LEFT = 0x00ff00;
const RIGHT = 0xff0000;
const MIDDLE = 0xffffff;
const GRAY = 0x808080;

const info = document.getElementById('info');
const playButton = document.getElementById('play');
const slider = document.getElementById('slider');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 1000);
const controls = new OrbitControls(camera, renderer.domElement);

const data = await (await fetch('data.json')).json();
const joints = [];
const bones = [];
let index = 0;
let playing = false;
let startTime = 0;

function sideColor(name) {
  if (name.startsWith('l_')) return LEFT;
  if (name.startsWith('r_')) return RIGHT;
  return MIDDLE;
}

function makeLabel(text) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = '40px monospace';
  ctx.fillStyle = '#808080';
  ctx.textAlign = 'center';
  ctx.fillText(text, 64, 45);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas) }));
  sprite.scale.set(data.size * 0.08, data.size * 0.04, 1);
  return sprite;
}

// camera pyramids: apex = camera center, base = image corners
for (const cam of data.cameras) {
  const [apex, ...corners] = cam.pyramid.map((p) => new THREE.Vector3(...p));
  const points = [];
  for (let i = 0; i < 4; i++) points.push(apex, corners[i], corners[i], corners[(i + 1) % 4]);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  scene.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: GRAY })));

  const label = makeLabel(cam.name);
  label.position.set(apex.x, apex.y + data.size * 0.05, apex.z);
  scene.add(label);
}

// stick figure: spheres for joints, cylinders for bones
const jointGeometry = new THREE.SphereGeometry(data.size * 0.01);
for (const name of data.keypoints) {
  const mesh = new THREE.Mesh(jointGeometry, new THREE.MeshBasicMaterial({ color: sideColor(name) }));
  scene.add(mesh);
  joints.push(mesh);
}
const boneGeometry = new THREE.CylinderGeometry(data.size * 0.005, data.size * 0.005, 1);
for (const [a, b] of data.bones) {
  const ca = sideColor(data.keypoints[a]);
  const cb = sideColor(data.keypoints[b]);
  const mesh = new THREE.Mesh(boneGeometry, new THREE.MeshBasicMaterial({ color: ca === cb ? ca : MIDDLE }));
  scene.add(mesh);
  bones.push(mesh);
}

function showFrame(i) {
  index = i;
  const frame = data.frames[i];
  joints.forEach((mesh, j) => {
    mesh.visible = frame[j] !== null;
    if (mesh.visible) mesh.position.set(...frame[j]);
  });
  data.bones.forEach(([a, b], k) => {
    const mesh = bones[k];
    mesh.visible = frame[a] !== null && frame[b] !== null;
    if (!mesh.visible) return;
    // stretch the unit cylinder from joint a to joint b
    const pa = new THREE.Vector3(...frame[a]);
    const pb = new THREE.Vector3(...frame[b]);
    mesh.position.copy(pa).add(pb).multiplyScalar(0.5);
    mesh.scale.set(1, pa.distanceTo(pb), 1);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pb.sub(pa).normalize());
  });
  slider.value = i;
  info.textContent = `frame ${i + 1} / ${data.frames.length}   ${data.times[i].toFixed(2)} s`;
}

function setPlaying(value) {
  playing = value;
  startTime = performance.now() - data.times[index] * 1000;
  playButton.textContent = playing ? 'Pause' : 'Play';
}

function animate(now) {
  requestAnimationFrame(animate);
  if (playing) {
    let t = (now - startTime) / 1000;
    if (t > data.times[data.times.length - 1]) {
      startTime = now;
      t = 0;
      showFrame(0);
    }
    let i = index;
    while (i + 1 < data.frames.length && data.times[i + 1] <= t) i++;
    if (i !== index) showFrame(i);
  }
  controls.update();
  renderer.render(scene, camera);
}

playButton.onclick = () => setPlaying(!playing);
slider.oninput = () => {
  setPlaying(false);
  showFrame(Number(slider.value));
};
window.onkeydown = (e) => {
  if (e.key === ' ') {
    e.preventDefault();
    setPlaying(!playing);
  }
};
window.onresize = () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
};

const focus = new THREE.Vector3(...data.focus);
camera.position.set(focus.x + 0.8 * data.size, focus.y + 0.9 * data.size, focus.z + 1.7 * data.size);
controls.target.copy(focus);
slider.max = data.frames.length - 1;
showFrame(0);
requestAnimationFrame(animate);
