import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { baseToScene, cadToScene, heldCenter, BATTERY_CENTER_Z, DEMO_PICKS, DEMO_TARGETS } from './scene-geometry.mjs';

// Default camera and top view look along -scene Z: smaller scene X is left.
const left = Object.values(DEMO_PICKS).map(p => baseToScene([...p, 0]));
const right = Object.values(DEMO_TARGETS).map(p => baseToScene([...p, 0]));
assert(Math.max(...left.map(p => p[0])) < Math.min(...right.map(p => p[0])));
assert(baseToScene([...DEMO_PICKS.red, 0])[0] < baseToScene([...DEMO_PICKS.yellow, 0])[0]);
assert(DEMO_PICKS.red[1] < DEMO_PICKS.blue[1]);
const origin = baseToScene([0,0,0]);
const basis = [[1,0,0],[0,1,0],[0,0,1]].map(p=>baseToScene(p).map((v,i)=>v-origin[i]));
const [a,b,c] = basis;
const det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
assert(Math.abs(det-1)<1e-12, 'Coordinate mapping must preserve handedness');
const vertices = readFileSync(new URL('../tmp/pdfs/battery.stl', import.meta.url), 'utf8')
  .split(/\r?\n/).filter(l=>l.trim().startsWith('vertex ')).map(l=>l.trim().split(/\s+/).slice(1).map(Number));
assert(vertices.length > 1000);
const transformed = vertices.map(cadToScene);
const ys = transformed.map(p=>p[1] + BATTERY_CENTER_Z);
assert(Math.abs(Math.min(...ys)) < 1e-12, 'Model bottom must sit on the table');
assert(Math.abs(Math.max(...ys)-.08) < 1e-12);
assert(cadToScene([0,0,-30])[1] > cadToScene([0,0,50])[1], 'Handle must be above closed base');
const pickZ = .069207;
assert(Math.abs(heldCenter([.2,-.3,pickZ],pickZ)[2]-BATTERY_CENTER_Z)<1e-12, 'Closing must not snap center height');
assert(Math.abs(heldCenter([.2,-.3,pickZ+.1],pickZ)[2]-(BATTERY_CENTER_Z+.1))<1e-12);
console.log('PASS: left/right, colour order, proper rotation, actual STL height/up, continuous grasp height');
