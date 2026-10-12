// Display geometry only. These example XY values are NOT robot calibration.
export const BASE_HEIGHT = 0.16;
export const BATTERY_CENTER_Z = 0.04;
export const DEMO_PICKS = {
  red: [0.20, -0.36], yellow: [0.11, -0.36],
  blue: [0.20, -0.26], green: [0.11, -0.26],
};
export const DEMO_TARGETS = {
  T0: [-0.16, -0.31],
  P1: [-0.07, -0.38], P2: [-0.24, -0.38], P3: [-0.155, -0.20],
};

// Proper rotation (det=+1): base +X is screen-left, +Z is scene-up.
export function baseToScene([x, y, z]) { return [-x, z + BASE_HEIGHT, y]; }

// Supplied CAD: handle at Z=-30 mm, closed base at Z=+50 mm.
// Its bounding-box centre is (0,0,10) mm. Rotate +90 degrees about X.
export function cadToScene([x, y, z]) { return [x / 1000, (10 - z) / 1000, y / 1000]; }

// Preserve the contact transform through close, lift and release.
export function heldCenter(tcp, contactZ) {
  return [tcp[0], tcp[1], tcp[2] + BATTERY_CENTER_Z - contactZ];
}
