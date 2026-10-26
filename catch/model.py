"""MJCF of the UR5e (MuJoCo Menagerie, unmodified file) with the four-arm hand on the flange, a parcel and the scene.

The Menagerie file is read and composed as text: the hand is inserted after the flange site `attachment_site`,
the robot is lifted on a pedestal, the UR collision geoms get their own collision bits, and the keyframe is removed.
Hand frame: +z is the hand axis, out of the palm; the arms are built with the hinge axis -y of each arm
frame, positive angles close the arm). One motor (slide joint "motor") pulls one fixed tendon per arm
e_a = g * sum_j q_aj - s, limited to [0, slack] with the series stiffness.
Collision bits: parcel contype 1 / conaffinity 1; hand 2 / 1; UR 4 / 1; floor and pedestal 8 / 1. The parcel
touches everything, the hand and the UR do not touch each other or the floor.
"""
from pathlib import Path

import mujoco
import numpy as np

from .config import Params

ROOT = Path(__file__).resolve().parents[1]
UR_DIR = ROOT / "third_party" / "mujoco_menagerie" / "universal_robots_ur5e"
UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
             "wrist_3_joint")


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_axis(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate(([np.cos(ang / 2)], np.sin(ang / 2) * axis))


def _f(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def hand_xml(p: Params):
    pitch, width = np.asarray(p.pitch_m), np.asarray(p.width_m)
    stops, k = p.stop_rad, p.joint_stiffness()
    mass = p.arm_mass_kg * pitch * width / np.sum(pitch * width)
    h, out, z0 = p.phalanx_h_m, np.radians(p.outward_limit_deg), p.housing_m
    b = ['<body name="hand" pos="0 0.1 0" quat="-1 1 0 0" childclass="hand">']
    if p.wrist_stroke_m > 0:                    # passive compliant wrist: the whole hand slides back on a spring
        b.append(f'  <joint name="wrist" type="slide" axis="0 0 1" range="{-p.wrist_stroke_m:.6g} 0" '
                 f'stiffness="{p.wrist_k:.6g}" damping="{p.wrist_c:.6g}" springref="{p.wrist_preload_m:.6g}" '
                 'armature="0.001" solreflimit="0.01 1"/>')
    b.append(f'  <geom name="housing" type="cylinder" size="0.045 {z0 / 2}" pos="0 0 {z0 / 2}" '
             f'mass="{p.housing_mass_kg}" contype="0" conaffinity="0" rgba="0.25 0.25 0.28 1"/>')
    palm = (f'<geom name="palm" type="box" size="{_f([p.palm_half_m, p.palm_half_m, p.palm_thick_m / 2])}" '
            f'pos="0 0 {z0 + p.palm_thick_m / 2}" mass="{p.pad_mass_kg if p.pad_stroke_m > 0 else 0.05}" '
            'rgba="0.4 0.6 0.5 1"/>')
    if p.pad_stroke_m > 0:                      # light palm plate on a spring
        b += ['  <body name="pad">',
              f'    <joint name="pad" type="slide" axis="0 0 1" range="{-p.pad_stroke_m:.6g} 0" '
              f'stiffness="{p.pad_k:.6g}" damping="{p.pad_c:.6g}" springref="{p.pad_preload_m:.6g}" '
              'armature="0.0002" solreflimit="0.01 1"/>',
              '    ' + palm, '  </body>']
    else:
        b.append('  ' + palm)
    b += [
         f'  <site name="palm_c" pos="0 0 {z0 + p.palm_thick_m}" size="0.004"/>',
         f'  <body name="spool" pos="0 0 {z0 / 2}">',
         f'    <joint name="motor" type="slide" axis="0 0 1" range="{-p.stroke_min_m} {p.stroke_max_m}" '
         f'damping="{p.motor_damping():.6g}" armature="{p.motor_mass_kg}" stiffness="0"/>',
         '    <geom type="sphere" size="0.008" mass="0.01" contype="0" conaffinity="0"/>',
         '  </body>']
    for a in range(p.n_arms):
        az = 2 * np.pi * a / p.n_arms
        quat = _quat_mul(_quat_axis([0, 0, 1], az), _quat_axis([0, 1, 0], np.radians(p.mount_deg)))
        pos = [p.root_radius_m * np.cos(az), p.root_radius_m * np.sin(az), z0]
        ind = "  "
        b.append(f'{ind}<body name="a{a}_0" pos="{_f(pos)}" quat="{_f(quat)}">')
        for j in range(len(pitch)):
            i2 = ind + "  " * (j + 1)
            if j > 0:
                b.append(f'{ind + "  " * j}<body name="a{a}_{j}" pos="0 0 {pitch[j - 1]:.6g}">')
            b.append(f'{i2}<joint name="a{a}_j{j}" type="hinge" axis="0 -1 0" range="{-out:.6g} {stops[j]:.6g}" '
                     f'stiffness="{k[j]:.6g}"/>')
            b.append(f'{i2}<geom name="a{a}_g{j}" type="box" size="{_f([h / 2, width[j] / 2, pitch[j] * 0.49])}" '
                     f'pos="{_f([h / 2, 0, pitch[j] / 2])}" mass="{mass[j]:.6g}" rgba="0.85 0.85 0.85 1"/>')
        b.append(f'{ind + "  " * len(pitch)}<site name="a{a}_tip" pos="0 0 {pitch[-1]:.6g}" size="0.003"/>')
        for j in range(len(pitch) - 1, -1, -1):
            b.append(f'{ind + "  " * j}</body>')
    b.append('</body>')
    tendons = ['<tendon>']
    for a in range(p.n_arms):
        tendons.append(f'  <fixed name="cable{a}" range="{-p.cable_stretch_m:.6g} {p.cable_slack_m:.6g}" '
                       f'solreflimit="{-p.series_stiffness:.6g} {-p.series_damping:.6g}">')
        for j in range(len(pitch)):
            tendons.append(f'    <joint joint="a{a}_j{j}" coef="{p.guide_m:.6g}"/>')
        tendons.append('    <joint joint="motor" coef="-1"/>')
        tendons.append('  </fixed>')
    tendons.append('</tendon>')
    actuator = (f'<motor name="hand_motor" joint="motor" gear="{p.n_arms * p.t_max_n * p.motor_force_scale:.6g}" '
                'ctrlrange="-1 1"/>')
    if p.semi_active and p.pad_stroke_m > 0:
        actuator += f'\n    <damper name="pad_damper" joint="pad" kv="1" ctrlrange="0 {p.pad_c_max:.6g}"/>'
    if p.semi_active and p.wrist_stroke_m > 0:
        actuator += f'\n    <damper name="wrist_damper" joint="wrist" kv="1" ctrlrange="0 {p.wrist_c_max:.6g}"/>'
    return "\n".join(b), "\n".join(tendons), actuator


def world_xml(p: Params):
    hb = p.base_height_m
    s = 0.035
    return "\n".join([
        '<light name="key" pos="1.5 -1.5 3" dir="-0.4 0.4 -1" directional="true" diffuse="0.7 0.7 0.7"/>',
        '<geom name="floor" type="plane" size="0 0 0.05" material="groundplane" contype="8" conaffinity="1"/>',
        f'<geom name="pedestal" type="box" size="0.12 0.12 {hb / 2}" pos="0 0 {hb / 2}" rgba="0.3 0.3 0.32 1" '
        'contype="8" conaffinity="1"/>',
        '<body name="parcel" pos="2 0 1.3">',
        '  <freejoint name="parcel_free"/>',
        f'  <geom name="parcel" type="box" size="{s} {s} {s}" mass="0.2" rgba="0.72 0.55 0.35 1" '
        f'friction="{p.box_friction} 0.005 0.0001" contype="1" conaffinity="1"/>',
        '</body>',
        # operator: visual only
        '<body name="operator" pos="2.9 0 0">',
        '  <geom type="capsule" fromto="0 -0.1 0 0 -0.1 0.85" size="0.07" rgba="0.2 0.25 0.4 1" contype="0" conaffinity="0"/>',
        '  <geom type="capsule" fromto="0 0.1 0 0 0.1 0.85" size="0.07" rgba="0.2 0.25 0.4 1" contype="0" conaffinity="0"/>',
        '  <geom type="capsule" fromto="0 0 0.9 0 0 1.45" size="0.16" rgba="0.75 0.4 0.2 1" contype="0" conaffinity="0"/>',
        '  <geom type="sphere" pos="0 0 1.62" size="0.11" rgba="0.9 0.75 0.6 1" contype="0" conaffinity="0"/>',
        '</body>',
    ])


SCENE_ASSETS = """
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global azimuth="150" elevation="-15" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.65 0.75" rgb2="0.15 0.15 0.2" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.55 0.55 0.52" rgb2="0.5 0.5 0.47"
      markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
  </asset>
"""


def build_xml(p: Params) -> str:
    s = (UR_DIR / "ur5e.xml").read_text()

    def sub(old, new):
        nonlocal s
        assert s.count(old) == 1, old
        s = s.replace(old, new)

    sub('meshdir="assets"', f'meshdir="{UR_DIR / "assets"}"')
    sub('<option integrator="implicitfast"/>', f'<option integrator="implicitfast" timestep="{p.dt_s}"/>' + SCENE_ASSETS)
    sub('<geom type="capsule" group="3"/>', '<geom type="capsule" group="3" contype="4" conaffinity="1"/>')
    hand_default = (f'<default class="hand"><joint damping="{p.joint_damping}" armature="{p.joint_armature}" '
                    f'solreflimit="0.005 1"/><geom contype="2" conaffinity="1" condim="3" '
                    f'friction="{p.hand_friction} 0.005 0.0001" solref="0.005 1" solimp="0.9 0.95 0.001"/></default>')
    sub('    <default class="ur5e">', '    ' + hand_default + '\n    <default class="ur5e">')
    sub('<body name="base" quat="0 0 0 -1" childclass="ur5e">',
        f'<body name="base" pos="0 0 {p.base_height_m}" quat="0 0 0 -1" childclass="ur5e">')
    body, tendons, act = hand_xml(p)
    site = '<site name="attachment_site" pos="0 0.1 0" quat="-1 1 0 0"/>'
    sub(site, site + "\n" + body)
    sub('  </worldbody>', world_xml(p) + '\n  </worldbody>\n' + tendons)
    sub('  </actuator>', '    ' + act + '\n  </actuator>')
    i0, i1 = s.index('  <keyframe>'), s.index('</keyframe>') + len('</keyframe>')
    s = s[:i0] + s[i1:]
    return s


class CatchModel:
    def __init__(self, p: Params = None):
        self.p = p or Params()
        self.xml = build_xml(self.p)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.ur_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in UR_JOINTS])
        self.ur_d = np.array([m.jnt_dofadr[m.joint(n).id] for n in UR_JOINTS])
        self.ur_act = np.array([m.actuator(n).id for n in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1",
                                                           "wrist_2", "wrist_3")])
        self.motor_act = m.actuator("hand_motor").id
        self.damper_act = [m.actuator(n).id for n in ("pad_damper", "wrist_damper") if mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) >= 0]
        self.comp_q = {n: m.jnt_qposadr[m.joint(n).id] for n in ("pad", "wrist")
                       if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0}
        self.comp_d = {n: m.jnt_dofadr[m.joint(n).id] for n in self.comp_q}
        self.motor_q = m.jnt_qposadr[m.joint("motor").id]
        self.motor_d = m.jnt_dofadr[m.joint("motor").id]
        nj = len(self.p.pitch_m)
        self.arm_q = np.array([[m.jnt_qposadr[m.joint(f"a{a}_j{j}").id] for j in range(nj)]
                               for a in range(self.p.n_arms)])
        self.parcel_body = m.body("parcel").id
        self.parcel_geom = m.geom("parcel").id
        self.parcel_q = m.jnt_qposadr[m.joint("parcel_free").id]
        self.parcel_d = m.jnt_dofadr[m.joint("parcel_free").id]
        self.palm_site = m.site("palm_c").id
        self.hand_body = m.body("hand").id
        self.hand_geoms = set(i for i in range(m.ngeom) if m.geom_contype[i] == 2)
        self.ur_geoms = set(i for i in range(m.ngeom) if m.geom_contype[i] == 4)
        self.floor_geoms = set(i for i in range(m.ngeom) if m.geom_contype[i] == 8)

    def set_hand(self, t_max_n, close_speed_mps):
        """Change motor tension and speed at run time (gear, motor damping, spine stiffness with the same Pi)."""
        m, p = self.model, self.p
        p.t_max_n, p.close_speed_mps = float(t_max_n), float(close_speed_mps)
        m.actuator_gear[self.motor_act, 0] = p.n_arms * p.t_max_n * p.motor_force_scale
        m.dof_damping[self.motor_d] = p.motor_damping()
        k = p.joint_stiffness()
        for a in range(p.n_arms):
            for j in range(len(p.pitch_m)):
                m.jnt_stiffness[m.joint(f"a{a}_j{j}").id] = k[j]

    def set_degradation(self, force_scale, stretch_m):
        """Degraded actuation at run time: motor force (and no-load speed) scale, tendon elongation; spine unchanged."""
        m, p = self.model, self.p
        p.motor_force_scale, p.cable_stretch_m = float(force_scale), float(stretch_m)
        m.actuator_gear[self.motor_act, 0] = p.n_arms * p.t_max_n * p.motor_force_scale
        for a in range(p.n_arms):
            m.tendon_range[m.tendon(f"cable{a}").id] = [-p.cable_stretch_m, p.cable_slack_m]   # pull side shifts

    def set_parcel(self, sides, mass):
        """Resize the parcel box (half sides) and set mass and inertia consistently."""
        m, g, b = self.model, self.parcel_geom, self.parcel_body
        h = np.asarray(sides, float) / 2
        m.geom_size[g] = h
        m.geom_rbound[g] = float(np.linalg.norm(h))
        m.geom_aabb[g] = np.concatenate(([0, 0, 0], h))
        m.body_mass[b] = mass
        m.body_inertia[b] = mass / 3 * np.array([h[1] ** 2 + h[2] ** 2, h[0] ** 2 + h[2] ** 2, h[0] ** 2 + h[1] ** 2])
        mujoco.mj_setConst(m, self.data)       # the mass matrix of simple bodies is precomputed: refresh it
