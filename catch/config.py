"""Parameters of the catching simulation. All values are design assumptions, not measured properties.

Hand geometry, stops and tendon guide are those of the printed four-arm prototype. The motor is a design
assumption, faster and stronger than the prototype's: closing in about 0.1-0.2 s, and enough tension to hold the
preset shape under gravity. Robot, parcels, throws and camera are new
assumptions of this project.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    # --- hand geometry (printed prototype arm) ----------------------------------------------------------------------------
    pitch_m: tuple = (0.016, 0.02061, 0.02061, 0.02061, 0.02061, 0.014, 0.014, 0.014, 0.03773)
    stop_deg: tuple = (20.0, 0.0, 0.0, 0.0, 0.0, 30.0, 30.0, 30.0, 0.0)
    width_m: tuple = (0.042, 0.040, 0.038, 0.036, 0.034, 0.032, 0.030, 0.028, 0.026)
    phalanx_h_m: float = 0.0148
    root_radius_m: float = 0.0728
    mount_deg: float = 20.0                 # open arm leans outwards by this angle from the hand axis
    outward_limit_deg: float = 30.0
    n_arms: int = 4
    guide_m: float = 0.007
    arm_mass_kg: float = 0.0255
    palm_half_m: float = 0.045
    palm_thick_m: float = 0.008
    housing_m: float = 0.04                 # flange to palm: motor and spool housing (assumption)
    housing_mass_kg: float = 0.30
    # --- optional passive compliance (design study of 18 Oct 2026; 0 = rigid, the nominal hand) --------------------
    wrist_stroke_m: float = 0.0             # slide between flange and hand along the hand axis
    wrist_k: float = 200.0                  # N/m
    wrist_c: float = 40.0                   # N s/m
    wrist_preload_m: float = 0.03           # spring pre-compression: keeps the hand extended under its weight
    pad_stroke_m: float = 0.0               # light palm plate on a spring, inside the hand
    pad_k: float = 100.0
    pad_c: float = 8.0
    pad_preload_m: float = 0.01
    pad_mass_kg: float = 0.02
    semi_active: bool = False               # variable dampers on pad and wrist (damping commanded at run time)
    pad_c_max: float = 200.0                # N s/m
    wrist_c_max: float = 400.0
    # --- hand actuation (ERF assumption: faster, stronger motor) ------------------------------------------------------
    t_max_n: float = 20.0                   # maximum closing tension per arm
    actuation_ratio: float = 3.0            # Pi = T_max / T_b, design rule; sets the spine stiffness
    stroke_min_m: float = 0.003
    stroke_max_m: float = 0.016
    close_speed_mps: float = 0.25           # stroke speed at full force: sets the motor damping
    motor_mass_kg: float = 0.02             # reflected inertia of motor and spool
    series_stiffness: float = 10000.0       # N/m per cable
    series_damping: float = 50.0
    cable_slack_m: float = 0.002
    motor_force_scale: float = 1.0          # degradation study (19 Oct): motor force and no-load speed scale together
    cable_stretch_m: float = 0.0            # degradation study: tendon elongation, taken up by the motor before it pulls
    joint_damping: float = 0.01
    joint_armature: float = 2e-6
    hand_friction: float = 0.8
    # --- robot ----------------------------------------------------------------------------------------------------------
    base_height_m: float = 0.80             # UR5e on a pedestal
    tcp_v_max: float = 1.5                  # m/s, commanded TCP speed limit
    tcp_w_max: float = 4.0                  # rad/s
    joint_v_max: float = 3.14               # rad/s (UR5e: 180 deg/s per joint)
    # --- parcels -------------------------------------------------------------------------------------------------------
    box_side_m: tuple = (0.08, 0.10)        # each side; the preset shape is sized for a 10 cm cube
    box_dims_m: tuple = ()                  # fixed parcel sides (box-shape study, 19 Oct 2026); empty = random sides
    box_mass_kg: tuple = (0.10, 0.40)
    box_friction: float = 0.6
    # --- throws (operator in front of the robot, along +x) ------------------------------------------------------------
    release_x: tuple = (1.8, 2.6)
    release_y: tuple = (-0.4, 0.4)
    release_z: tuple = (1.1, 1.6)
    release_polar: tuple = ()               # (r_min, r_max, az_min_deg, az_max_deg): release at horizontal distance r
                                            # from the aim point and azimuth az (0 = in front, +x) instead of release_x/y
    aim_dz: float = 0.30                    # the operator aims at the hand in the ready pose: (0.45, 0, base + 0.30)
    flight_s: tuple = (0.50, 0.80)
    aim_noise_m: float = 0.12               # operator inaccuracy, standard deviation per axis
    spin_max: float = 8.0                   # rad/s
    # --- camera (fixed depth camera tracking the parcel centre) ---------------------------------------------------------
    cam_rate_hz: float = 60.0
    cam_latency_s: float = 0.033
    cam_noise_m: float = 0.005
    cam_dropout: float = 0.02
    cam_model: str = "ideal"              # "ideal": parcel centre seen everywhere; "pinhole": camera with position, field
                                            # of view, occlusion by robot and hand, depth noise growing with distance
    cam_pos_m: tuple = (-0.3, -0.6, 2.0)    # pinhole camera (assumption): raised behind the robot, looking at the throws
    cam_target_m: tuple = (1.2, 0.0, 1.3)
    cam_fov_deg: tuple = (87.0, 58.0)       # horizontal, vertical (typical stereo depth camera)
    cam_range_m: tuple = (0.3, 6.0)
    cam_depth_k: float = 0.0025             # pinhole: axial noise std = cam_noise_m + k * z^2 (z in m)
    cam_calib_m: float = 0.0                # pinhole: calibration offset, fixed per throw, random direction (m)
    cam_calib_deg: float = 0.0              # pinhole: camera rotation error, fixed per throw, random axis (deg)
    cam_occlusion: bool = True              # pinhole: frame lost when a robot or hand geom hides the parcel centre
    cam_pos_random: tuple = ()              # pinhole: (x0, x1, |y|0, |y|1, z0, z1) camera position drawn per throw,
                                            # on a random side of the throwing lane, looking at cam_target_m
    cam2_pos_m: tuple = ()                  # pinhole: second camera (same target, own calibration error); the
                                            # measurements of the cameras that see the parcel are averaged
    air_cd: float = 0.0                     # parcel drag coefficient (0: no air); the operator aims allowing for it
    kf_meas_std: float = 0.0                # Kalman filter tuning: measurement std (m); 0 = from the camera model
    kf_acc_std: float = 0.5                 # Kalman filter tuning: process noise (unmodelled acceleration, m/s^2)
    # --- simulation ------------------------------------------------------------------------------------------------------
    dt_s: float = 0.001
    control_hz: float = 100.0
    extra: dict = field(default_factory=dict)

    @property
    def stop_rad(self):
        return np.radians(self.stop_deg)

    def spine_ei(self):
        th, p = self.stop_rad, np.asarray(self.pitch_m)
        t_b = self.t_max_n / self.actuation_ratio
        return t_b / np.max(th / (self.guide_m * p))

    def joint_stiffness(self):
        return self.spine_ei() / np.asarray(self.pitch_m)

    def motor_damping(self):
        return self.n_arms * self.t_max_n / self.close_speed_mps
