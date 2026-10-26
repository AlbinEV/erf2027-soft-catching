# Technical details

Values used in the simulations and tables cited by the paper. The code, trained networks, configurations and results of
every test are in this repository (`catch/config.py`, `runs/`).

## Model and test parameters (design values)

| Element | Value |
|---|---|
| Hand | 4 arms of 9 phalanges on an elastic spine; stops at 20° at the root and 3 × 30° at the corner; dorsal stops at 30° (rigid hand) or 0° (designed hand); 0.5 kg |
| Motor | maximum force and no-load speed: 20 N per arm, tendon at 0.25 m/s; closing to 95 % of the stops in 0.19 s (0.13 s with dorsal stops at 0°); 0.27 / 0.39 / 0.76 s at 0.125 / 0.0625 / 0.03 m/s |
| Compliance | palm: 5 cm stroke, 8 N s/m (first hand) or 40 N s/m (designed hand), 100 N/m spring; wrist: 15 or 10 cm stroke, 80 N s/m, 150 N/m spring |
| Degradations | motor 0.03–0.25 m/s; force and no-load speed scaled down to 50 % (supply voltage); tendon stretched up to 4 mm, taken up by the motor before it pulls (with 4 mm the stops are not reached) |
| Nominal throws | release 1.8–2.6 m in front of the robot, ±0.4 m sideways, 1.1–1.6 m high; aimed at the hand with 12 cm error per axis; flight 0.5–0.8 s (3.4–4.6 m/s at release); spin up to 8 rad/s; boxes of 8–10 cm, 0.1–0.4 kg |
| Varied throws | release 1–4 m from the aim point, azimuth ±45°, flight 0.3–1.0 s |
| Ideal camera | parcel centre seen everywhere; 60 Hz, 33 ms latency, 5 mm noise, 2 % of frames lost |
| Realistic camera | at (−0.3, −0.6, 2.0) m, pointing at (1.2, 0, 1.3) m; field of view 87° × 58°, range 0.3–6 m; frame lost if the robot or the hand hides the parcel centre; axial noise 5 mm + 0.0025 z² (m, z in m); calibration error fixed per throw; random position: x from −0.6 to 1.0 m, \|y\| from 0.5 to 1.6 m, z from 1.4 to 2.3 m |
| Tests | 500 throws per condition (1000 or 3000 where stated), never used to train or select the models; 95 % Wilson intervals, paired exact McNemar test |

## Trained networks

Students and executors: DAgger, network 62 → 256 → 256 → 7 (67 inputs for the executors). Supervisors: PPO, 100 000
throws, network 21 → 64 → 64 → 5, with a reward that favours the catch and penalises force and accelerations. The
policies command the palm velocity and the motor at 100 Hz.

| Network (folder in `runs/`) | Training conditions |
|---|---|
| student (`dagger_v1`, `dagger_proto`) | rigid or designed hand; nominal throws, ideal camera; 600 teacher throws + 12 iterations of 300 |
| executor (`dagger_theta2`, `dagger_theta_deg`) | random catch parameters; mass 0.1–0.8 kg, flight 0.45–0.8 s, aim 6–20 cm; motor 0.12–0.25 m/s (rigid hand) or degradations (designed hand); 2000 + 16 × 1000 throws |
| supervisor (`hi2_cse`, `hi2_deg`) | as the executor; camera, parcel size and encoder |
| student, realistic camera (`dagger_proto_cam`, `dagger_proto_campos`) | fixed or random position; random noise, calibration (up to 2.5 cm and 2.5°), latency (33–70 ms) and lost frames (0–12 %) |
| student, varied throws (`dagger_proto_wide`, `dagger_proto_all_long`) | varied throws; ideal camera, or realistic camera in a random position with random errors (3000 + 20 × 1200 throws) |

Supervisors of the rigid hand, sensor sets: camera only; also parcel size and motor encoder; also a wrist force sensor.
A second training gives the same results within one point, and a rule with their mean parameters and the closing ratio
scaled from the encoder stays within 2 points.

## Classical pipeline

The measurement noise of the Kalman filter is chosen on the selection throws between 5 mm and 16 cm (best 6 cm,
99.2 %). With the initial value of 5 mm, equal to the camera noise, the classical pipeline catches 94.2 % of the nominal
throws (88.0 % with the rigid hand). Tuned for each hard condition, it gives 91.1 % on the varied throws (5 mm), 95.0 %
with 2 cm of camera noise (15 cm), 96.8 % with the degraded camera (15 cm) and 82.9 % on the varied throws with the
camera in a random position (5 mm).

## Parcel shapes

Rigid hand, 300 throws per shape. The cage is designed for a 10 cm cube, with which the catches are 98–99 %. The hand
catches 80–94 % of the cubes of 8–14 cm and of the boxes up to 20 × 10 × 5 cm, while small parcels pass between the
arms (6 cm cube: 52 %).

## Designed hand with degraded actuation

Catches (%) over 500 throws; in brackets the true closing time. Teacher with the true time; student trained in nominal
conditions; rules and supervisors with the executor retrained on the degradations.

| Condition | Teacher | Student | Classical | Fixed rule | Encoder rule | RL (2 seeds) |
|---|---|---|---|---|---|---|
| nominal (0.13 s) | 100 | 99.8 | 94.2 | 100 | 100 | 100 / 100 |
| motor 50 % (0.27 s) | 99.8 | 100 | 93.8 | 100 | 99.8 | 100 / 100 |
| motor 25 % (0.39 s) | 99.8 | 99.0 | 92.6 | 99.2 | 99.4 | 99.8 / 99.8 |
| motor 12 % (0.76 s) | 93.8 | 92.2 | 90.8 | 97.6 | 98.0 | 96.0 / 98.0 |
| voltage 75 % (0.22 s) | 99.4 | 100 | 94.6 | 100 | 100 | 100 / 100 |
| voltage 50 % (0.32 s) | 99.4 | 100 | 93.4 | 99.8 | 99.2 | 100 / 100 |
| tendon +2 mm (0.20 s) | 100 | 100 | 94.6 | 100 | 100 | 100 / 100 |
| tendon +4 mm | 99.6 | 99.8 | 94.0 | 99.2 | 99.4 | 99.8 / 99.8 |
| motor 50 %, voltage 75 %, tendon +2 mm (0.39 s) | 99.8 | 99.6 | 92.8 | 99.2 | 99.4 | 100 / 99.8 |
| motor 25 %, voltage 75 %, tendon +2 mm (0.66 s) | 98.0 | 96.2 | 91.4 | 98.2 | 99.0 | 98.2 / 99.6 |

## Varied throws with the ideal camera

3000 throws: catches (%) by direction, distance, flight time and release speed.

| | Teacher | Classical | Student (nominal) | Student (varied) |
|---|---|---|---|---|
| azimuth 0–15 / 15–30 / 30–45° | 98 / 98 / 98 | 92 / 90 / 91 | 89 / 78 / 44 | 98 / 98 / 97 |
| distance 1–2 / 2–3 / 3–4 m | 99 / 99 / 96 | 92 / 92 / 90 | 86 / 72 / 53 | 99 / 98 / 96 |
| flight 0.3–0.45 / 0.45–0.6 / 0.6–1 s | 93 / 99 / 100 | 84 / 93 / 93 | 36 / 61 / 85 | 90 / 99 / 100 |
| speed < 6 / 6–8 / > 8 m/s | 99 / 98 / 85 | 93 / 91 / 76 | 82 / 42 / 19 | 99 / 97 / 82 |
| all | 98.1 | 91.1 | 70.2 | 97.6 |

## Rigid hand: supervisors against the nominal fixed rule

The figure `figures/fig_rl_delta_en.pdf` shows the difference in catches with respect to the nominal fixed rule
(closing ratio 0.45), on the same 500 throws per condition, with paired 95 % interval. The motor at 0.10 m/s is outside
the training range.
