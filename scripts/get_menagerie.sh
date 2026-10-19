#!/bin/bash
# Fetch the UR5e model of MuJoCo Menagerie (BSD-3-Clause) at the commit used for the paper. The model file is
# used unmodified; catch/model.py composes the scene from it.
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --filter=blob:none https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
git -C third_party/mujoco_menagerie checkout --quiet 62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35
echo "MuJoCo Menagerie at $(git -C third_party/mujoco_menagerie rev-parse --short HEAD)"
