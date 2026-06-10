#!/usr/bin/env python3
"""Check what joints are actually available in G1 after building"""

import genesis as gs

# Initialize Genesis
gs.init(backend=gs.cpu)

# Create scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02),
    show_viewer=False,
)

# Add plane
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# Add G1
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/home/sehaj/one-leg-robot/rl-training-genesis/models/g1_12dof.urdf",
        pos=(0.0, 0.0, 1.0),
    ),
)

# Build scene
print("Building scene...")
scene.build(n_envs=1)

# Now check what joints exist
print(f"\nRobot has {robot.n_joints} joints total")
print(f"Robot has {robot.n_dofs} DOFs total\n")

print("All joint names:")
print("="*60)
for i in range(robot.n_joints):
    joint = robot.joints[i]
    print(f"{i:2d}. {joint.name:30s} | DOF start: {joint.dof_start:2d} | n_qs: {joint.n_qs}")

print("\n" + "="*60)
print("Controllable joints (dof_start >= 0):")
print("="*60)
for i in range(robot.n_joints):
    joint = robot.joints[i]
    if joint.dof_start >= 0:
        print(f"{joint.name:30s} | DOF: {joint.dof_start}")

print("\nDone!")
