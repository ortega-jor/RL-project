import os
os.environ["MUJOCO_GL"] = "glfw"  # Use GLFW backend for GPU rendering on Windows
import mujoco
import gymnasium as gym
import time

# ------------------- FUNCTIONS ---------------------------#
# Open gui and run for # of steps
def gui_test(steps):
    for _ in range(steps):
        action = env.action_space.sample()  # Sample random actions
        # print(env.action_space)
        obs, reward, done, truncated, info = env.step(action)
        env.render()  # Explicitly call render to ensure visualization
        if done:
            env.reset()
    env.close()

# Print parameters in a concise table format
def print_model_parameters(model):
    print("=== Walker2d-v5 Model Parameters ===")
    
    # Body Parameters
    print("\nBody Parameters:")
    print(f"{'ID':<4} {'Name':<20} {'Mass (kg)':<15} {'Center of Mass (x, y, z)':<30}")
    print("-" * 70)
    for i in range(model.nbody):
        body_name = model.body(i).name
        name = body_name.decode('utf-8') if isinstance(body_name, bytes) else body_name if body_name else f"body_{i}"
        mass = model.body_mass[i]
        com = model.body_ipos[i]
        print(f"{i:<4} {name:<20} {mass:<15.4f} {str(com):<30}")

    # Geometry Parameters (no Type)
    print("\nGeometry Parameters:")
    print(f"{'ID':<4} {'Name':<20} {'Size':<25}")
    print("-" * 50)
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        name = geom_name.decode('utf-8') if isinstance(geom_name, bytes) else geom_name if geom_name else f"geom_{i}"
        size = model.geom_size[i]
        print(f"{i:<4} {name:<20} {str(size):<25}")

    # Joint Parameters (no Type, no Initial Angles)
    print("\nJoint Parameters:")
    print(f"{'ID':<4} {'Name':<20} {'Limits (rad)':<25}")
    print("-" * 50)
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        name = joint_name.decode('utf-8') if isinstance(joint_name, bytes) else joint_name if joint_name else f"joint_{i}"
        limits = model.jnt_range[i]
        print(f"{i:<4} {name:<20} {limits[0]:>5.2f} to {limits[1]:<5.2f}")

# Function to print and modify geometry parameters
def modify_geometry_parameters(model):
    # Modify link lengths (example: double the length of thigh and leg) MuJoCo uses capsules which is made up of [radius, hal-length]
    model.geom_size[1] = [0.05, 0.05, 0.]  # torso_geom: 
    model.geom_size[2], model.geom_size[5] = [0.05, 0.225, 0.], [0.05, 0.225, 0.]  # thigh_geom, thigh_left_geom : [radius, hal-length, 0.]
    model.geom_size[3], model.geom_size[6] = [0.04, 0.25, 0.], [0.04, 0.25, 0.]  # leg_geom, leg_left_geom: [radius, hal-length, 0.]
    model.geom_size[4], model.geom_size[7] = [0.06, 0.1, 0.], [0.06, 0.1, 0.]  # foot_geom, foot_left_geom: [radius, hal-length, 0.]

    # Adjust joint positions so they are correctly placed
    # model.body_pos[2], model.body_pos[5] = [0, 0, -0.22], [0, 0, -0.22]  # thigh position (half thigh length)
    # model.body_pos[3], model.body_pos[6] = [0, 0, -0.72], [0, 0, -0.72]  # leg position (at knee, full thigh length)
    # model.body_pos[4], model.body_pos[7] = [0, 0, -0.81], [0, 0, -0.81]  # foot position (at ankle, full leg length)

def modify_mass_parameters(model):
    # Modify the mass of specific body parts
    model.body_mass[1] = 4.15  # Set mass of torso 
    model.body_mass[2], model.body_mass[5] = 0.6, 0.6  # Set mass of thigh
    model.body_mass[3], model.body_mass[6] = 0.3, 0.3  # Set mass of leg
    model.body_mass[4], model.body_mass[7]= 0.1, 0.1  # Set mass of foot

def modify_positions(model):
    # Joint positions are relative to the parent body they connect to
    torso_length = 2 * model.geom_size[1][1]  # 0.1
#     thigh_length = 2 * model.geom_size[2][1]  # 0.22
#     leg_length = 2 * model.geom_size[3][1]    # 0.40

    # Body positions: Move each body to the end of the previous geom    
    model.body_pos[2] = [0, 0, -torso_length]    # thigh (below torso)
    model.body_pos[5] = [0, 0, -torso_length]    # thigh_left
#     model.body_pos[3] = [0, 0, -thigh_length]    # leg (below thigh)
#     model.body_pos[6] = [0, 0, -thigh_length]    # leg_left
#     model.body_pos[4] = [0, 0, -leg_length]      # foot (below leg)
#     model.body_pos[7] = [0, 0, -leg_length]      # foot_left

    # Joint positions: Keep at body origin since body_pos handles offsets
    model.jnt_pos[1] = [0, 0, -torso_length]  # thigh_joint (hip)
    model.jnt_pos[4] = [0, 0, -torso_length]  # thigh_left_joint
#     model.jnt_pos[2] = [0, 0, -thigh_length]  # leg_joint (knee)
#     model.jnt_pos[5] = [0, 0, -thigh_length]  # leg_left_joint
#     model.jnt_pos[3] = [0, 0, -leg_length]  # foot_joint (ankle)
#     model.jnt_pos[6] = [0, 0, -leg_length]  # foot_left_joint
## ----------------------------------------------------------------------------------------- ##


#******************************************** MAIN ***************************************#
# Create the MuJoCo environment
env = gym.make("Walker2d-v5", render_mode="human")
env.reset()
model = env.unwrapped.model # Access the MuJoCo model

# Modify model geometry (link lenghts)
modify_geometry_parameters(model)
modify_mass_parameters(model)
# modify_positions(model)

########## Printing for test ##########
#print("Initial state from env:", env.unwrapped.data.qpos)
print_model_parameters(model)
gui_test(5000) #open gui and run for 500 steps to visualize

#****************************************************************************************#

env.close()