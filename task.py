import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1 # 3

        self.state_size = self.action_repeat * 1 # 6 --> 1 (only z-position)
        self.action_low = 0 # 0 --> 400 for takeoff/hover
        self.action_high = 900 #900
        self.action_size =  1 # 4 propellers to have the same thurst for takeoff/hover
        self.max_z_distance = self.sim.upper_bounds[2] - self.sim.lower_bounds[2]

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.max_error_pos = 10.0 # distance unit


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 1.-.3*(abs(self.sim.pose[2] - self.target_pos[2]))
        # reward = 1.-.3*(np.linalg.norm(self.sim.pose[:3] - self.target_pos)).sum()

        # Scale reward to [-1, 1]
#         reward = 1 - abs(self.sim.pose[2] - self.target_pos[2]) / self.max_z_distance # [0, 1]
#         reward = - (z_distance*2) #- abs(self.sim.v[2]) * 0.01
        # print('z={:3.2f}, z_distance={:3.2f}, reward={:3.2f}'.format(self.sim.pos[2], z_distance, reward))

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = rotor_speeds * 4 # put the same thurst for all 4 propellers
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose[2]) # z-pos only
        next_state = np.array(pose_all) #np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = np.array([self.sim.pose[2]] * self.action_repeat)
        return state