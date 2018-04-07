import numpy as np
import csv
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

max_position = 300.0  # meters
max_angles = (22. / 7. / 180.) * 45  # 45 degrees
max_velocity = 10.0
max_angle_velocity = (22. / 7. / 180.) * 45  # 45 degrees/s

def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def earth_to_body_frame(ii, jj, kk):
    # C^b_n
    R = [[C(kk) * C(jj), C(kk) * S(jj) * S(ii) - S(kk) * C(ii), C(kk) * S(jj) * C(ii) + S(kk) * S(ii)],
         [S(kk) * C(jj), S(kk) * S(jj) * S(ii) + C(kk) * C(ii), S(kk) * S(jj) * C(ii) - C(kk) * S(ii)],
         [-S(jj), C(jj) * S(ii), C(jj) * C(ii)]]
    return np.array(R)


def body_to_earth_frame(ii, jj, kk):
    # C^n_b
    return np.transpose(earth_to_body_frame(ii, jj, kk))



class PhysicsSim():
    def __init__(self, reset_state=None):

        if reset_state is not None:
            self.reset_state = reset_state
        else:
            self.reset_state = None

        self.n_rotors = 4
        self.max_rotor_speed = 900.0

        self.gravity = -9.81  # m/s
        self.rho = 1.2
        self.mass = 0.958  # 300 g
        self.dt = 1 / 100.0  # Timestep
        self.C_d = 0.3
        self.l_to_rotor = 0.4
        self.propeller_size = 0.1
        width, length, height = .51, .51, .235
        self.dims = np.array([width, length, height])  # x, y, z dimensions of quadcopter
        self.areas = np.array([length * height, width * height, width * length])
        I_x = 1 / 12. * self.mass * (height**2 + width**2)
        I_y = 1 / 12. * self.mass * (height**2 + length**2)  # 0.0112 was a measured value
        I_z = 1 / 12. * self.mass * (width**2 + length**2)
        self.moments_of_inertia = np.array([I_x, I_y, I_z])  # moments of inertia

        self.max_position = np.array([max_position / 2, max_position / 2, max_position])
        self.min_position = np.array([-max_position / 2, -max_position / 2, 0])

        self.reset()

    def reset(self, state=None):
        self.time = 0.0
        # if self.reset_state == 'ten-up':
        if True:
            self.runtime = 5.0
            self.pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
            self.v = np.array([+10.0, 0.0, 0.0])
            self.angular_v = np.array([0.0, 0.0, 0.0])
        else:
            self.runtime = np.inf
            self.pose = np.concatenate((
                np.random.uniform(self.min_position, self.max_position),
                np.random.uniform(-max_angles, max_angles, 3)))
            self.pose[5] = 0.0
            self.v = np.random.uniform(-max_velocity, max_velocity, 3)
            self.angular_v = np.random.uniform(-max_angle_velocity, max_angle_velocity, 3)
            self.angular_v[2] = 0.0

        self.linear_accel = np.array([0.0, 0.0, 0.0])
        self.angular_accels = np.array([0.0, 0.0, 0.0])
        self.prop_wind_speed = np.array([0., 0., 0., 0.])

        self.done = False
        self.info = dict(crash=False)


    def find_body_velocity(self):
        body_velocity = np.matmul(earth_to_body_frame(*list(self.pose[3:])), self.v)
        return body_velocity

    def get_linear_drag(self):
        linear_drag = 0.5 * self.rho * self.find_body_velocity()**2 * self.areas * self.C_d
        return linear_drag

    def get_linear_forces(self, thrusts):
        # Gravity
        gravity_force = self.mass * self.gravity * np.array([0, 0, 1])
        # Thrust
        thrust_body_force = np.array([0, 0, sum(thrusts)])
        # Drag
        drag_body_force = -self.get_linear_drag()
        body_forces = thrust_body_force + drag_body_force

        linear_forces = np.matmul(body_to_earth_frame(*list(self.pose[3:])), body_forces)
        linear_forces += gravity_force
        return linear_forces

    def get_moments(self, thrusts):
        thrust_moment = np.array([(thrusts[3] - thrusts[2]) * self.l_to_rotor,
                            (thrusts[1] - thrusts[0]) * self.l_to_rotor,
                            0])# (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1]) * self.T_q])  # Moment from thrust

        drag_moment =  self.C_d * 0.5 * self.rho * self.angular_v * np.absolute(self.angular_v) * self.areas * self.dims * self.dims
        moments = thrust_moment - drag_moment # + motor_inertia_moment
        return moments

    def calc_prop_wind_speed(self):
        body_velocity = self.find_body_velocity()
        phi_dot, theta_dot = self.angular_v[0], self.angular_v[1]
        s_0 = np.array([0., 0., theta_dot * self.l_to_rotor])
        s_1 = -s_0
        s_2 = np.array([0., 0., phi_dot * self.l_to_rotor])
        s_3 = -s_2
        speeds = [s_0, s_1, s_2, s_3]
        for num in range(4):
            perpendicular_speed = speeds[num] + body_velocity
            self.prop_wind_speed[num] = perpendicular_speed[2]

    def get_propeler_thrust(self, rotor_speeds):
        '''calculates net thrust (thrust - drag) based on velocity
        of propeller and incoming power'''
        thrusts = []
        for prop_number in range(4):
            V = self.prop_wind_speed[prop_number]
            D = self.propeller_size
            n = rotor_speeds[prop_number]
            J = V / n * D
            # From http://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
            C_T = max(.12 - .07*max(0, J)-.1*max(0, J)**2, 0)
            thrusts.append(C_T * self.rho * n**2 * D**4)
        return thrusts

    def next_timestep(self, rotor_speeds):
        self.calc_prop_wind_speed()
        thrusts = self.get_propeler_thrust(rotor_speeds)
        self.linear_accel = self.get_linear_forces(thrusts) / self.mass

        position = self.pose[:3] + self.v * self.dt + 0.5 * self.linear_accel * self.dt**2
        self.v += self.linear_accel * self.dt

        moments = self.get_moments(thrusts)

        self.angular_accels = moments / self.moments_of_inertia
        angles = self.pose[3:] + self.angular_v * self.dt + 0.5 * self.angular_accels * self.angular_accels * self.dt
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        self.angular_v = self.angular_v + self.angular_accels * self.dt

        new_positions = []
        for ii in range(3):
            if position[ii] <= self.min_position[ii]:
                new_positions.append(self.min_position[ii])
                self.info['crash'] = True
                self.done = True
            elif position[ii] > self.max_position[ii]:
                new_positions.append(self.max_position[ii])
                self.info['crash'] = True
                self.done = True
            else:
                new_positions.append(position[ii])

        self.pose = np.array(new_positions + list(angles))
        self.time += self.dt

        # Not required in AI Gym. Gym times out according to the registry.
        # if self.time > self.runtime:
        #     self.info['timeout'] = True
        #     self.done = True


class QuadCopter(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):

        self.sim = PhysicsSim()

        self.action_space = spaces.Box(
            low=-self.sim.max_rotor_speed, high=self.sim.max_rotor_speed,
            shape=(self.sim.n_rotors,), dtype=float)

        high = self._get_obs() * 1.e10
        self.observation_space = spaces.Box(low=-high, high=high, dtype=float)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        rotor_speeds = u
        done = self.sim.next_timestep(rotor_speeds)
        reward = self._get_reward()

        return self._get_obs(), reward, self.sim.done, self.sim.info

    def reset(self, **kwargs):
        self.sim.reset()

    def _get_reward(self):
        reward = 1.0
        return reward

    def _get_obs(self):
        state = np.concatenate((
            self.sim.pose[:3],  # positions
            self.sim.pose[3:],  # angles
            self.sim.v,
            self.sim.angular_v,
            self.sim.linear_accel,
            self.sim.angular_accels,
            ))
        return state
