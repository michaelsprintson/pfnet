import numpy as np
import copy

class ref_pf():
    # def __init__(self, init_part_states, init_part_weights, x, y, params):
    def __init__(self, params):
        # self.state = [init_part_states, init_part_weights]
        # self.x = x
        # self.y = y
        self.params = params
    
    def observation(self, p_states, observation):
        p_pos = p_states[:,0] + (p_states[:,1] * self.params['time_step']) + (0.5 * p_states[:,2] * self.params['time_step']**2)
        diff_val = abs(p_pos - observation) ** 2
        # return (diff_val / sum(diff_val))
        return (1 / diff_val) / sum(1 / diff_val)
    
    def systematic_resample(self, weights):
        N = len(weights)

        # make N subdivisions, choose positions 
        # with a consistent random offset
        positions = (np.arange(N) + np.random.random()) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def resample_from_index(self, particles, weights):
        new_particles = copy.deepcopy(particles)
        new_weights = copy.deepcopy(weights)
        indexes = self.systematic_resample(weights)
        new_particles[:] = particles[indexes]
        new_weights.resize(len(particles))
        new_weights.fill(1.0 / len(weights))
        return new_particles, new_weights


    def transition(self, p_states):
        pos_std = self.params['transition_std'][0]# position std
        vel_std = self.params['transition_std'][1]# velocity std
        acc_std = self.params['transition_std'][2]# acceleration std

        part_y = p_states[:,0]
        part_v = p_states[:,1]
        part_a = p_states[:,2]

        noise_y = np.random.normal(size = part_y.shape, loc=0.0, scale=1.0) * pos_std
        noise_v = np.random.normal(size = part_v.shape, loc=0.0, scale=1.0) * vel_std
        noise_a = np.random.normal(size = part_a.shape, loc=0.0, scale=1.0) * acc_std

        delta_y = part_y + (part_v * self.params['time_step']) + (0.5 * part_a * self.params['time_step']**2) + noise_y
        delta_v = part_v + (part_a * self.params['time_step']) + noise_v
        delta_a = part_a + noise_a

        new_p_states = np.zeros(p_states.shape)
        new_p_states[:,0] = delta_y
        new_p_states[:,1] = delta_v
        new_p_states[:,2] = delta_a
        return new_p_states