import numpy as np
import copy
from belief import Belief

class Agent:
    def __init__(self, id, pos, heading, agent_params, weapon_effectiveness_dict, target_ids, Ts):
        # self.max_glide_ratio = 1.
        # self.weapon_effectiveness = 0.75 # TODO: may be on an agent-target basis
        # self.attrition = 0.75 # TODO: may be on an agent-target basis
        # self.pa = 0.00004
        self.id = id
        self.state = np.array([pos.item(0), pos.item(1), agent_params["agent_velocity"], heading])
        
        self.alt = agent_params["agent_spawn_alt"]
        self.max_psidot = agent_params["max_psidot"]
        self.max_glide_ratio = agent_params["max_glide_ratio"]
        self.da = agent_params["num_attrition_sections"]
        self.pa = agent_params["pa"]
        self.collision_buffer = agent_params["collision_buffer"]
        
        self.Ts = Ts
        self.kp_psi = 3. # TODO: tune
        
        self.weapon_effectiveness = weapon_effectiveness_dict[id]
        
        self.prev_state = copy.copy(self.state)
        
        self.belief = Belief(weapon_effectiveness_dict, target_ids)
            
    def assign_target(self, target):
        if self.is_reachable(target):
            self.target = target
            self.calc_attrition()
            self.belief.update_agent_estimate(self.id, self.target.id, self.attrition_prob, num_hops = 0)
            return True
        return False
    
    # TODO: def update_estimates(self):
        
        
    
    def is_reachable(self, target):
        v = self.state[2]
        zdot = v/np.linalg.norm(self.state[:2] - target.pos)*self.alt

        if abs(v/zdot) > self.max_glide_ratio:
            return False
        else:
            return True
    
    def decision(self, probability):
        rand = np.random.random()
        return rand < probability
    
    def receive_belief(self, rec_belief):
        for est_id, est_values in rec_belief.agent_estimates.items():
            if est_values['num_hops'] + 1 < self.belief.agent_estimates[est_id]['num_hops']:
                    self.belief.update_agent_estimate(est_id, est_values['assignment'], est_values['attrition_probability'], est_values['num_hops'] + 1)

    
    def check_collision(self):
        if np.linalg.norm(self.target.pos - self.state[:2]) < self.collision_buffer:
            return True, self.decision(self.weapon_effectiveness)
        
        # check to see if agents collided between time steps
        relative_speed = np.linalg.norm(np.zeros(2) - self.state[2]) # assumes target is stationary
        check_buffer = relative_speed*self.Ts # this is the minimum distance the simulation at this time step is able to detect
        
        if np.linalg.norm(self.target.pos - self.state[:2]) < check_buffer*1.1: # if the distance between the two agents is less than the buffer (plus 50% cushion)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot([self.prev_state[0], self.state[0]], [self.prev_state[1], self.state[1]])
            # plt.scatter(self.target.pos[0], self.target.pos[1])
            # plt.show(block=False)
            
            new_ts = int(relative_speed/self.collision_buffer) + 1
            agent1_pos_list = np.linspace(self.prev_state[:2], self.state[:2], new_ts)
            agent2_pos_list = np.array([self.target.pos for i in range(len(agent1_pos_list))]) # assumes stationary target
            
            distance = np.zeros(len(agent1_pos_list))
            for i in range(new_ts):
                distance[i] = np.linalg.norm(agent1_pos_list[i] - agent2_pos_list[i]) # only needed for plotting
                if np.linalg.norm(agent1_pos_list[i] - agent2_pos_list[i]) < self.collision_buffer:
                    return True, self.decision(self.weapon_effectiveness)
        
        return False, False
    
    def calc_attrition(self):
        dij = np.linalg.norm(self.target.pos - self.state[:2]) # this is 2D distance
        d_int = dij/self.da 
        self.attrition_prob =  1 - (1 - self.pa)**d_int
        
        # update current estimate of self 
        self.belief.update_agent_estimate(self.id, self.target.id, self.attrition_prob, num_hops = 0)
    
    def check_attrition(self):
        self.calc_attrition()
        return self.decision(self.attrition_prob)
            
        
    def update_dynamics(self):
        target_pos = self.target.pos[:2]
        seeker_pos = self.state[:2]
        
        crs_cmd = np.arctan2(target_pos.item(1) - seeker_pos.item(1), target_pos.item(0) - seeker_pos.item(0))
        
        self.RK4([crs_cmd])
        
        self.state[3] = self.bound_angle(self.state[3])
        
    def bound_angle(self, angle):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle <= -np.pi:
            angle += 2*np.pi
        return angle
    
    def derivatives(self, state, crs_cmd):
        
        n, e, V, psi = state
        
        n_dot = V*np.cos(psi)
        e_dot = V*np.sin(psi)
        V_dot = 0
        
        crs_error = self.bound_angle(crs_cmd - psi)
        psi_dot = self.kp_psi*crs_error
            
        psi_dot = self.saturate(psi_dot)
        
        return np.array([n_dot, e_dot, V_dot, psi_dot])
        
    def saturate(self, psidot_cmd):
        if abs(psidot_cmd) > self.max_psidot:
            return np.sign(psidot_cmd)*self.max_psidot
        else:
            return psidot_cmd
        
    def RK4(self, inputs):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = self.derivatives(self.state, *inputs)
        k2 = self.derivatives(self.state + self.Ts/2.*k1, *inputs)
        k3 = self.derivatives(self.state + self.Ts/2.*k2, *inputs)
        k4 = self.derivatives(self.state + self.Ts*k3, *inputs)
        self.state += self.Ts/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        