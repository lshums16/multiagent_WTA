import numpy as np

class Agent:
    def __init__(self, id, pos, heading, agent_params, weapon_effectiveness, Ts):
        # self.max_glide_ratio = 1.
        # self.weapon_effectiveness = 0.75 # TODO: may be on an agent-target basis
        # self.attrition = 0.75 # TODO: may be on an agent-target basis
        # self.pa = 0.00004
        self.id = id
        self.state = np.array([pos.item(0), pos.item(1), agent_params["agent_velocity"], heading])
        
        self.alt = agent_params["agent_spawn_alt"]
        self.max_psidot = agent_params["max_psidot"]
        self.max_glide_ratio = agent_params["max_glide_ratio"]
        
        self.Ts = Ts
        self.kp_psi = 1. # TODO: tune
        
        self.weapon_effectiveness = weapon_effectiveness
    
    def assign_target(self, target):
        self.target = target
       
    # # needs to be done every time-step 
    # def update_attrition(self):
    #     dij = np.linalg.norm(target.pos - seeker.pos) # this is 2D distance
    #     d_int = dij/da # TODO: define da
    #     self.attrition =  1 - (1 - self.pa)**d_int
        
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
        
        