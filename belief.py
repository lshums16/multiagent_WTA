import numpy as np

class Belief:
    def __init__(self, weapon_effect_dict, target_ids):
        self.agent_estimates = {}
        for id in weapon_effect_dict:
            self.agent_estimates[id] = {"assignment": None,
                                        "weapon_effectiveness": weapon_effect_dict[id],
                                        "attrition_probability": None,
                                        "num_hops": np.inf
                                        }
        
        self.target_kill_prob = {}
        for id in target_ids:
            self.target_kill_prob[id] = 0
                
    def calc_kill_prob(self, seekers):
        product = 1
        
        for agent in seekers: # seekers is a list of seekers targeting the current target
            product *= (1 - agent["weapon_effectiveness"] + agent["weapon_effectiveness"]*agent["attrition_probability"])
            
        return 1 - product
            
    def update_kill_probabilities(self):
        for target_id in self.target_kill_prob:
            seekers = [self.agent_estimates[agent_est] for agent_est in self.agent_estimates if self.agent_estimates[agent_est]['assignment'] == target_id] # get a list of agents who you belief are targeting the target
            self.target_kill_prob[target_id] = self.calc_kill_prob(seekers)
            
    def update_agent_estimate(self, agent_id, target_id, attrition_prob, num_hops):
        self.agent_estimates[agent_id]["assignment"] = target_id
        self.agent_estimates[agent_id]["num_hops"] = num_hops
        self.agent_estimates[agent_id]["attrition_probability"] = attrition_prob
        
    def reset_hops(self, agent_id):
        for est_id in self.agent_estimates:
            if est_id != agent_id:
                self.agent_estimates[est_id]["num_hops"] = np.inf
        
        
    
        
    