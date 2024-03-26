# TODO: have each agent calculate what it needs to after communication rounds are complete for a timestep
# TODO: tune attrition (da)?

from agent import Agent
from target import Target
import numpy as np
import matplotlib.pyplot as plt
import copy

def calc_EV(agents, targets):
    exp_val = 0
    for target in targets.values():
        kill_prob = target.calc_kill_prob([agent for agent in agents.values() if hasattr(agent, "target") and agent.target.id == target.id])
        exp_val += (1 - kill_prob)*target.value
        
    return exp_val

# Assumes same comms range for all agents
def update_adj_matrix(n, agents, comms_range):
    A = np.identity(n)
    for i in agents:
        for j in agents:
            if np.linalg.norm(agents[i].state[:2] - agents[j].state[:2]) <= comms_range:
                A[i, j] = 1
                A[j, i] = 1
    
    return A

def communicate(A, agents):
    for i in range(len(agents) - 1): # runs a communication round N - 1 times
        # dist_matrix = floyd_warshall(A)
        beliefs_dict = {}
        for agent_id in agents:
            beliefs_dict[agent_id] = copy.copy(agents[agent_id].belief)
            
        for agent1_id in agents:
            for agent2_id in agents:
                if agent1_id != agent2_id and A[agent1_id, agent2_id] != 0:
                    agents[agent2_id].receive_belief(beliefs_dict[agent1_id])
                    
    for agent in agents.values():
        agent.belief.reset_hops(agent.id) # this resets all hops (except agent's own) to infinity so that in the next round of communication, proper updates happen
        

target_spawn_dim = 750. # length (m) of the square within which the targets spawn randomly
agent_spawn_dim = 750. # length (m) of the square within which the agents spawn randomly
agent_target_spawn_dist = 500. # distance (m) between agent spawn square and target spawn square

comms_range = 600.

agent_params = {"agent_spawn_alt": 500.,
                "agent_velocity": 50.,
                "max_psidot": 1.265,
                "max_glide_ratio": 6.,
                "num_attrition_sections": 100, # TODO: what should I set this as?
                "pa": 0.00004,
                "collision_buffer": 5.0 # TODO: change?
                } 

round_ts = 1e-1
dec_ts = 10e-6
end_time = 100.0

num_targets = 5
des_kill_prob = 0.7 # For now, this is for all targets. Some simulations may need to create it separately

num_agents = 5
weapon_effectiveness_dict = {}
for i in range(num_agents):
    weapon_effectiveness_dict[i] = 0.9 # For now, this is for all targets. Some simulations may need to create is separately


# init targets   
active_targets = {}
for i in range(num_targets):
    pos = np.array([np.random.uniform(high = target_spawn_dim), np.random.uniform(high = target_spawn_dim)])
    active_targets[i] = Target(i, pos, des_kill_prob, value = 1) # TODO: value = des_kill_prob?


# init agents and assign targets 
active_agents = {}
for i in range(num_agents):
    pos = np.array([np.random.uniform(high = agent_spawn_dim), np.random.uniform(high = agent_spawn_dim) + agent_target_spawn_dist])
    
    heading = np.random.uniform(low = -np.pi, high = np.pi)
    
    agent = Agent(i, pos, heading, agent_params, weapon_effectiveness_dict, active_targets.keys(), round_ts) # assumes all agents have same effectiveness, but a different value can be put here if needed
    
    agent.assign_target(active_targets[i]) # TODO: fix later, but for now, assign Agent 1 to Target 1
    
    active_agents[i] = agent



inactive_targets = {} 
inactive_agents = {}

sim_time = 0
plt_init = False
while sim_time < end_time:
    A = update_adj_matrix(num_agents, active_agents, comms_range)
    communicate(A, active_agents)
    sim_time += round_ts
    plt.clf()
    for id, agent in active_agents.items():
        agent.update_dynamics()
        
        agent_collided, target_destroyed = agent.check_collision()
        if agent_collided:
            inactive_agents[id] = agent
            print(f"[{sim_time:.2f}]: Agent {id} has collided with Target {agent.target.id}")
            if target_destroyed:
                inactive_targets[agent.target.id] = agent.target
                print(f"[{sim_time:.2f}]: Target {agent.target.id} has been destroyed")
        
        elif agent.check_attrition():
            inactive_agents[id] = agent
            print(f"[{sim_time:.2f}]: Agent {id} has been attrited")
            
        agent.update_estimates()
        
        agent_pos = agent.state[:2]
        agent_heading = agent.state.item(3)
        
        # plot agent position
        plt.scatter(agent_pos.item(1), agent_pos.item(0), color = 'blue')
        
        # plot agent heading direction
        line_len = 100.
        plt.plot([agent_pos.item(1), agent_pos.item(1) + line_len*np.sin(agent_heading)], [agent_pos.item(0), agent_pos.item(0) + line_len*np.cos(agent_heading)], color = 'green')
    
    # remove any inactive agents from the active list
    for id in inactive_agents.keys():
        if id in active_agents:
            
            # Make all agents aware of destroyed target
            for agent in active_agents.values():
                del agent.belief.agent_estimates[id]

            del active_agents[id]
               
    
    # remove any inactive agents from the activev list
    for id in inactive_targets.keys():
        if id in active_targets:
            active_targets[id].des_kill_prob = 0 # set target's desired kill prob to zero (maybe not necessary since there is an "inactive_targets" dict)
            
            # Make all agents aware of destroyed target
            for agent in active_agents.values():
                del agent.belief.target_kill_prob[id]
                
            del active_targets[id]
    
    
    for target in active_targets.values():
        plt.scatter(target.pos.item(1), target.pos.item(0), color = 'red')
        
    plt.gca().set_aspect("equal")
    if not plt_init:
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt_init = True
    else:
        plt.xlim(xlim)
        plt.ylim(ylim)
    
    plt.gca().text(xlim[1]*0.87, ylim[1]*1.02, f"t = {sim_time:.2f}s")    
    
    plt.pause(round_ts)