# TODO: tune attrition (da)?

from agent import Agent
from target import Target
import numpy as np
import matplotlib.pyplot as plt

target_spawn_dim = 750. # length (m) of the square within which the targets spawn randomly
agent_spawn_dim = 750. # length (m) of the square within which the agents spawn randomly
agent_target_spawn_dist = 2500. # distance (m) between agent spawn square and target spawn square

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

num_targets = 10
des_kill_prob = 0.7 # For now, this is for all targets. Some simulations may need to create it separately

num_agents = 10
weapon_effectiveness = 0.9 # For now, this is for all targets. Some simulations may need to create is separately

# dictionaries containing agents/targets that have not been destroyed and update at every time step
active_agents = {}
active_targets = {}

# dictionaries containing agents/targets that have been destroyed
inactive_agents = {}
inactive_targets = {}

# init targets    
for i in range(num_targets):
    pos = np.array([np.random.uniform(high = target_spawn_dim), np.random.uniform(high = target_spawn_dim)])
    
    active_targets[i] = Target(i, pos, des_kill_prob)

# init agents and assign targets 
for i in range(num_agents):
    pos = np.array([np.random.uniform(high = agent_spawn_dim), np.random.uniform(high = agent_spawn_dim) + agent_target_spawn_dist])
    
    heading = np.random.uniform(low = -np.pi, high = np.pi)
    
    agent = Agent(i, pos, heading, agent_params, weapon_effectiveness, round_ts)
    
    agent.assign_target(active_targets[i]) # TODO: fix later, but for now, assign Agent 1 to Target 1
    
    active_agents[i] = agent
    
sim_time = 0
plt_init = False
while sim_time < end_time:
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
            del active_agents[id]
    
    # remove any inactive agents from the activev list
    for id in inactive_targets.keys():
        if id in active_targets:
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