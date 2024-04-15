# TODO: check/debug (not bugs found yet) greedy function and traditional cost function
# TODO: tune attrition (da)?

from agent import Agent
from target import Target
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import LinearSegmentedColormap
import plotter

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

# returns a list of agents ordered from highest to lowest weapon effectiveness
def order_agents(agents):
    agent_list = [agent for agent in agents.values()]
    
    # return list sorted in descending order by weapon effectiveness
    agent_list.sort(reverse=True, key=lambda agent:agent.weapon_effectiveness) 
    return agent_list

def target_assignment(A, agents, cost_function):
    agent_list = order_agents(agents)
    for i in range(len(agent_list)):
        agent_list[i].select_target('greedy', cost_function)
        communicate(A, agents)
        # TODO: call commmunicate here

cost_function = 'traditional' # which cost function to use ("traditional", "sufficiency threshold", "tiered", or "completion")
animate = False
plot_kill_prob = False
np.random.seed(9)

target_spawn_dim = 750. # length (m) of the square within which the targets spawn randomly
agent_spawn_dim = 750. # length (m) of the square within which the agents spawn randomly
agent_target_spawn_dist = 2500. # distance (m) between agent spawn square and target spawn square

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

num_targets = 6
des_kill_prob = 0.01 # For now, this is for all targets. Some simulations may need to create it separately

num_agents = 8
weapon_effectiveness_dict = {}
for i in range(num_agents):
    weapon_effectiveness_dict[i] = 0.7 # For now, this is for all agents. Some simulations may need to create is separately


# init targets   
active_targets = {}
for i in range(num_targets):
    pos = np.array([np.random.uniform(high = target_spawn_dim), np.random.uniform(high = target_spawn_dim)])
    if i < 2:
        active_targets[i] = Target(i, pos, 0.9, value = 0.9) 
    elif i < 4:
        active_targets[i] = Target(i, pos, 0.8, value = 0.8)
    else:
        active_targets[i] = Target(i, pos, 0.7, value = 0.7) 


# init agents and assign targets 
active_agents = {}
for i in range(num_agents):
    pos = np.array([np.random.uniform(high = agent_spawn_dim), np.random.uniform(high = agent_spawn_dim) + agent_target_spawn_dist])
    
    heading = np.random.uniform(low = -np.pi, high = np.pi)
    
    agent = Agent(i, pos, heading, agent_params, weapon_effectiveness_dict, active_targets, round_ts) # assumes all agents have same effectiveness, but a different value can be put here if needed
    
    active_agents[i] = agent

A = update_adj_matrix(num_agents, active_agents, comms_range)
target_assignment(A, active_agents, cost_function)

inactive_targets = {} 
inactive_agents = {}

sim_time = 0
sim_time_hist = [sim_time]
target_kill_probabilities_hist = {}
agent_assignment_hist = {}
for target_id in active_targets:
    target_kill_probabilities_hist[target_id] = [0]
for agent_id in active_agents:
    agent_assignment_hist[agent_id] = [active_agents[agent_id].target.id]

if animate:
    anim_plt_init = False
    anim_fig, anim_ax = plt.subplots()
if plot_kill_prob:
    kill_fig, kill_ax = plt.subplots()
while sim_time < end_time and len(active_targets) != 0:
    A = update_adj_matrix(num_agents, active_agents, comms_range)
    target_assignment(A, active_agents, cost_function)
    sim_time += round_ts
    sim_time_hist.append(sim_time)
    if animate:
        anim_ax.cla()
    if plot_kill_prob:
        kill_ax.cla()
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
        
        if animate:
            # plot agent position
            anim_ax.scatter(agent_pos.item(1), agent_pos.item(0), color = 'blue')
            
            # plot agent heading direction
            line_len = 50.
            anim_ax.plot([agent_pos.item(1), agent_pos.item(1) + line_len*np.sin(agent_heading)], [agent_pos.item(0), agent_pos.item(0) + line_len*np.cos(agent_heading)], color = 'green')
            
            # plot connection between agent and target
            anim_ax.plot([agent_pos.item(1), agent.target.pos.item(1)], [agent_pos.item(0), agent.target.pos.item(0)], color = 'c')
    # remove any inactive agents from the active list
    for id in inactive_agents.keys():
        if id in active_agents:
            
            # Make all agents aware of destroyed target
            for agent in active_agents.values():
                del agent.belief.agent_estimates[id]

            del active_agents[id]
               
    
    # remove any inactive agents from the active list
    for id in inactive_targets.keys():
        if id in active_targets:
            active_targets[id].des_kill_prob = 0 # set target's desired kill prob to zero (maybe not necessary since there is an "inactive_targets" dict)
            
            # Make all agents aware of destroyed target
            for agent in active_agents.values():
                del agent.belief.target_kill_prob[id]
                
            del active_targets[id]
    
    
    
    if len(active_agents) > 0:
        for agent_id in agent_assignment_hist:
            if agent_id in active_agents:
                agent_assignment_hist[agent_id].append(active_agents[agent_id].target.id)
        
        target_kill_probabilities = active_agents[list(active_agents.keys())[0]].belief.target_kill_prob
        for target_id in target_kill_probabilities_hist:
            if target_id in target_kill_probabilities:
                target_kill_probabilities_hist[target_id].append(target_kill_probabilities[target_id])
        for target in active_targets.values():
            if animate:
                if target.id < 2:
                    anim_ax.scatter(target.pos.item(1), target.pos.item(0), color = 'red')
                elif target.id < 4:
                    anim_ax.scatter(target.pos.item(1), target.pos.item(0), color = 'orange')
                else:
                    anim_ax.scatter(target.pos.item(1), target.pos.item(0), color = 'yellow')
        
        if plot_kill_prob:
            for target_id, kill_prob_hist in target_kill_probabilities_hist.items():    
                if target_id < 2:
                    color = 'red'
                elif target_id < 4:
                    color = 'orange'
                else:
                    color = 'yellow'
                
                len_hist = len(kill_prob_hist)
                kill_ax.plot(sim_time_hist[:len_hist], kill_prob_hist, color = color)
                
        
        if animate:   
            anim_ax.set_aspect("equal")
            if not anim_plt_init:
                xlim = anim_ax.get_xlim()
                ylim = anim_ax.get_ylim()
                anim_plt_init = True
            else:
                anim_ax.set_xlim(xlim)
                anim_ax.set_ylim(ylim)
            
            anim_ax.text(xlim[1]*0.87, ylim[1]*1.02, f"t = {sim_time:.2f}s")    
        
        if animate or plot_kill_prob:
            plt.pause(round_ts)
    else:
        break

plotter.plot_achieved_pk(target_kill_probabilities_hist, num_targets)

plotter.plot_agent_assignments(agent_assignment_hist, num_agents, num_targets)