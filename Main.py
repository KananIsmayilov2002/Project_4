from Gridworld import GridWorld 
from Q_learning import QLearning, extract_policy, evaluate_agent 
'''
1. QLearning: the Q-learning algorithm class (the agent's learning brain),
2. extract_policy: a function that reads the learned Q-values and builds a policy (best actions),
3. evaluate_agent: a function to test how well the learned policy works.
'''
import random # Imports Python’s standard random module to generate random numbers (for randomizing goal location).

if __name__ == "__main__":
    GRID_SIZE = 8 # Sets the grid (environment) 
    goal = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)) # Randomly selects a goal position (x, y) somewhere inside the 40x40 grid.
    env = GridWorld(grid_size=GRID_SIZE, goal_state=goal, stochastic=True) # Creates the environment object env:Grid size is 40x40, Goal position is the random one picked above, stochastic=True means the environment is a bit random: actions may not always behave exactly as planned (good for training smarter agents).
    agent = QLearning(env, world_id=0) # Creates a QLearning agent:The agent knows the environment (env), world_id=0 could be used if multiple worlds are being trained separately (depends on QLearning class implementation).
    rewards = agent.train(episodes=1000) # Trains the agent using Q-learning for 1500 episodes:An episode = from start to reaching the goal (or failing), Rewards stores the rewards the agent got in each episode, which can be plotted later to see learning progress.
    policy = extract_policy(agent.Q, env) # After training, extracts the final policy from the learned Q-table:Policy = "for each cell, what's the best action?"
    evaluate_agent(env, policy, episodes=10) # Tests the agent: Runs 10 new episodes following the learned policy, Checks how well the agent reaches the goal using its knowledge.

