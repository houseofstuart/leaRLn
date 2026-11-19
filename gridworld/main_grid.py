import gymnasium as gym
from tqdm import tqdm
from grid import GridWorldEnv
from agent import GridWorldAgent
from plot import plot_grid

def main():

    lr = 0.2
    n_episodes = 50000
    start_epsilon = 0.3
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = GridWorldEnv()

    agent = GridWorldAgent(
        env=env,
        learning_rate=lr,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon
    )
    train(env, agent, n_episodes)
    plot_grid(q_values=agent.q_values, grid_size=env.size)


def train(env: GridWorldEnv, agent: GridWorldAgent, n_episodes: int):

    for episode in tqdm(range(n_episodes)):

        # Start a new agent in the grid
        obs, info = env.reset()
        done = False

        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)
            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)
            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

    


if __name__ == "__main__":
    main()