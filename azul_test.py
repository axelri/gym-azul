import gym
import sys

from gym_azul.agents.greedy_agent import GreedyAgent
from gym_azul.agents.random_agent import RandomAgent
from gym_azul.envs import AzulEnv
from gym_azul.spaces.from_azul_spaces import game_action_from_action
from gym_azul.util.format_utils import format_action

if __name__ == "__main__":
    num_players = int(sys.argv[1])
    max_turns = int(sys.argv[2])
    arg_agent = sys.argv[3]

    agents = {
        "random": RandomAgent(),
        "greedy": GreedyAgent()
    }
    agent = agents[arg_agent]

    env: AzulEnv = gym.make('gym_azul:azul-v0',
                            num_players=num_players, max_turns=max_turns)

    for i_episode in range(1):
        episode_done = False
        observation = env.reset()
        print("Initial board")
        print(env.render())

        step = 0
        for step in range(2000):
            print(f"Step: {step}")

            done = False
            legal_actions = env.legal_actions()
            action = agent.act(env.to_play(), legal_actions, observation)

            if action not in legal_actions:
                print(f"Illegal action: {game_action_from_action(action)}")
            else:
                observation, reward, done, info = env.step(action)
                print(f"Action: {format_action(action)}")
                print(f"Info  : {info}")
                print(env.render())

            if done:
                episode_done = True
                print(f"Episode finished after {step} steps")
                break

        if not episode_done:
            print(f"Episode did not finish after {step} steps")
    env.close()
