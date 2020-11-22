import gym  # type: ignore
import sys

from gym_azul.constants import MAX_PLAYERS
from gym_azul.agents import GreedyAgent, RandomAgent
from gym_azul.envs import AzulEnv
from gym_azul.model import action_from_action_num, Player
from gym_azul.util.format_utils import format_action_num

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

    total_reward = [0.0] * MAX_PLAYERS

    for i_episode in range(1):
        episode_done = False
        observation = env.reset()
        print("Initial board")
        env.render()

        step = 0
        for step in range(2000):
            print(f"Step: {step}")

            done = False
            legal_actions = env.legal_actions()
            current_player = Player(env.to_play())

            action = agent.act(current_player, legal_actions, observation)

            if action not in legal_actions:
                print(f"Illegal action: {action_from_action_num(action)}")
            else:
                observation, reward, done, info = env.step(action)
                total_reward[current_player] += reward
                print(f"Action: {format_action_num(action)}")
                print(f"Info  : {info}")
                env.render()
                print()

            print(f"Rewards after action: {total_reward}")
            print()

            if done:
                episode_done = True
                print(f"Episode finished after {step} steps")
                break

        if not episode_done:
            print(f"Episode did not finish after {step} steps")
    env.close()
