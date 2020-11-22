from typing import List

import gym  # type: ignore


class MuzeroEnv(gym.Env):
    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def to_play(self) -> int:
        """
        Returns the current player.
        """
        pass

    def legal_action(self) -> List[int]:
        """
        Returns list of legal actions
        """
        pass

    def expert_action(self) -> int:
        """
        Returns benchmark action in multiplayer games.
        """
        pass
