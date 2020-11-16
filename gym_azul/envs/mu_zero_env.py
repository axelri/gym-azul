from typing import List

import gym


class MuzeroEnv(gym.Env):
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
