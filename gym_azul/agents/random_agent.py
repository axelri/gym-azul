from typing import List, Optional

import numpy as np  # type: ignore

from numpy.random import default_rng, Generator  # type: ignore

from gym_azul.constants import Player
from gym_azul.agents.azul_agent import AzulAgent


class RandomAgent(AzulAgent):
    """
    Totally random, stupid agent
    """
    random: Generator

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed is None:
            self.random = default_rng()
        else:
            self.random = default_rng(seed)

    def act(
        self,
        player: Player,
        legal_actions: List[int],
        observation: np.ndarray
    ) -> int:
        action: List[int] = self.random.choice(legal_actions, 1)
        return action[0]
