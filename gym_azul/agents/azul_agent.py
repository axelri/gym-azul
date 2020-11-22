from typing import List, Optional
import numpy as np  # type: ignore

from gym_azul.constants import Player


class AzulAgent(object):
    def __init__(self, seed: Optional[int]):
        self.seed = seed

    def act(
        self,
        player: Player,
        legal_actions: List[int],
        observation: np.ndarray
    ) -> int:
        pass
