from typing import List, Optional
import numpy as np  # type: ignore


class AzulAgent(object):
    def __init__(self, seed: Optional[int]):
        self.seed = seed

    def act(
        self,
        player: int,
        legal_actions: List[int],
        observation: np.ndarray
    ) -> int:
        pass
