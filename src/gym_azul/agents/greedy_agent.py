from typing import List, Tuple

import numpy as np  # type: ignore

from numpy.random import default_rng, Generator  # type: ignore

from gym_azul.agents.azul_agent import AzulAgent
from gym_azul.game.game_calcs import free_pattern_line_tiles
from gym_azul.game.game_state import AzulState, AzulPlayerState
from gym_azul.game.game_utils import TOTAL_SLOTS, TOTAL_COLORS, TOTAL_LINES
from gym_azul.spaces.from_azul_spaces import state_from_observation
from gym_azul.spaces.to_azul_spaces import action_from_game_action


class GreedyAgent(AzulAgent):
    """
    Select the largest pile and build from the bottom
    """
    random: Generator

    def __init__(self, seed=None):
        super(GreedyAgent, self).__init__()
        if seed is None:
            self.random = default_rng()
        else:
            self.random = default_rng(seed)

    def act(self, player: int, legal_actions: List[int],
        observation: np.ndarray) -> int:

        state: AzulState = state_from_observation(observation)

        # Pick slot with largest pile
        piles: List[Tuple[int, int, int]] = []
        for slot in range(TOTAL_SLOTS):
            for color in range(TOTAL_COLORS):
                tile_count = state.slots[slot, color]
                if tile_count > 0:
                    piles.append((slot, color, tile_count))
        piles.sort(key=lambda x: x[2], reverse=True)

        player_state: AzulPlayerState = state.players[player]
        wall = player_state.wall
        pattern_lines = player_state.pattern_lines

        # Put in largest pattern line
        for line in reversed(range(TOTAL_LINES)):
            for slot, color, tile_count in piles:
                free_tiles = free_pattern_line_tiles(wall, pattern_lines,
                                                     color, line)

                # if we can place, then do it!
                if free_tiles > 0:
                    game_action = (slot, color, line)
                    return action_from_game_action(game_action)

        # If we can't place anything, take the smallest pile
        piles.reverse()
        slot, color, tile_count = piles[0]
        game_action = (slot, color, 0)

        return action_from_game_action(game_action)
