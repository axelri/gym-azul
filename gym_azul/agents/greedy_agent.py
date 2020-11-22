from typing import List, Optional, NamedTuple

import numpy as np  # type: ignore
from numpy.random import default_rng, Generator  # type: ignore

from gym_azul.agents.azul_agent import AzulAgent
from gym_azul.constants import Slot, \
    Line
from gym_azul.game import free_pattern_line_tiles, wall_color_column
from gym_azul.model import AzulState, AzulPlayerState, Color
from gym_azul.model import state_from_observation, action_num_from_action, \
    Action


class Pile(NamedTuple):
    slot: Slot
    color: Color
    amount: int


class GreedyAgent(AzulAgent):
    """
    Select the largest pile and build from the bottom
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
        player: int,
        legal_actions: List[int],
        observation: np.ndarray
    ) -> int:

        state: AzulState = state_from_observation(observation)

        # Pick slot with largest pile
        piles: List[Pile] = []
        for slot in Slot:
            for color in Color:
                amount = state.slots[slot][color]
                if amount > 0:
                    piles.append(Pile(slot, color, amount))
        piles.sort(key=lambda x: x.amount, reverse=True)

        player_state: AzulPlayerState = state.players[player]
        wall = player_state.wall
        pattern_lines = player_state.pattern_lines

        # Put in largest pattern line
        for line in reversed(Line):
            for slot, color, amount in piles:
                allowed_column = wall_color_column(color, line)

                free_tiles = free_pattern_line_tiles(
                    wall, pattern_lines, color, line, allowed_column)

                # if we can place, then do it!
                if free_tiles > 0:
                    print("Free tiles")
                    print(free_tiles)
                    action = Action(slot, color, line)
                    return action_num_from_action(action)

        # If we can't place anything, take the smallest pile
        piles.reverse()
        slot, color, _amount = piles[0]
        line = Line.LINE_1
        action = Action(slot, color, line)
        action_num = action_num_from_action(action)

        if action_num not in legal_actions:
            raise Exception(f"Trying to play illegal action {action}")

        return action_num_from_action(action)
