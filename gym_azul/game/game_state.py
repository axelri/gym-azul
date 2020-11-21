from typing import List, Optional

import numpy as np  # type: ignore

from gym_azul.game.game_utils import TOTAL_COLORS, \
    TILES_PER_COLOR, TOTAL_SLOTS


class AzulPlayerState(object):
    """
    Azul player model
    """
    points: int
    pattern_lines: np.ndarray
    wall: np.ndarray
    floor_line: np.ndarray

    def __init__(self, points: int = 0,
        pattern_lines: Optional[np.ndarray] = None,
        wall: Optional[np.ndarray] = None,
        floor_line: Optional[np.ndarray] = None) -> None:
        self.points = points
        if pattern_lines is None:
            self.pattern_lines = np.tile(np.array([6, 0], dtype=np.int32),
                                         (5, 1))
        else:
            self.pattern_lines = pattern_lines

        if wall is None:
            self.wall = np.full((5, 5), 0, dtype=np.int32)
        else:
            self.wall = wall

        if floor_line is None:
            self.floor_line = np.full(7, 6, dtype=np.int32)
        else:
            self.floor_line = floor_line


class AzulState(object):
    """
    Azul game model
    """
    players: List[AzulPlayerState]
    slots: np.ndarray
    bag: np.ndarray
    lid: np.ndarray
    starting_marker: int

    def __init__(self, num_players: int,
        players: Optional[List[AzulPlayerState]] = None,
        slots: Optional[np.ndarray] = None,
        bag: Optional[np.ndarray] = None,
        lid: Optional[np.ndarray] = None,
        starting_marker: Optional[int] = None) -> None:
        if players is None:
            self.players = [AzulPlayerState() for _player in range(num_players)]
        else:
            self.players = players

        if slots is None:
            self.slots = self.slots = np.full((TOTAL_SLOTS, TOTAL_COLORS), 0,
                                              dtype=np.int32)
        else:
            self.slots = slots

        if bag is None:
            self.bag = np.full(TOTAL_COLORS, TILES_PER_COLOR, dtype=np.int32)
        else:
            self.bag = bag

        if lid is None:
            self.lid = np.full(TOTAL_COLORS, 0, dtype=np.int32)
        else:
            self.lid = lid

        if starting_marker is None:
            self.starting_marker = num_players
        else:
            self.starting_marker = starting_marker
