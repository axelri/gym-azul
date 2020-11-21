from typing import Tuple

import numpy as np  # type: ignore

from gym_azul.game.game_state import AzulState, AzulPlayerState
from gym_azul.game.game_utils import TOTAL_COLORS, TOTAL_LINES


def obs_pattern_lines(board_pattern_lines: np.ndarray) -> np.ndarray:
    """
    5 x 5
    """
    pattern_lines = np.full((5, 5), 6, dtype=np.int32)
    for line in range(5):
        color = board_pattern_lines[line, 0]
        amount = board_pattern_lines[line, 1]
        for column in range(amount):
            pattern_lines[line, column] = color

    return pattern_lines


def obs_wall(board_wall: np.ndarray) -> np.ndarray:
    return np.copy(board_wall)


def obs_floor_line(floor_line: np.ndarray) -> np.ndarray:
    """
    7 x 5
    """
    values = np.split(floor_line, len(floor_line))
    return np.tile(values, 5)


def obs_points(points: int) -> np.ndarray:
    """
    1 x 5
    """
    return np.full((1, 5), points, dtype=np.int32)


def obs_starting_marker(has_starting_marker: bool) -> np.ndarray:
    """
    1 x 5
    """
    return np.full((1, 5), int(has_starting_marker), dtype=np.int32)


def player_channel(player: AzulPlayerState,
    has_starting_marker: bool) -> np.ndarray:
    """
    10 x 10
    """

    pattern_lines = obs_pattern_lines(player.pattern_lines)
    wall = obs_wall(player.wall)
    floor_line = obs_floor_line(player.floor_line)
    points = obs_points(player.points)
    starting_marker = obs_starting_marker(has_starting_marker)
    player_padding = np.full((1, 5), 0, dtype=np.int32)

    return np.concatenate(
        (
            np.concatenate(
                (pattern_lines, wall), axis=0),
            np.concatenate(
                (floor_line, points, starting_marker, player_padding), axis=0),
        ),
        axis=1
    )


def obs_slots(board_slots: np.ndarray) -> np.ndarray:
    """
    10 x 5
    """
    slots = np.copy(board_slots)
    slots_height, _ = slots.shape
    diff = 10 - slots_height
    if diff > 0:
        padding = np.full((diff, 5), 0, dtype=np.int32)
        slots = np.concatenate((slots, padding), axis=0)

    return slots


def obs_bag(board_bag: np.ndarray) -> np.ndarray:
    """
    5 x 5
    """
    bag_values = np.split(board_bag, len(board_bag))
    bag = np.tile(bag_values, 5)
    return bag


def obs_lid(board_lid: np.ndarray) -> np.ndarray:
    """
    5 x 5
    """
    lid_values = np.split(board_lid, len(board_lid))
    lid = np.tile(lid_values, 5)
    return lid


def board_channel(board_slots: np.ndarray, board_bag: np.ndarray,
    board_lid: np.ndarray) -> np.ndarray:
    """
    10 x 10
    """

    slots = obs_slots(board_slots)
    bag = obs_bag(board_bag)
    lid = obs_lid(board_lid)

    return np.concatenate(
        (
            slots,
            np.concatenate((bag, lid), axis=0),
        ),
        axis=1
    )


def observation_from_state(state: AzulState) -> np.ndarray:
    """
    N+1 x 10 x 10
    """
    starting_marker = state.starting_marker

    players = []
    for player_idx, player in enumerate(state.players):
        players.append(player_channel(player, starting_marker == player_idx))

    state = board_channel(state.slots, state.bag, state.lid)

    observation = np.dstack((*players, state))

    # format requires channel to be first dimension
    # dstack makes this the last dimension
    return np.moveaxis(observation, 2, 0)


def action_from_game_action(game_action: Tuple[int, int, int]) -> int:
    slot, color, line = game_action

    slot_value = slot * (TOTAL_LINES * TOTAL_COLORS)
    color_value = color * TOTAL_LINES
    line_value = line

    return slot_value + color_value + line_value
