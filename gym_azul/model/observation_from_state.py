from typing import List, Dict

import numpy as np  # type: ignore

from gym_azul.constants import TOTAL_LINES, TOTAL_COLUMNS, ColorTile, Tile, \
    Color, TOTAL_SLOTS, TOTAL_COLORS, Player, PLAYER_INACTIVE_OBS
from gym_azul.model.state import AzulState, AzulPlayerState, PatternLine


def obs_pattern_lines(state_pattern_lines: List[PatternLine]) -> np.ndarray:
    """
    5 x 5
    """
    pattern_lines = np.full((TOTAL_LINES, TOTAL_COLUMNS), ColorTile.EMPTY,
                            dtype=np.int32)

    for line in range(5):
        color = state_pattern_lines[line].color
        amount = state_pattern_lines[line].amount
        for column in range(amount):
            pattern_lines[line, column] = color

    return pattern_lines


def obs_wall(state_wall: List[List[ColorTile]]) -> np.ndarray:
    wall = np.full((TOTAL_LINES, TOTAL_COLUMNS), ColorTile.EMPTY,
                   dtype=np.int32)
    for line_idx, line in enumerate(state_wall):
        for col_idx, color in enumerate(line):
            wall[line_idx, col_idx] = color
    return wall


def obs_floor_line(state_floor_line: List[Tile]) -> np.ndarray:
    """
    7 x 5
    """
    split = np.split(np.array(state_floor_line), len(state_floor_line))
    return np.tile(split, 5)


def obs_points(state_points: int) -> np.ndarray:
    """
    1 x 5
    """
    return np.full((1, 5), state_points, dtype=np.int32)


def obs_starting_marker(state_has_starting_marker: bool) -> np.ndarray:
    """
    1 x 5
    """
    return np.full((1, 5), int(state_has_starting_marker), dtype=np.int32)


def obs_player_turn(state_has_next_turn: bool) -> np.ndarray:
    """
    1 x 5
    """
    return np.full((1, 5), int(state_has_next_turn), dtype=np.int32)


def player_channel(
    player_state: AzulPlayerState,
    has_starting_marker: bool,
    has_next_turn: bool,
    player_in_game: bool
) -> np.ndarray:
    """
    10 x 10
    """

    if not player_in_game:
        return np.full((10, 10), PLAYER_INACTIVE_OBS, dtype=np.int32)

    pattern_lines = obs_pattern_lines(player_state.pattern_lines)
    wall = obs_wall(player_state.wall)
    floor_line = obs_floor_line(player_state.floor_line)
    points = obs_points(player_state.points)
    starting_marker = obs_starting_marker(has_starting_marker)
    next_turn = obs_player_turn(has_next_turn)

    return np.concatenate(
        (
            np.concatenate(
                (pattern_lines, wall), axis=0),
            np.concatenate(
                (floor_line, points, starting_marker, next_turn), axis=0),
        ),
        axis=1
    )


def obs_slots(state_slots: List[Dict[Color, int]]) -> np.ndarray:
    """
    10 x 5
    """

    slots = np.full((TOTAL_SLOTS, TOTAL_COLORS), 0, dtype=np.int32)

    for slot_idx, slot in enumerate(state_slots):
        for color, count in slot.items():
            slots[slot_idx, color] = count

    return slots


def obs_bag(state_bag: Dict[Color, int]) -> np.ndarray:
    """
    5 x 5
    """
    bag_values = [0] * TOTAL_COLORS
    for color, count in state_bag.items():
        bag_values[color] = count

    split = np.split(np.array(bag_values), len(bag_values))
    return np.tile(split, 5)


def obs_lid(state_lid: Dict[Color, int]) -> np.ndarray:
    """
    5 x 5
    """
    lid_values = [0] * TOTAL_COLORS
    for color, count in state_lid.items():
        lid_values[color] = count

    split = np.split(np.array(lid_values), len(lid_values))
    return np.tile(split, 5)


def board_channel(
    state_slots: List[Dict[Color, int]],
    state_bag: Dict[Color, int],
    state_lid: Dict[Color, int]
) -> np.ndarray:
    """
    10 x 10
    """

    slots = obs_slots(state_slots)
    bag = obs_bag(state_bag)
    lid = obs_lid(state_lid)

    return np.concatenate(
        (
            slots,
            np.concatenate((bag, lid), axis=0),
        ),
        axis=1
    )


def observation_from_state(state: AzulState) -> np.ndarray:
    """
    5 x 10 x 10
    """
    starting_marker = state.starting_marker

    player_obs: List[np.ndarray] = []
    for player in Player:
        player_state = state.players[player]
        player_obs.append(
            player_channel(
                player_state,
                starting_marker == player,
                state.current_player == player,
                player < state.num_players
            ))
    board_obs: np.ndarray = board_channel(state.slots, state.bag, state.lid)

    observation = np.dstack((*player_obs, board_obs))
    moved_axis = np.moveaxis(observation, 2, 0)

    # format requires channel to be first dimension
    # dstack makes this the last dimension
    return moved_axis
