from typing import Tuple

import numpy as np  # type: ignore
from gym import spaces, Space  # type: ignore

from gym_azul.game.game_utils import get_num_factories, TOTAL_COLORS, \
    MAX_POINTS, TILES_PER_COLOR, TOTAL_LINES, TOTAL_SLOTS


def wall_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    0 = no tile
    1 = tile is placed
    """
    low = np.full((5, 5), 0, dtype=np.int32)
    high = np.full((5, 5), 1, dtype=np.int32)
    return low, high


def pattern_lines_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    value = tile color/empty
    """
    low = np.full((5, 5), 0, dtype=np.int32)
    high = np.tril(np.full((5, 5), 6, dtype=np.int32))
    return low, high


def floor_line_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    7 x 5
    row repeated value = floor line column value
    """
    low = np.full((7, 5), 0, dtype=np.int32)
    high = np.full((7, 5), 6, dtype=np.int32)
    return low, high


def points_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5
    row repeated value = points value
    """
    low = np.full((1, 5), 0, dtype=np.int32)
    high = np.full((1, 5), MAX_POINTS, dtype=np.int32)
    return low, high


def starting_marker_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5
    row repeated value = 0: does not have, 1: does have
    """
    low = np.full((1, 5), 0, dtype=np.int32)
    high = np.full((1, 5), 0, dtype=np.int32)
    return low, high


def player_channel() -> Tuple[np.ndarray, np.ndarray]:
    """
    10 x 10

    Layout:
    Wal Flo
    Pat Poi
        Sta
    """
    wall_low, wall_high = wall_matrix()
    pattern_lines_low, pattern_lines_high = pattern_lines_matrix()
    floor_lines_low, floor_lines_high = floor_line_matrix()
    points_low, points_high = points_matrix()
    starting_marker_low, starting_marker_high = starting_marker_matrix()
    padding_low = np.full((1, 5), 0, dtype=np.int32)
    padding_high = np.full((1, 5), 0, dtype=np.int32)

    low = np.concatenate(
        (
            np.concatenate(
                (wall_low, pattern_lines_low),
                axis=0),
            np.concatenate(
                (floor_lines_low, points_low, starting_marker_low, padding_low),
                axis=0
            )
        ),
        axis=1
    )

    high = np.concatenate(
        (
            np.concatenate(
                (wall_high, pattern_lines_high),
                axis=0),
            np.concatenate(
                (floor_lines_high, points_high, starting_marker_high,
                 padding_high),
                axis=0
            )
        ),
        axis=1
    )

    return low, high


def factories_matrix(num_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    9 x 5
    row = factory
    column value = tile count for color
    """
    max_factories = 9
    num_factories = get_num_factories(num_players)
    low = np.full((max_factories, 5), 0, dtype=np.int32)
    high = np.full((num_factories, 5), 4, dtype=np.int32)
    if num_factories != max_factories:
        diff = max_factories - num_factories
        high_padding = np.full((diff, 5), 4, dtype=np.int32)
        high = np.concatenate(
            (high, high_padding),
            axis=0)

    return low, high


def center_matrix(num_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5
    column value = tile count for color
    """
    num_factories = get_num_factories(num_players)
    max_center_tiles_per_color = num_factories * 3
    low = np.full((1, 5), 0, dtype=np.int32)
    high = np.full((1, 5), max_center_tiles_per_color, dtype=np.int32)

    return low, high


def bag_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    repeated row value = tile count for color
    """
    low = np.full((5, 5), 0, dtype=np.int32)
    high = np.full((5, 5), TILES_PER_COLOR, dtype=np.int32)

    return low, high


def lid_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    repeated row value = tile count for color
    """
    low = np.full((5, 5), 0, dtype=np.int32)
    high = np.full((5, 5), TILES_PER_COLOR, dtype=np.int32)

    return low, high


def board_channel(num_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    10 x 10
    Layout:
    Cen Bag
    Fac Lid
    """

    center_low, center_high = center_matrix(num_players)
    factories_low, factories_high = factories_matrix(num_players)
    bag_low, bag_high = bag_matrix()
    lid_low, lid_high = lid_matrix()

    low = np.concatenate(
        (
            np.concatenate(
                (center_low, factories_low),
                axis=0),
            np.concatenate(
                (bag_low, lid_low),
                axis=0
            )
        ),
        axis=1
    )

    high = np.concatenate(
        (
            np.concatenate(
                (center_high, factories_high),
                axis=0),
            np.concatenate(
                (bag_high, lid_high),
                axis=0
            )
        ),
        axis=1
    )

    return low, high


def observation_space(num_players: int) -> Space:
    """
    Matrix:
    N+1 x 10 x 10
    Layout:
    P1, P2 ... Board
    """
    player_low, player_high = player_channel()
    players_low = [player_low] * num_players
    players_high = [player_high] * num_players
    board_low, board_high = board_channel(num_players)

    low = [
        *players_low,
        board_low
    ]

    high = [
        *players_high,
        board_high
    ]

    # Format: (channel, height, width)
    observation_low = np.moveaxis(np.dstack(low), 2, 0)
    observation_high = np.moveaxis(np.dstack(high), 2, 0)
    observation_shape = (num_players + 1, 10, 10)

    return spaces.Box(
        low=observation_low,
        high=observation_high,
        shape=observation_shape,
        dtype=np.int32
    )


def action_space() -> Space:
    """
    Scalar value:
    Slot x Color x Line
    """
    return spaces.Discrete(TOTAL_SLOTS * TOTAL_COLORS * TOTAL_LINES)
