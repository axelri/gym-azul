from typing import Tuple

import numpy as np  # type: ignore
from gym import spaces, Space  # type: ignore

from gym_azul.constants import get_num_factories, MAX_POINTS, TILES_PER_COLOR, \
    TOTAL_COLORS, TILES_PER_FACTORY, Tile, FLOOR_LINE_SIZE, TOTAL_LINES, \
    TOTAL_COLUMNS, ColorTile


def wall_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    value = placed tile

    | Row/Col | 0                     | ... | 4                     |
    |---------|-----------------------|-----|-----------------------|
    | 0       | Wall Line 1, Column 1 | ... | Wall Line 5, Column 1 |
    | ...     | ...                   | ... | ...                   |
    | 4       | Wall Line 5, Column 1 | ... | Wall Line 5, Column 5 |
    """

    low = np.full((TOTAL_LINES, TOTAL_COLUMNS), 0,
                  dtype=np.int32)
    high = np.full((TOTAL_LINES, TOTAL_COLUMNS), ColorTile.EMPTY,
                   dtype=np.int32)
    return low, high


def pattern_lines_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    value = placed tile

    | Row/Col | 0       | 1       | 2       | 3       | 4       |
    |---------|---------|---------|---------|---------|---------|
    | 0       | PL[0,0] | X       | X       | X       | X       |
    | 1       | PL[1,0] | PL[1,1] | X       | X       | X       |
    | 2       | PL[2,0] | PL[2,1] | PL[2,2] | X       | X       |
    | 3       | PL[3,0] | PL[3,1] | PL[3,2] | PL[3,3] | X       |
    | 4       | PL[4,0] | PL[4,1] | PL[4,2] | PL[4,3] | PL[4,4] |
    """

    low = np.full((5, 5), 0, dtype=np.int32)
    high = np.tril(np.full((5, 5), 6, dtype=np.int32))
    return low, high


def floor_line_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    7 x 5
    value = placed tile

    | Row | Col 0-4           |
    |-----|-------------------|
    | 0   | Floor line tile 1 |
    | ... | ...               |
    | 6   | Floor line tile 7 |
    """

    low = np.full((FLOOR_LINE_SIZE, 5), 0, dtype=np.int32)
    high = np.full((FLOOR_LINE_SIZE, 5), Tile.STARTING_TOKEN, dtype=np.int32)
    return low, high


def points_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5

    | Row | Col 0-4        |
    |-----|----------------|
    | 0   | Current points |
    """

    low = np.full((1, 5), 0, dtype=np.int32)
    high = np.full((1, 5), MAX_POINTS, dtype=np.int32)
    return low, high


def starting_marker_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5

    | Row | Col 0-4                                                    |
    |-----|------------------------------------------------------------|
    | 0   | 0 = Does not have starting marker, 1 = Has starting marker |
    """

    low = np.full((1, 5), int(False), dtype=np.int32)
    high = np.full((1, 5), int(True), dtype=np.int32)
    return low, high


def player_turn_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5

    | Row | Col 0-4                              |
    |-----|--------------------------------------|
    | 0   | 0 = Not player turn, 1 = Player turn |
    """

    low = np.full((1, 5), int(False), dtype=np.int32)
    high = np.full((1, 5), int(True), dtype=np.int32)
    return low, high


def player_channel() -> Tuple[np.ndarray, np.ndarray]:
    """
    10 x 10

    Layout:
    | Row | Col 0-4       | Col 5-9         |
    |-----|---------------|-----------------|
    | 0   | Wall          | Floor Line      |
    | 1   | Wall          | Floor Line      |
    | 2   | Wall          | Floor Line      |
    | 3   | Wall          | Floor Line      |
    | 4   | Wall          | Floor Line      |
    | 5   | Pattern Lines | Floor Line      |
    | 6   | Pattern Lines | Floor Line      |
    | 7   | Pattern Lines | Points          |
    | 8   | Pattern Lines | Starting Marker |
    | 9   | Pattern Lines | Player Turn     |
    """

    wall_low, wall_high = wall_matrix()
    pattern_lines_low, pattern_lines_high = pattern_lines_matrix()
    floor_lines_low, floor_lines_high = floor_line_matrix()
    points_low, points_high = points_matrix()
    starting_marker_low, starting_marker_high = starting_marker_matrix()
    player_turn_low, player_turn_high = player_turn_matrix()

    low = np.concatenate(
        (
            np.concatenate(
                (wall_low, pattern_lines_low),
                axis=0),
            np.concatenate(
                (floor_lines_low, points_low, starting_marker_low,
                 player_turn_low),
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
                 player_turn_high),
                axis=0
            )
        ),
        axis=1
    )

    return low, high


def factories_matrix(num_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    9 x 5

    | Row/Col | 0                     | ... | 4                     |
    |---------|-----------------------|-----|-----------------------|
    | 0       | Factory 1, blue count | ... | Factory 1, cyan count |
    | ...     | ...                   | ... | ...                   |
    | 9       | Factory 9, blue count | ... | Factory 9, cyan count |
    """

    max_factories = 9
    num_factories = get_num_factories(num_players)
    low = np.full((max_factories, 5), 0, dtype=np.int32)
    high = np.full((num_factories, 5), TILES_PER_FACTORY, dtype=np.int32)
    if num_factories != max_factories:
        diff = max_factories - num_factories
        high_padding = np.full((diff, 5), TILES_PER_FACTORY, dtype=np.int32)
        high = np.concatenate(
            (high, high_padding),
            axis=0)

    return low, high


def center_matrix(num_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1 x 5

    | Row/Col | 0                  | ... | 4                  |
    |---------|--------------------|-----|--------------------|
    | 0       | Center, blue count | ... | Center, cyan count |
    """

    num_factories = get_num_factories(num_players)
    max_center_tiles_per_color = num_factories * 3
    low = np.full((1, 5), 0, dtype=np.int32)
    high = np.full((1, 5), max_center_tiles_per_color, dtype=np.int32)

    return low, high


def bag_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5

    | Row | Col 0-4          |
    |-----|------------------|
    | 0   | Bag blue count   |
    | 1   | Bag yellow count |
    | 2   | Bag red count    |
    | 3   | Bag black count  |
    | 4   | Bag cyan count   |
    """
    low = np.full((TOTAL_COLORS, 5), 0, dtype=np.int32)
    high = np.full((TOTAL_COLORS, 5), TILES_PER_COLOR, dtype=np.int32)

    return low, high


def lid_matrix() -> Tuple[np.ndarray, np.ndarray]:
    """
    5 x 5
    
    | Row | Col 0-4          |
    |-----|------------------|
    | 0   | Lid blue count   |
    | 1   | Lid yellow count |
    | 2   | Lid red count    |
    | 3   | Lid black count  |
    | 4   | Lid cyan count   |
    """
    low = np.full((5, 5), 0, dtype=np.int32)
    high = np.full((5, 5), TILES_PER_COLOR, dtype=np.int32)

    return low, high


def board_channel(num_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    10 x 10
    Layout:


    | Row | Col 0-4 | Col 5-9 |
    |-----|---------|---------|
    | 0   | Center  | Bag     |
    | 1   | Factory | Bag     |
    | 2   | Factory | Bag     |
    | 3   | Factory | Bag     |
    | 4   | Factory | Bag     |
    | 5   | Factory | Lid     |
    | 6   | Factory | Lid     |
    | 7   | Factory | Lid     |
    | 8   | Factory | Lid     |
    | 9   | Factory | Lid     |
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

    | Channel | Row 0-9, Col 0-9 |
    |---------|------------------|
    | 1       | Player 1         |
    | ...     |                  |
    | N       | Player N + 1     |
    | N+1     | Shared Board     |
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
