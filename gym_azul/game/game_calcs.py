from typing import Tuple, List, Optional

import numpy as np

from gym_azul.game.game_state import AzulPlayerState
from gym_azul.game.game_utils import wall_color_column, \
    max_tiles_for_line, PENALTIES, TOTAL_COLORS


def is_next_round(slots: np.ndarray) -> bool:
    """
    Check whether all slots are empty
    """
    return not np.any(slots)


def is_game_over(player_boards: List[AzulPlayerState]) -> bool:
    """
    Check if any player has a full wall row
    """
    for player_board in player_boards:
        wall = player_board.wall
        has_full_line = np.any(np.all(wall, axis=1))
        if has_full_line:
            return True

    return False


def calc_bonus_points(wall: np.ndarray) -> int:
    full_lines = np.sum(np.all(wall, axis=1))
    full_columns = np.sum(np.all(wall, axis=0))

    full_colors = 0
    for color in range(TOTAL_COLORS):
        all_tiles = np.array([wall[line, wall_color_column(color, line)]
                              for line in range(5)])
        if np.all(all_tiles):
            full_colors += 1

    return (full_lines * 2) + (full_columns * 7) + (full_colors * 10)


def calc_will_place_this_round(pattern_lines: np.ndarray) -> List[
    Tuple[int, int]]:
    placed = []

    for line in range(5):
        max_tiles = max_tiles_for_line(line)
        color = pattern_lines[line, 0]
        current_tiles = pattern_lines[line, 1]

        if max_tiles == current_tiles:
            placed.append((line, color))

    return placed


def calc_wall_score(wall: np.ndarray, pattern_lines: np.ndarray, color: int,
    line: int, deep_check: bool = True) -> Tuple[int, int]:
    """
    What score will we add if this is placed?
    """

    wall_column = wall_color_column(color, line)

    will_place_this_round = calc_will_place_this_round(pattern_lines)

    # tiles are placed in order from the top
    # these tiles are not affected by this tile, but will affect this tile
    will_place_before = filter(lambda x: x[0] < line, will_place_this_round)

    # these tiles are affected by this tile, but do not affect this tile
    will_place_after = filter(lambda x: x[0] > line, will_place_this_round)

    # calculate on future wall
    new_wall = np.copy(wall)
    for new_wall_line, new_wall_color in will_place_before:
        new_wall_column = wall_color_column(new_wall_color, new_wall_line)
        new_wall[new_wall_line, new_wall_column] = 1

    left = 0
    for column in range(wall_column - 1, -1, -1):
        if new_wall[line, column]:
            left += 1
        else:
            break

    right = 0
    for column in range(wall_column + 1, 5, 1):
        if new_wall[line, column]:
            right += 1
        else:
            break

    up = 0
    for row in range(line - 1, -1, -1):
        if new_wall[row, wall_column]:
            up += 1
        else:
            break

    down = 0
    for row in range(line + 1, 5, 1):
        if new_wall[row, wall_column]:
            down += 1
        else:
            break

    horizontal_points = left + right + 1
    vertical_points = up + down + 1

    round_score = 0
    round_score += horizontal_points
    round_score += vertical_points

    if deep_check:
        # check how this tile will affect tiles afterwards
        new_pattern_lines = np.copy(pattern_lines)
        new_pattern_lines[line, 0] = color
        new_pattern_lines[line, 1] = max_tiles_for_line(line)

        for row, c in will_place_after:
            old_round_score, _ = calc_wall_score(wall, pattern_lines, c,
                                                 row, deep_check=False)
            new_round_score, _ = calc_wall_score(wall, new_pattern_lines, c,
                                                 row, deep_check=False)

            round_score += (new_round_score - old_round_score)

    bonus_score = 0
    # full row
    if horizontal_points == 5:
        bonus_score += 2

    # full column
    if vertical_points == 5:
        bonus_score += 7

    # all colors filled
    all_tiles = np.array([wall[line, wall_color_column(color, line)]
                          for line in range(5)])
    if np.all(all_tiles):
        bonus_score += 10

    return round_score, bonus_score


def free_pattern_line_tiles(wall: np.ndarray,
    pattern_lines: np.ndarray, color: int, line: int) -> int:
    """
    How many tiles can we place on this pattern line?
    """
    column = wall_color_column(color, line)
    # No free tiles: already in wall
    if wall[line, column]:
        return 0

    line_color = pattern_lines[line, 0]
    current_tiles = pattern_lines[line, 1]

    # Pattern line is using another color
    if current_tiles > 0 and line_color != color:
        return 0

    max_tiles = max_tiles_for_line(line)
    free_tiles = max_tiles - current_tiles
    return free_tiles


PatternLineRes = Tuple[int, int, int]


def calc_place_pattern_line(tiles: int, wall: np.ndarray,
    pattern_lines: np.ndarray,
    color: int, line: int) -> PatternLineRes:
    """
    Place tiles on pattern line
    :return: (placed_tiles, did_place_wall, reward)
    """
    free_tiles = free_pattern_line_tiles(wall, pattern_lines, color, line)
    tiles_to_place = min(free_tiles, tiles)

    # Can not place any tiles
    if tiles_to_place == 0:
        return 0, 0, 0

    # Place the tiles on the pattern line
    round_reward = 0
    bonus_reward = 0
    if tiles_to_place == free_tiles:
        # We can place on wall
        round_score, bonus_score = calc_wall_score(wall, pattern_lines, color,
                                                   line)
        round_reward = round_score
        bonus_reward = bonus_score

    return tiles_to_place, round_reward, bonus_reward


FloorLineRes = Tuple[List[Tuple[int, int]], List[int], int]


def calc_place_floor_line(tiles: int, color: int, floor_line: np.ndarray,
    slot: int, starting_marker_in_center: bool) -> FloorLineRes:
    """
    Place tiles on the floor line
    :return: placed_tiles, discard, penalty
    """
    discard = [color] * tiles

    # Take start token (color=5)
    if slot == 0 and starting_marker_in_center:
        discard.append(5)

    # Place discarded tokens on floor line
    round_penalty = 0
    placed_tiles = []
    for index, floor_line_tile in enumerate(floor_line):
        if floor_line_tile == 6 and len(discard) > 0:
            tile = discard.pop()
            placed_tiles.append((index, tile))
            round_penalty += PENALTIES[index]

    return placed_tiles, discard, round_penalty


Placement = Tuple[int, List[Tuple[int, int]], List[int]]
Reward = Tuple[int, int, int]


def calc_move(player_board: AzulPlayerState, slots: np.ndarray,
    starting_marker_in_center: bool,
    slot: int, color: int, line: int) -> Optional[Tuple[Placement, Reward]]:
    """
    Checks if the move given move (draw_slot, pattern_line, amount_place)
    is valid. Returns None for invalid actions

    Returns:
        (placed_pattern_line, placed_floor_line, discarded, did_place_wall,
            final_reward) | None
    """

    tiles = slots[slot, color]
    # Player has to pick up some tiles
    if tiles == 0:
        return None

    place_pattern_line, round_reward, bonus_reward = calc_place_pattern_line(
        tiles,
        player_board.wall,
        player_board.pattern_lines,
        color,
        line)

    place_floor_line, discard, round_penalty = calc_place_floor_line(
        tiles - place_pattern_line,
        color,
        player_board.floor_line,
        slot,
        starting_marker_in_center)

    return (place_pattern_line, place_floor_line, discard), (
        round_reward, bonus_reward, round_penalty)
