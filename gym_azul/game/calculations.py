import copy
from typing import Tuple, List, Optional, Dict, cast

import numpy as np  # type: ignore

from gym_azul.constants import max_tiles_for_line, \
    PENALTIES, Tile, Color, ColorTile, Line, Slot, FloorLineTile, TOTAL_COLUMNS, \
    TOTAL_LINES, TOTAL_COLORS
from gym_azul.game.move_model import Reward, Move, FloorLineMove, \
    PatternLineMove, PlacePattern, ScoreDelta, PlaceTile, PlaceFloorLine
from gym_azul.game.rules import wall_color_column
from gym_azul.model import Action, AzulPlayerState, PatternLine


def is_next_round(slots: List[Dict[Color, int]]) -> bool:
    """
    Check whether all slots are empty
    """
    for slot in slots:
        for amount in slot.values():
            if amount != 0:
                return False

    return True


def is_game_over(player_boards: List[AzulPlayerState]) -> bool:
    """
    Check if any player has a full wall row
    """
    for player_board in player_boards:
        wall = player_board.wall
        for line in wall:
            columns_filled = 0
            for tile in line:
                if tile != Tile.EMPTY:
                    columns_filled += 1
            if columns_filled == TOTAL_COLUMNS:
                return True
    return False


def calc_bonus_points(wall: List[List[ColorTile]]) -> int:
    line_count = [0] * TOTAL_LINES
    column_count = [0] * TOTAL_COLUMNS

    for line in range(TOTAL_LINES):
        for column in range(TOTAL_COLUMNS):
            tile = wall[line][column]
            if tile != ColorTile.EMPTY:
                line_count[line] += 1
                column_count[column] += 1

    color_count = [0] * TOTAL_COLORS
    for color in Color:
        for line in Line:
            color_column = wall_color_column(color, line)
            tile = wall[line][color_column]
            if tile != ColorTile.EMPTY:
                color_count[color] += 1

    full_lines = 0
    for count in line_count:
        if count == TOTAL_COLUMNS:
            full_lines += 1

    full_columns = 0
    for count in column_count:
        if count == TOTAL_LINES:
            full_columns += 1

    full_colors = 0
    for count in color_count:
        if count == TOTAL_LINES:
            full_colors += 1

    return (full_lines * 2) + (full_columns * 7) + (full_colors * 10)


def calc_will_place_this_round(
    pattern_lines: List[PatternLine]
) -> List[PlaceTile]:
    placed = []

    for line in Line:
        max_tiles = max_tiles_for_line(line)
        line_color = pattern_lines[line].color
        line_amount = pattern_lines[line].amount

        if max_tiles == line_amount:
            placed.append(PlaceTile(line, cast(Color, line_color)))

    return placed


def calc_wall_score(
    wall: List[List[ColorTile]],
    pattern_lines: List[PatternLine],
    color: Color,
    line: Line,
    deep_check: bool = True
) -> ScoreDelta:
    """
    What score will we add if this is placed?
    """

    wall_column = wall_color_column(color, line)

    will_place_this_round = calc_will_place_this_round(pattern_lines)

    # tiles are placed in order from the top
    # these tiles are not affected by this tile, but will affect this tile
    will_place_before = filter(lambda x: x.line < line, will_place_this_round)

    # these tiles are affected by this tile, but do not affect this tile
    will_place_after = filter(lambda x: x.line > line, will_place_this_round)

    # calculate on future wall
    new_wall = copy.deepcopy(wall)
    for new_wall_line, new_wall_color in will_place_before:
        new_wall_column = wall_color_column(new_wall_color, new_wall_line)
        new_wall[new_wall_line][new_wall_column] = new_wall_color

    left = 0
    for column in range(wall_column - 1, -1, -1):
        if new_wall[line][column] != ColorTile.EMPTY:
            left += 1
        else:
            break

    right = 0
    for column in range(wall_column + 1, TOTAL_COLUMNS, 1):
        if new_wall[line][column] != ColorTile.EMPTY:
            right += 1
        else:
            break

    up = 0
    for row in range(line - 1, -1, -1):
        if new_wall[row][wall_column] != ColorTile.EMPTY:
            up += 1
        else:
            break

    down = 0
    for row in range(line + 1, TOTAL_COLUMNS, 1):
        if new_wall[row][wall_column] != ColorTile.EMPTY:
            down += 1
        else:
            break

    horizontal_points = left + right
    if horizontal_points > 0:
        # add placed tile
        horizontal_points += 1

    vertical_points = up + down
    if vertical_points > 0:
        # add placed tile
        vertical_points += 1

    single_points = 0
    if horizontal_points == 0 and vertical_points == 0:
        single_points = 1

    round_score = horizontal_points + vertical_points + single_points

    if deep_check:
        # check how this tile will affect tiles afterwards
        new_pattern_lines = copy.deepcopy(pattern_lines)
        new_pattern_lines[line].color = color
        new_pattern_lines[line].amount = max_tiles_for_line(line)

        for place_line, place_color in will_place_after:
            old_round_score, _ = calc_wall_score(
                wall, pattern_lines, place_color, place_line,
                deep_check=False)
            new_round_score, _ = calc_wall_score(
                wall, new_pattern_lines, place_color, place_line,
                deep_check=False)

            round_score += (new_round_score - old_round_score)

    bonus_score = 0
    # full row
    if horizontal_points == 5:
        bonus_score += 2

    # full column
    if vertical_points == 5:
        bonus_score += 7

    # all colors filled
    all_tiles = np.array([wall[line][wall_color_column(color, line)]
                          for line in Line])
    if np.all(all_tiles):
        bonus_score += 10

    return ScoreDelta(round_score, bonus_score)


def free_pattern_line_tiles(
    wall: List[List[ColorTile]],
    pattern_lines: List[PatternLine],
    color: Color,
    line: Line
) -> int:
    """
    How many tiles can we place on this pattern line?
    """
    column = wall_color_column(color, line)
    # No free tiles: color already in wall
    if wall[line][column] == cast(ColorTile, color):
        return 0

    line_color = pattern_lines[line].color
    line_amount = pattern_lines[line].amount

    # Pattern line is using another color
    if line_amount > 0 and line_color != color:
        return 0

    max_tiles = max_tiles_for_line(line)
    free_tiles = max_tiles - line_amount
    return free_tiles


def calc_place_pattern_line(
    action: Action,
    tiles_amount: int,
    wall: List[List[ColorTile]],
    pattern_lines: List[PatternLine]
) -> PatternLineMove:
    """
    Place tiles on pattern line
    """
    color, line = (action.color, action.line)

    free_tiles = free_pattern_line_tiles(wall, pattern_lines, color, line)
    amount = min(free_tiles, tiles_amount)

    place_pattern = PlacePattern(line, color, amount)

    # Can not place any tiles
    if amount == 0:
        return PatternLineMove(place_pattern, 0, 0)

    # Place the tiles on the pattern line
    round_reward = 0
    bonus_reward = 0
    if amount == free_tiles:
        # We can place on wall
        round_score_delta, bonus_score_delta = calc_wall_score(
            wall, pattern_lines, color, line)
        round_reward = round_score_delta
        bonus_reward = bonus_score_delta

    return PatternLineMove(place_pattern, round_reward, bonus_reward)


def calc_place_floor_line(
    action: Action,
    tiles_amount: int,
    floor_line: List[Tile],
    starting_marker_in_center: bool
) -> FloorLineMove:
    """
    Place tiles on the floor line
    """

    slot, color = (action.slot, action.color)
    discard = [cast(Tile, color)] * tiles_amount

    if slot == Slot.CENTER and starting_marker_in_center:
        discard.append(Tile.STARTING_TOKEN.value)

    # Place discarded tokens on floor line
    round_penalty = 0
    placed_tiles = []
    for index, floor_line_tile in enumerate(floor_line):
        if floor_line_tile == Tile.EMPTY and len(discard) > 0:
            tile = discard.pop()
            placed_tiles.append(PlaceFloorLine(FloorLineTile(index), tile))
            round_penalty += PENALTIES[index]

    return FloorLineMove(placed_tiles, discard, round_penalty)


def calc_move(
    player_board: AzulPlayerState,
    slots: List[Dict[Color, int]],
    starting_marker_in_center: bool,
    action: Action
) -> Optional[Tuple[Move, Reward]]:
    """
    Checks if the move given move (draw_slot, pattern_line, amount_place)
    is valid. Returns None for invalid actions
    """

    slot, color, line, column = action

    amount_tiles: int = slots[slot][color]
    # Player has to pick up some tiles
    if amount_tiles == 0:
        return None

    place_pattern_line, round_reward, bonus_reward = calc_place_pattern_line(
        action,
        amount_tiles,
        player_board.wall,
        player_board.pattern_lines)

    place_floor_line, discard, round_penalty = calc_place_floor_line(
        action,
        amount_tiles - place_pattern_line.amount,
        player_board.floor_line,
        starting_marker_in_center)

    move = Move(place_pattern_line, place_floor_line, discard)
    reward = Reward(round_reward, bonus_reward, round_penalty)

    return move, reward
