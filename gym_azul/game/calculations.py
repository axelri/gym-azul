import copy
from typing import Tuple, List, Optional, Dict

from gym_azul.constants import max_tiles_for_line, \
    PENALTIES, Tile, Color, ColorTile, Line, Slot, FloorLineTile, \
    TOTAL_COLUMNS, TOTAL_LINES, TOTAL_COLORS
from gym_azul.game.move_model import Reward, Move, FloorLineMove, \
    PatternLineMove, PlacePattern, ScoreDelta, PlaceTile, PlaceFloorLine
from gym_azul.game.rules import wall_color_column, can_place_tile
from gym_azul.model import Action, AzulPlayerState, PatternLine, Column, Player


def is_next_round(slots: List[Dict[Color, int]]) -> bool:
    """
    Check whether all slots are empty
    """
    for slot in Slot:
        slot_amount = slots[slot]
        for amount in slot_amount.values():
            if amount != 0:
                return False

    return True


def is_game_over(player_boards: List[AzulPlayerState]) -> bool:
    """
    Check if any player has a full wall row
    """

    for player in Player:
        player_board = player_boards[player]
        wall = player_board.wall
        for line in Line:
            wall_line = wall[line]
            columns_filled = 0
            for column in Column:
                wall_tile = wall_line[column]
                if wall_tile != Tile.EMPTY:
                    columns_filled += 1
            if columns_filled == TOTAL_COLUMNS:
                return True
    return False


def calc_bonus_points(wall: List[List[ColorTile]]) -> int:
    line_count = [0] * TOTAL_LINES
    column_count = [0] * TOTAL_COLUMNS

    for line in Line:
        for column in Column:
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
            placed.append(PlaceTile(line, Color(line_color)))

    return placed


def calc_wall_score(
    wall: List[List[ColorTile]],
    pattern_lines: List[PatternLine],
    color: Color,
    line: Line,
    column: Column,
    deep_check: bool = True
) -> ScoreDelta:
    """
    What score will we add if this is placed?
    """

    will_place_this_round = calc_will_place_this_round(pattern_lines)

    # tiles are placed in order from the top
    # these tiles are not affected by this tile, but will affect this tile
    will_place_before = filter(lambda x: x.line < line, will_place_this_round)

    # these tiles are affected by this tile, but do not affect this tile
    will_place_after = filter(lambda x: x.line > line, will_place_this_round)

    # calculate on future wall
    new_wall = copy.deepcopy(wall)
    for place_line_before, place_color_before in will_place_before:
        new_wall_column = wall_color_column(
            place_color_before, place_line_before)
        new_wall[place_line_before][new_wall_column] = place_color_before

    left = 0
    for col in range(column - 1, -1, -1):
        if new_wall[line][col] != ColorTile.EMPTY:
            left += 1
        else:
            break

    right = 0
    for col in range(column + 1, TOTAL_COLUMNS, 1):
        if new_wall[line][col] != ColorTile.EMPTY:
            right += 1
        else:
            break

    up = 0
    for row in range(line - 1, -1, -1):
        if new_wall[row][column] != ColorTile.EMPTY:
            up += 1
        else:
            break

    down = 0
    for row in range(line + 1, TOTAL_COLUMNS, 1):
        if new_wall[row][column] != ColorTile.EMPTY:
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
        pattern_lines_after_placing = copy.deepcopy(pattern_lines)
        pattern_lines_after_placing[line].color = color
        pattern_lines_after_placing[line].amount = max_tiles_for_line(line)

        for place_line_after, place_color_after in will_place_after:
            place_column_after = wall_color_column(
                place_color_after, place_line_after)

            old_round_score, _ = calc_wall_score(
                wall, pattern_lines, place_color_after,
                place_line_after, place_column_after, deep_check=False)

            new_round_score, _ = calc_wall_score(
                wall, pattern_lines_after_placing, place_color_after,
                place_line_after, place_column_after, deep_check=False)

            round_score += (new_round_score - old_round_score)

    bonus_score = 0
    # full row
    if horizontal_points == 5:
        bonus_score += 2

    # full column
    if vertical_points == 5:
        bonus_score += 7

    # all colors filled
    all_tiles_filled = True
    for color_line in Line:
        color_col = wall_color_column(color, color_line)
        if wall[color_line][color_col] != color:
            all_tiles_filled = False
            break

    if all_tiles_filled:
        bonus_score += 10

    return ScoreDelta(round_score, bonus_score)


def free_pattern_line_tiles(
    wall: List[List[ColorTile]],
    pattern_lines: List[PatternLine],
    color: Color,
    line: Line,
    column: Column,
    advanced: bool = False
) -> int:
    """
    How many tiles can we place on this pattern line?
    """
    if not can_place_tile(wall, color, line, column, advanced):
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
    color: Color,
    line: Line,
    column: Column,
    tiles_amount: int,
    wall: List[List[ColorTile]],
    pattern_lines: List[PatternLine],
) -> PatternLineMove:
    """
    Place tiles on pattern line
    """

    free_tiles = free_pattern_line_tiles(
        wall, pattern_lines, color, line, column)
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
            wall, pattern_lines, color, line, column)
        round_reward = round_score_delta
        bonus_reward = bonus_score_delta

    return PatternLineMove(place_pattern, round_reward, bonus_reward)


def calc_place_floor_line(
    slot: Slot,
    color: Color,
    tiles_amount: int,
    floor_line: List[Tile],
    starting_marker_in_center: bool
) -> FloorLineMove:
    """
    Place tiles on the floor line
    """

    discard = [Tile(color)] * tiles_amount

    if slot == Slot.CENTER and starting_marker_in_center:
        discard.append(Tile.STARTING_TOKEN)

    # Place discarded tokens on floor line
    round_penalty = 0
    placed_tiles = []
    for floor_line_tile in FloorLineTile:
        tile = floor_line[floor_line_tile]
        if tile == Tile.EMPTY and len(discard) > 0:
            tile_to_place = discard.pop()
            placed_tiles.append(PlaceFloorLine(
                FloorLineTile(floor_line_tile), tile_to_place))
            round_penalty += PENALTIES[floor_line_tile]

    return FloorLineMove(placed_tiles, discard, round_penalty)


def calc_move(
    player_board: AzulPlayerState,
    slots: List[Dict[Color, int]],
    starting_marker_in_center: bool,
    action: Action,
    advanced: bool = False
) -> Optional[Tuple[Move, Reward]]:
    """
    Checks if the move given move is valid. Returns None for invalid actions
    """

    if advanced:
        raise NotImplemented("Not implemented yet")

    slot, color, line = action

    amount_tiles: int = slots[slot][color]
    # Player has to pick up some tiles
    if amount_tiles == 0:
        return None

    allowed_column = wall_color_column(color, line)

    place_pattern_line, round_reward, bonus_reward = calc_place_pattern_line(
        color,
        line,
        allowed_column,
        amount_tiles,
        player_board.wall,
        player_board.pattern_lines)

    place_floor_line, discard, round_penalty = calc_place_floor_line(
        slot,
        color,
        amount_tiles - place_pattern_line.amount,
        player_board.floor_line,
        starting_marker_in_center)

    move = Move(place_pattern_line, place_floor_line, discard)
    reward = Reward(round_reward, bonus_reward, round_penalty)

    return move, reward
