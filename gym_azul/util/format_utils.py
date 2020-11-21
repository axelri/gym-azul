from typing import List, Dict, cast

import numpy as np  # type: ignore

from gym_azul.constants import TOTAL_COLORS, max_tiles_for_line, Tile, \
    TOTAL_LINES, \
    TOTAL_COLUMNS, Color, ColorTile
from gym_azul.game import Action, Line
from gym_azul.model import action_from_action_num, AzulState, AzulPlayerState, \
    state_from_observation, PatternLine

COLOR_ABBR: Dict[Tile, str] = {
    Tile.BLUE: "B",
    Tile.YELLOW: "Y",
    Tile.RED: "R",
    Tile.BLACK: "K",
    Tile.CYAN: "C",
    Tile.EMPTY: "-",
    Tile.STARTING_TOKEN: "S",
}


def format_tile(tile: Tile) -> str:
    return COLOR_ABBR[tile]


def format_floor_line(floor_line: List[Tile]) -> str:
    line = ""
    for tile in floor_line:
        line += format_tile(tile)
    return line


def format_wall(wall: List[List[ColorTile]]) -> List[str]:
    lines = []
    for line in range(TOTAL_LINES):
        line_str = ""
        for column in range(TOTAL_COLUMNS):
            color = wall[line][column]
            line_str += format_tile(cast(Tile, color))

        lines.append(line_str)

    return lines


def format_pattern_lines(pattern_lines: List[PatternLine]) -> List[str]:
    lines = []

    for line_idx, line in enumerate(pattern_lines):
        line_str = ""
        max_tiles = max_tiles_for_line(Line(line_idx))

        color = line.color
        amount = line.amount
        line_str += format_tile(cast(Tile, color)) * amount

        empty = max_tiles - amount
        line_str += format_tile(Tile.EMPTY) * empty

        padding = TOTAL_COLUMNS - max_tiles
        line_str += " " * padding
        lines.append(line_str)

    return lines


def format_player(
    player_idx: int,
    player_state: AzulPlayerState,
    has_starting_marker: bool,
    has_next_turn: bool
) -> str:
    formatted_pattern_lines = format_pattern_lines(player_state.pattern_lines)
    formatted_wall = format_wall(player_state.wall)
    formatted_floor = format_floor_line(player_state.floor_line)

    board = []
    for pattern_line, wall_line in zip(formatted_pattern_lines, formatted_wall):
        board.append(pattern_line + " " + wall_line)
    board.append("Fl: " + formatted_floor)

    lines = [
        f"Player {player_idx + 1}",
        f"Points: {player_state.points:03}",
        f"Has sm: {has_starting_marker}",
        f"Has nt: {has_next_turn}",
        *board
    ]
    return "\n".join(lines)


def format_color_count(count: Dict[Color, int]) -> str:
    line = ""
    for color in range(TOTAL_COLORS):
        color_value = Color(color)
        amount = count[color_value]
        format = format_tile(cast(Tile, color))
        line += f"{format}:{amount:02} "

    return line


def format_slot_name(slot_idx):
    if slot_idx == 0:
        return "Cen"
    else:
        return f"Fa{slot_idx}"


def format_key_value(name: str, counts: str) -> str:
    return f"{name} => {counts}"


def format_slots(slots: List[Dict[Color, int]]) -> str:
    lines = []
    for slot_idx, slot in enumerate(slots):
        slot_name = format_slot_name(slot_idx)
        slot_counts = format_color_count(slot)
        lines.append(format_key_value(slot_name, slot_counts))

    return "\n".join(lines)


def format_starting_marker(starting_marker, num_players):
    marker_value = f"Pl{starting_marker + 1}"
    if starting_marker == num_players:
        marker_value = "Cen"

    return format_key_value("Stm", marker_value)


def format_state(state: AzulState):
    starting_marker = state.starting_marker
    player = state.player
    players = []

    for player_idx, player_state in enumerate(state.players):
        players.append(format_player(
            player_idx,
            player_state,
            starting_marker == player_idx,
            player == player_idx
        ))
    formatted_players = "\n".join(players)

    formatted_slots = format_slots(state.slots)
    formatted_bag = format_key_value("Bag", format_color_count(state.bag))
    formatted_lid = format_key_value("Lid", format_color_count(state.lid))
    formatted_tile_state = "\n".join([formatted_bag, formatted_lid])

    return "\n".join([formatted_players, formatted_slots, formatted_tile_state])


def format_observation(observation: np.ndarray) -> str:
    state = state_from_observation(observation)
    return format_state(state)


def format_action_num(action_num) -> str:
    return format_action(action_from_action_num(action_num))


def format_action(action: Action) -> str:
    slot, color, line, column = action
    formatted_slot = format_slot_name(slot)
    formatted_color = format_tile(color)

    return f"Slot: {formatted_slot}, Color: {formatted_color}, Line: {line + 1}"
