from typing import List

from gym_azul.game.game_utils import TOTAL_COLORS, \
    max_tiles_for_line, wall_column_color

import numpy as np

from gym_azul.spaces.from_azul_spaces import game_action_from_action


def format_color(color: int) -> str:
    colors = ["B", "Y", "R", "K", "C", "S", "-"]
    return colors[color]


def format_floor_line(floor_line: np.ndarray) -> str:
    line = ""
    for tile in range(7):
        color = floor_line[tile, 0]
        if color == 6:
            line += "-"
        elif color == 5:
            line += "S"
        else:
            line += format_color(color)
    return line


def format_wall(wall: np.ndarray) -> List[str]:
    lines = []
    for line in range(5):
        line_str = ""
        for column in range(5):
            color = wall_column_color(column, line)
            if wall[line, column]:
                line_str += format_color(color)
            else:
                line_str += "-"

        lines.append(line_str)

    return lines


def format_pattern_lines(pattern_lines: np.ndarray) -> List[str]:
    max_columns = 5
    lines = []

    for line in range(5):
        line_str = ""
        max_tiles = max_tiles_for_line(line)
        for column in range(max_tiles):
            value = pattern_lines[line, column]
            line_str += format_color(value)

        padding = max_columns - max_tiles
        line_str += " " * padding
        lines.append(line_str)

    return lines


def format_player(player_idx: int, player_channel: np.ndarray) -> str:
    pattern_lines = player_channel[0:5, 0:5]
    wall = player_channel[5:10, 0:5]
    floor_line = player_channel[0:7, 5:10]

    points = player_channel[7, 5:10]
    has_starting_marker = player_channel[8, 5:10]
    points_value = points[0]
    has_starting_marker_value = has_starting_marker[0]

    formatted_pattern_lines = format_pattern_lines(pattern_lines)
    formatted_wall = format_wall(wall)
    formatted_floor = format_floor_line(floor_line)

    board = []
    for pattern_line, wall_line in zip(formatted_pattern_lines, formatted_wall):
        board.append(pattern_line + " " + wall_line)
    board.append("Fl: " + formatted_floor)

    lines = [
        f"Player {player_idx + 1}",
        f"Points: {points_value:03}",
        f"Has sm: {bool(has_starting_marker_value)}",
        *board
    ]

    return "\n".join(lines)


def format_color_count(count: np.ndarray) -> str:
    line = ""
    for color in range(TOTAL_COLORS):
        line += f"{format_color(color)}:{count[color]:02} "

    return line


def format_slot_name(slot_idx):
    if slot_idx == 0:
        return "Cen"
    else:
        return f"Fa{slot_idx}"


def format_key_value(name: str, counts: str) -> str:
    return f"{name} => {counts}"


def format_slots(slots: np.ndarray) -> str:
    num_slots = slots.shape[0]

    lines = []
    for slot_idx in range(num_slots):
        slot = slots[slot_idx, :]
        slot_name = format_slot_name(slot_idx)
        slot_counts = format_color_count(slot)
        lines.append(format_key_value(slot_name, slot_counts))

    return "\n".join(lines)


def format_starting_marker(starting_marker, num_players):
    marker_value = f"Pl{starting_marker + 1}"
    if starting_marker == num_players:
        marker_value = "Cen"

    return format_key_value("Stm", marker_value)


def format_observation(observation: np.ndarray) -> str:
    """
    (players+1) x 10 x 10
    """

    channels, _, _ = observation.shape

    num_players = channels - 1
    players = []
    for player_idx in range(num_players):
        player_channel = observation[player_idx, :, :]
        players.append(format_player(player_idx, player_channel))
    formatted_players = "\n".join(players)

    slots = observation[-1, 0:10, 0:5]
    formatted_slots = format_slots(slots)

    bag = observation[-1, 0:5, 5:10]
    bag_values = bag[:, 0].T
    lid = observation[-1, 5:10, 5:10]
    lid_values = lid[:, 0].T
    formatted_bag = format_key_value("Bag", format_color_count(bag_values))
    formatted_lid = format_key_value("Lid", format_color_count(lid_values))
    formatted_tile_state = "\n".join([formatted_bag, formatted_lid])

    return "\n".join([formatted_players, formatted_slots, formatted_tile_state])


def format_action(action: int) -> str:
    slot, color, line = game_action_from_action(action)
    formatted_slot = format_slot_name(slot)
    formatted_color = format_color(color)

    return f"Slot: {formatted_slot}, Color: {formatted_color}, Line: {line + 1}"
