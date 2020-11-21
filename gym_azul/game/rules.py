from typing import List, Dict, cast

from gym_azul.constants.constants import TOTAL_COLUMNS, Color, \
    Slot, Line, Column, ColorTile
from gym_azul.model import Action


def wall_color_column(color: Color, line: Line) -> Column:
    """
    Rotate all colors one step right per line
    """
    column_offset = line
    column_value = (color + column_offset) % 5

    return Column(column_value)


def wall_column_color(column: Column, line: Line) -> int:
    """
    Rotate all columns one step left per line
    """
    column_offset = line
    color_value = (column - column_offset) % 5
    color_value_pos = color_value + 5 if color_value < 0 else color_value

    return color_value_pos


def can_place_tile(
    wall: List[List[ColorTile]],
    action: Action,
) -> bool:
    slot, color, line, column = action
    for wall_line in Line:
        if wall[wall_line][column] == cast(ColorTile, color):
            # color is already in same column
            return False

    for wall_column in Column:
        if wall[line][wall_column] == cast(ColorTile, color):
            # color is already in same line
            return False

    return True


def generate_legal_actions(
    slots: List[Dict[Color, int]],
    wall: List[List[ColorTile]],
    advanced=False
) -> List[Action]:
    """
    Illegal actions:
    1. Pick a color from a slot with no tiles of that color
    2. Place a tile on the wall where this tile has already been placed
    """
    legal_actions = []
    for slot_idx, slot in enumerate(slots):
        for color, amount in slot.items():
            for line in Line:
                if advanced:
                    # We can place our tile on the wall column we choose
                    for column in range(TOTAL_COLUMNS):
                        action = Action(Slot(slot_idx),
                                        Color(color),
                                        line,
                                        Column(column))
                        if amount > 0 and can_place_tile(wall, action):
                            legal_actions.append(action)
                else:
                    if amount > 0:
                        # We can only place the tile on a fixed column
                        allowed_column = wall_color_column(color, line)
                        legal_actions.append(
                            Action(Slot(slot_idx),
                                   Color(color),
                                   line,
                                   Column(allowed_column)))

    return legal_actions
