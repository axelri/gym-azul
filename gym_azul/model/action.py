from typing import NamedTuple

from gym import Space, spaces

from gym_azul.constants import TOTAL_SLOTS, TOTAL_COLORS, TOTAL_LINES, \
    TOTAL_COLUMNS, Slot, Color, Line, Column


class Action(NamedTuple):
    slot: Slot
    color: Color
    line: Line
    column: Column


def action_space() -> Space:
    """
    Scalar value:
    Note: Column is only used in advanced mode

    Slot x Color x Line x Column
    """
    return spaces.Discrete(
        TOTAL_SLOTS * TOTAL_COLORS * TOTAL_LINES * TOTAL_COLUMNS)


def action_num_from_action(game_action: Action) -> int:
    slot, color, line, column = game_action

    slot_value = slot * (TOTAL_LINES * TOTAL_COLORS * TOTAL_COLUMNS)
    color_value = color * (TOTAL_LINES * TOTAL_COLUMNS)
    line_value = line * TOTAL_COLUMNS
    column_value = column

    return slot_value + color_value + line_value + column_value


def action_from_action_num(action: int) -> Action:
    slot = action // (TOTAL_COLORS * TOTAL_LINES * TOTAL_COLUMNS)
    after_slot = action % (TOTAL_COLORS * TOTAL_LINES * TOTAL_COLUMNS)

    color = after_slot // (TOTAL_LINES * TOTAL_COLUMNS)
    after_color = after_slot % (TOTAL_LINES * TOTAL_COLUMNS)

    line = after_color // TOTAL_COLUMNS
    after_line = color % TOTAL_COLUMNS

    column = after_line

    return Action(Slot(slot), Color(color), Line(line), Column(column))
