from typing import NamedTuple, Dict

from gym import Space, spaces

from gym_azul.constants import TOTAL_SLOTS, TOTAL_COLORS, TOTAL_LINES, \
    TOTAL_COLUMNS, Slot, Color, Line, Column


class Action(NamedTuple):
    slot: Slot
    color: Color
    line: Line


class AdvancedAction(NamedTuple):
    slot: Slot
    color: Color
    line: Line
    wallPlacement: Dict[Line, Column]


def action_space() -> Space:
    """
    Scalar value:

    Slot x Color x Line x Column
    """
    place_on_pattern_line = TOTAL_SLOTS * TOTAL_COLORS * TOTAL_LINES
    return spaces.Discrete(place_on_pattern_line)


def action_num_from_action(action: Action) -> int:
    slot, color, line = action

    slot_value = slot * (TOTAL_LINES * TOTAL_COLORS)
    color_value = color * TOTAL_LINES
    line_value = line

    return slot_value + color_value + line_value


def action_from_action_num(action: int) -> Action:
    slot = action // (TOTAL_COLORS * TOTAL_LINES)
    after_slot = action % (TOTAL_COLORS * TOTAL_LINES)

    color = after_slot // TOTAL_LINES
    after_color = after_slot % TOTAL_LINES

    line = after_color

    return Action(Slot(slot), Color(color), Line(line))


# TODO: implement advanced mode
def advanced_action_space() -> Space:
    """
    Scalar value:

    Slot x Color x Line x (Line x Column)
    """
    place_on_pattern_line = TOTAL_SLOTS * TOTAL_COLORS * TOTAL_LINES
    place_on_wall = TOTAL_LINES * TOTAL_COLUMNS

    return spaces.Discrete(place_on_pattern_line * place_on_wall)
