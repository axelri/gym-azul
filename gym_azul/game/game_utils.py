from typing import Dict, List, Tuple
import numpy as np  # type: ignore

TOTAL_LINES: int = 5

TOTAL_SLOTS: int = 10

TOTAL_COLORS: int = 5
# Tiles per color in the game
TILES_PER_COLOR: int = 20

# Maximum points achieved by placing full columns in every turn
MAX_POINTS: int = 141

# Place last column
MAX_POINTS_ROUND: int = 35

# 2 x 5  + 7 x 5 + 10 x 5
MAX_BONUS_POINTS: int = 95

# Number of factories for different player counts
NUM_FACTORIES: Dict[int, int] = {
    2: 5,
    3: 7,
    4: 9
}

PENALTIES: List[int] = [1, 1, 2, 2, 2, 3, 3]


def get_num_factories(num_players: int) -> int:
    num_factories = NUM_FACTORIES[num_players]
    if num_factories is None:
        raise Exception("Wrong number of players", num_players)
    return num_factories


def wall_color_column(color: int, line: int) -> int:
    """
    Rotate all colors one step right per line
    """
    column_offset = line
    column_value = (color + column_offset) % 5

    return column_value


def wall_column_color(column: int, line: int) -> int:
    """
    Rotate all columns one step left per line
    """
    column_offset = line
    color_value = (column - column_offset) % 5
    color_value_pos = color_value + 5 if color_value < 0 else color_value

    return color_value_pos


def max_tiles_for_line(line: int) -> int:
    return line + 1


def generate_legal_actions(slots: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Only illegal action is to take tiles from a slot with no tiles
    """
    legal_actions = []
    for slot in range(TOTAL_SLOTS):
        for color in range(TOTAL_COLORS):
            if slots[slot, color] > 0:
                for line in range(TOTAL_LINES):
                    legal_actions.append((slot, color, line))

    return legal_actions
