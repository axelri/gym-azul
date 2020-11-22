from enum import IntEnum
from typing import Dict, List


class Tile(IntEnum):
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    CYAN = 4
    EMPTY = 5
    STARTING_TOKEN = 6


class ColorTile(IntEnum):
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    CYAN = 4
    EMPTY = 5


class Color(IntEnum):
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    CYAN = 4


class Line(IntEnum):
    LINE_1 = 0
    LINE_2 = 1
    LINE_3 = 2
    LINE_4 = 3
    LINE_5 = 4


class Column(IntEnum):
    COLUMN_1 = 0
    COLUMN_2 = 1
    COLUMN_3 = 2
    COLUMN_4 = 3
    COLUMN_5 = 4


class FloorLineTile(IntEnum):
    TILE_1 = 0
    TILE_2 = 1
    TILE_3 = 2
    TILE_4 = 3
    TILE_5 = 4
    TILE_6 = 5
    TILE_7 = 6


class Slot(IntEnum):
    CENTER = 0
    FACTORY_1 = 1
    FACTORY_2 = 2
    FACTORY_3 = 3
    FACTORY_4 = 4
    FACTORY_5 = 5
    FACTORY_6 = 6
    FACTORY_7 = 7
    FACTORY_8 = 8
    FACTORY_9 = 9


TOTAL_LINES: int = len([line for line in Line])

TOTAL_COLUMNS: int = len([column for column in Column])

TOTAL_SLOTS: int = len([slot for slot in Slot])

TOTAL_COLORS: int = len([color for color in Color])

FLOOR_LINE_SIZE: int = len([tile for tile in FloorLineTile])

TILES_PER_COLOR: int = 20

TILES_PER_FACTORY: int = 4

MIN_PLAYERS: int = 2

MAX_PLAYERS: int = 4

STARTING_MARKER_CENTER: int = MAX_PLAYERS

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


def max_tiles_for_line(line: Line) -> int:
    return line + 1
