from enum import IntEnum
from typing import Dict, List


class NumPlayers(IntEnum):
    PLAYERS_2 = 2
    # TODO: support more players
    # PLAYERS_3 = 3
    # PLAYERS_4 = 4


class Player(IntEnum):
    PLAYER_1 = 0
    PLAYER_2 = 1
    # TODO: support more players
    # PLAYER_3 = 2
    # PLAYER_4 = 3


class LineAmount(IntEnum):
    AMOUNT_0 = 0
    AMOUNT_1 = 1
    AMOUNT_2 = 2
    AMOUNT_3 = 3
    AMOUNT_4 = 4
    AMOUNT_5 = 5


class StartingMarker(IntEnum):
    PLAYER_1 = 0
    PLAYER_2 = 1
    PLAYER_3 = 2
    PLAYER_4 = 3
    CENTER = 4


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

# TODO: Support more players
MAX_PLAYERS: NumPlayers = NumPlayers.PLAYERS_2

# Maximum points achieved by placing full columns in every turn
MAX_POINTS: int = 236

# Number of factories for different player counts
NUM_FACTORIES: Dict[int, int] = {
    2: 5,
    3: 7,
    4: 9
}

PENALTIES: List[int] = [1, 1, 2, 2, 2, 3, 3]

PLAYER_INACTIVE_OBS: int = -1


def get_num_factories(num_players: int) -> int:
    num_factories = NUM_FACTORIES[num_players]
    if num_factories is None:
        raise Exception("Wrong number of players", num_players)
    return num_factories


def max_tiles_for_line(line: Line) -> int:
    return line + 1
