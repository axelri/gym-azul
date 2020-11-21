from dataclasses import dataclass, field
from typing import List, Dict

from gym_azul.constants import MAX_PLAYERS, TOTAL_SLOTS, TILES_PER_COLOR, \
    Tile, ColorTile, TOTAL_LINES, TOTAL_COLUMNS, FLOOR_LINE_SIZE, Color


@dataclass
class PatternLine:
    color: ColorTile
    amount: int


def new_pattern_lines() -> List[PatternLine]:
    return [PatternLine(ColorTile.EMPTY, 0) for _line in
            range(TOTAL_LINES)]


def new_wall() -> List[List[ColorTile]]:
    wall = []
    for line in range(TOTAL_LINES):
        wall_line = []
        for column in range(TOTAL_COLUMNS):
            wall_line.append(ColorTile.EMPTY)
        wall.append(wall_line)

    return wall


def new_floor_line() -> List[Tile]:
    return [Tile.EMPTY for _column in range(FLOOR_LINE_SIZE)]


def new_bag() -> Dict[Color, int]:
    return {color: TILES_PER_COLOR for color in Color}


def new_lid() -> Dict[Color, int]:
    return {color: 0 for color in Color}


def new_slots() -> List[Dict[Color, int]]:
    return [{color: 0 for color in Color} for _slot in range(TOTAL_SLOTS)]


@dataclass
class AzulPlayerState:
    """
    Azul player model
    """
    points: int = 0
    pattern_lines: List[PatternLine] = field(default_factory=new_pattern_lines)
    wall: List[List[ColorTile]] = field(default_factory=new_wall)
    floor_line: List[Tile] = field(default_factory=new_floor_line)


@dataclass
class AzulState:
    """
    Azul game model
    """
    players: List[AzulPlayerState]
    slots: List[Dict[Color, int]] = field(default_factory=new_slots)
    bag: Dict[Color, int] = field(default_factory=new_bag)
    lid: Dict[Color, int] = field(default_factory=new_lid)
    starting_marker: int = MAX_PLAYERS
    player: int = 0
    turn: int = 0
    round: int = 0


def new_state(num_players: int, start_player: int):
    players = [AzulPlayerState() for _player in range(num_players)]
    return AzulState(players, player=start_player)
