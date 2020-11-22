from dataclasses import dataclass, field
from typing import List, Dict

from gym_azul.constants import TILES_PER_COLOR, \
    Tile, ColorTile, Color, Player, \
    StartingMarker, Slot, FloorLineTile, Line, Column, LineAmount, NumPlayers


@dataclass
class PatternLine:
    color: ColorTile
    amount: LineAmount


def new_pattern_lines() -> List[PatternLine]:
    return [PatternLine(ColorTile.EMPTY, LineAmount.AMOUNT_0) for _line in Line]


def new_wall() -> List[List[ColorTile]]:
    wall = []
    for _line in Line:
        wall_line = []
        for _column in Column:
            wall_line.append(ColorTile.EMPTY)
        wall.append(wall_line)

    return wall


def new_floor_line() -> List[Tile]:
    return [Tile.EMPTY for _tile in FloorLineTile]


def new_bag() -> Dict[Color, int]:
    return {color: TILES_PER_COLOR for color in Color}


def new_lid() -> Dict[Color, int]:
    return {color: 0 for color in Color}


def new_slots() -> List[Dict[Color, int]]:
    return [{color: 0 for color in Color} for _slot in Slot]


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
    starting_marker: StartingMarker = StartingMarker.CENTER
    current_player: Player = Player.PLAYER_1
    num_players: NumPlayers = NumPlayers.PLAYERS_2
    turn: int = 0
    round: int = 0


def new_state(num_players: NumPlayers, start_player: Player):
    players = [AzulPlayerState() for _player in Player]
    return AzulState(players=players,
                     num_players=num_players,
                     current_player=start_player)
