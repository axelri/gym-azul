from typing import List, NamedTuple, Dict

from gym_azul.constants import Line, Color, Column, Tile, FloorLineTile, \
    LineAmount


class PlacePattern(NamedTuple):
    line: Line
    color: Color
    amount: LineAmount


class PlaceLine(NamedTuple):
    column: Column
    color: Color


class PlaceFloorLine(NamedTuple):
    column: FloorLineTile
    tile: Tile


class PlaceTile(NamedTuple):
    line: Line
    color: Color


class ScoreDelta(NamedTuple):
    round_score_delta: int
    bonus_score_delta: int


class Move(NamedTuple):
    pattern_line: PlacePattern
    floor_line: List[PlaceFloorLine]
    discard: List[Tile]


class Reward(NamedTuple):
    before: int
    after: int


class FloorLineMove(NamedTuple):
    floor_line: List[PlaceFloorLine]
    discard: List[Tile]
    round_penalty: Reward


class PatternLineMove(NamedTuple):
    pattern_line: PlacePattern
    round_reward: Reward
    bonus_reward: Reward


class ActionResult(NamedTuple):
    reward: float
    info: Dict[str, int]
