# gym-azul

gym-azul is [Open AI Gym](https://github.com/openai/gym) implementation of the board game [Azul](https://en.wikipedia.org/wiki/Azul_(board_game)). The observation and action formats are designed to
work well with [muzero-general](https://github.com/werner-duvaud/muzero-general).

## Table of Contents

1. [Game-presentation](#Game-presentation)
   1. [Introduction](#Introduction)
   2. [Gameplay](#Gameplay)
2. [Rules](#Rules)
   1. [Legal actions](#Permitted-moves)
3. [Environments](#Environments)
   1. [Observations](#Observations)
   2. [Actions](#Actions)
   3. [Reward Function](#Reward-Function)
4. [Installation](#Installation)
   1. [Requirements](#Requirements)
   2. [Using pip](#Using-pip)
5. [Usage](#Usage)
   1. [In code](#In-code)
   2. [Example run](#Example-run)
6. [Development](#Development)
   1. [Type checking](#Type-checking)
   2. [Packaging](#Packaging)
7. [Misc](#Misc)
   1. [Contribute](#Contribute)
   2. [Credits](#Credits)
   
# Game-presentation

## Introduction

Azul is 2-4 player game designed by Michael Kiesling in 2017. The theme of the game is to tile a
wall of Portugese [azulejos](https://en.wikipedia.org/wiki/Azulejo).

## Gameplay

Each player takes turns picking and placing tiles. On every turn, a player chooses one *slot* and
*color* to pick up, and one *line* to place the tiles on. The player must then place all tiles from
that slot of that color, and place them on their lines.

# Rules

## Legal actions

All combinations of slot, color and line are legal except for when the slot has zero tiles of the
chosen color.

# Environments

## Observations

The observation is a matrix on the form `(5, 10, 10)`.

## Actions

The actions are integers representing the slot, color and line. In total there are 10 * 5 * 5 = 250
possible actions on each player turn.

## Reward Function

The reward function outputs the delta in score at the end of the current round, if the action
would be performed by the player.

# Installation

## Requirements

- Python 3.8+
- OpenAI gym
- NumPy

##  Using pip

```
$ git clone git@github.com:riese/gym-azul.git
$ pip install -e .
```

# Usage

## In code

```python
import gym

env = gym.make('gym_azul:azul-v0', num_players=2, max_turns=100)

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, _ = env.step(action)
    env.render()
```

## Example run

```
python azul_test.py 2 500 greedy
```

# Development

## Type checking

This project uses [mypy](http://mypy-lang.org/) to validate type hints. 
To run it first install mypy:

```
$ pip install mypy
```

Then run it against the source code:
```
mypy gym_azul
```

## Packaging

```
$ python setup.py sdist bdist_wheel
$ pip install gym-azul --no-index --find-links file:////$(pwd)/dist
```

Or install in dev mode (might confuse intellij):
```
$ pip install -e .
```

# Misc

## Contribute

Feel free to contribute to this project. You can fork this repository and implement whatever you want. Alternatively, open a new issue in case you need help or want to have a feature added.

## Credits

Inspired by [gym-abalone](https://github.com/towzeur/gym-abalone)
