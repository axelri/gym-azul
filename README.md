# gym-azul

gym-azul is [Open AI Gym](https://github.com/openai/gym) implementation of the board game [Azul](https://en.wikipedia.org/wiki/Azul_(board_game)).

## Table of Contents

1. [Game-presentation](#Game-presentation)
   1. [Introduction](#Introduction)
   2. [Object of the game](#Object-of-the-game)
   3. [Gameplay](#Gameplay)
2. [Rules](#Rules)
   1. [Starting-positions ](#Starting-positions)
   2. [Permitted-moves](#Permitted-moves)
   3. [Pushing-the-opponent](#Pushing-the-opponent)
      1. [Sumito](#Sumito)
      2. [Pac](#Pac)
   4. [Forbidden-moves](#Forbidden-moves)
3. [Environments](#Environments)
   1. [Features of Environment](#Features-of-Environment)
   2. [Episode termination](#Episode-termination)
   2. [Observations](#Observations)
   3. [Actions](#Actions)
   4. [Reward Function](#Reward-Function)
4. [Installation](#Installation)
5. [Usage](#Usage)
6. [Misc](#Misc)
   1. [Abalone variations](#Abalone-variations)
      1. [Game start variant](#Game-start-variant)
      2. [More player](#More-player)
      3. [Limited Time](#Limited-Time)
      4. [Blitz Competition](#Blitz-Competition)
   2. [Citation](#Citation)
   3. [Contribute](#Contribute)
   
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

The observation is a matrix on the form `(N+1, 10, 10)`, where N is the number of players.

## Actions

The actions are integers representing the slot, color and line. In total there are 10 * 5 * 5 = 250
possible actions on each player turn.

## Reward Function

The reward function outputs the delta in score at the end of the current round, if the action
would be performed by the player.

# Installation

## Requirements:

- Python 3.8+
- OpenAI gym
- NumPy

##  Using pip

```
$ git clone git@github.com:riees/gym-azul.git
$ cd gym-azul
$ pip install -e .
```

# Usage

```python
import gym

env = gym.make('gym_azul:azul-v0', num_players=2, max_turns=100)

done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, _ = env.step(action)
    env.render()
```

## Contribute

Feel free to contribute to this project. You can fork this repository and implement whatever you want. Alternatively, open a new issue in case you need help or want to have a feature added.

## Credits

Inspired by [gym-abalone](https://github.com/towzeur/gym-abalone)
