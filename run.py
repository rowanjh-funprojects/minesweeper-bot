import numpy as np
from PIL import ImageGrab
from playAgent import playAgent

from boardDetector import find_game, update_game
from minesweeper import MineSweeperGame

### Find New Game Window
# Take a screenshot
screengrab = np.array(ImageGrab.grab()) # for the full screen
# Find a minesweeper game grid
grid = find_game(screengrab, showVision=False)

# Create a representation of the minesweeper game window
game = MineSweeperGame.from_grid(grid)

### Update Game

### Get AI agent to make move
ai = playAgent(game)

# Make initial move
move = ai.plan_move()
ai.execute_move(move)


# Check all unknown cells for changes
# game = update_game(game)






# Mouse control

# AI brain

# Messagebox/UI
