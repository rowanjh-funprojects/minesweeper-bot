import numpy as np
import cv2
from PIL import ImageGrab
from playAgent import playAgent

from board_detector import find_game, update_game
from minesweeper import MineSweeperGame

### Find New Game Window
# Take a screenshot
screengrab = np.array(ImageGrab.grab()) # for the full screen
# Find a minesweeper game grid
squares = find_game(screengrab, showVision=True)

# Create a representation of the minesweeper game window
xvals = np.unique([x for (x,y,w,h) in squares])
yvals = np.unique([y for (x,y,w,h) in squares])
r = len(xvals)
c = len(yvals)
w = squares[0][2]
h = squares[0][3]
xmin = min(xvals)
ymin = min(yvals)
game = MineSweeperGame(r,c, w, h, xmin, ymin)

### Update Game
# Check all unknown cells for changes
game = update_game(game)

### Get AI agent to make move
ai = playAgent()
ai.init_new_game(game.width, game.height)




# Mouse control

# AI brain

# Messagebox/UI
