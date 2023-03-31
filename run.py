from PIL import ImageGrab, Image
import numpy as np
from board_detector import find_game
import cv2

### Board detector
# Take a screenshot
screengrab = np.array(ImageGrab.grab()) # for the full screen

# Find a minesweeper game grid
gs = find_game(screengrab)

# Convert the grid to game object class (maybe port into AI brain)
game = 


# draw squares on top of img
img = screengrab.copy()
cv2.rectangle(img,(gs[0][0],gs[0][1]),(gs[0][0]+gs[0][2],gs[0][1]+gs[0][3]),(0,255,0),3)
Image.fromarray(img).show()




# Mouse control

# AI brain

# Messagebox/UI
