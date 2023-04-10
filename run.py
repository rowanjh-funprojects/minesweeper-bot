import os
import sys
import subprocess
import time

from playAgent import playAgent

if '-autolaunch' in sys.argv:
    # Launch minesweeper X
    wd = os.getcwd()
    subprocess.Popen(os.path.join(wd, 'Minesweeper X.exe'))
    time.sleep(1)

### Create the AI 
ai = playAgent()

### Find game window with computer vision
# Find the minesweeper game grid
ai.find_game_grid() # ai.game_grid
ai.initialize_game_grid() # ai.game

# Start playing
ai.play()
