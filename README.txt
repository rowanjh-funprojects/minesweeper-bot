# Minesweeper bot
A gameplay agent that plays minesweeper. It uses computer vision to parse the
screen in order to find the game window and each game tile. It translate the 
images to an internal representation of the minesweeper game. It uses logical 
deductions to make inferences about which cells are mines or safes. Then when
the agent decides which move to make, it uses the mouse to click on the next
tile. 

Execute program with 'python run.py'.
The minesweeper window must be visible on the screen to run.

The agent is programmed to work with "Minesweeper X", a clone of the original game, 
see http://www.curtisbright.com/msx/

# TODO:
- Improve logic algorithm (a bit unreliable right now)
- Use a CNN instead of simple colour recognition to classify cells (for fun)
- Allow the game to re-discover the window if the game freezes.
