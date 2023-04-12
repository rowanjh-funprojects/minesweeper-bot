# Minesweeper bot
An agent that plays minesweeper. It uses computer vision to parse the
screen and find the game window. Images are translated to an internal representation
of the minesweeper game. Then logical deductions are used to make inferences about 
which cells are mines or safes. The agent controls the mouse to click on tiles in the
game window. 

To run the bot:
1. Clone repo
2. Ensure all dependencies are installed
3. Ensure minesweeper window is visible on screen.
4. Execute program with the command `python run.py`.

The agent was programmed to work in windows with "Minesweeper X", a clone of the original game, 
see http://www.curtisbright.com/msx/

# TODO:
- Improve guessing logic (currently purely random)
- Use a CNN instead of simple colour recognition to classify cells (for fun)
- Allow the game to discover a partially completed game.
- Improve grid detection if game window is rescaled
