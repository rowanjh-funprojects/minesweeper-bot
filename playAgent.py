import random
import mouse
import keyboard
import time
import numpy as np
import cv2
import os
import subprocess

from PIL import Image, ImageGrab, ImageDraw

from minesweeper import MineSweeperGame


# Make program aware of windows DPI scaling
import platform
import ctypes
if platform.system() == 'Windows':
    PROCESS_PER_MONITOR_DPI_AWARE = 2
    ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

SPEED = 0.1

class playAgent():
    """
    Minesweeper game player
    Interface: 
    ai.add_knowledge- add knowledge about the game board to the AI (cell, count) 
    ai.make_safe_move- make a safe move
    ai.make_random_move- make a random move

    """

    def __init__(self):

        # Keep track of which cells have been clicked on
        self.moves_made = set()
        self.last_move = (None, None)

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.flags = set()
        self.safes = set()
        self.unresolved_conclusions = []

        # List of sentences about the game known to be true
        self.knowledge = []

        # Game window and game state
        self.game_grid = None
        self.game = None
        self.lost = False
        self.won = False


    def activate(self):
        """
        Activate the AI to play the game.
        """
        print("Hello! I'm a Minesweeper AI")
        while True:
            print("Do you want me to launch the game for you? (y/n) ", end = "")
            response = input()
            if response == 'y':
                # Launch minesweeper X
                wd = os.getcwd()
                subprocess.Popen(os.path.join(wd, 'Minesweeper X.exe'))
                time.sleep(1)
                break
            elif response == 'n':
                break

        print("\nPlease make sure the full minesweeper game window is visible ")
        print("The game should be fresh (no cells clicked)")

        while True:
            print("\nAre you ready? (y/n) ", end = "")
            response = input()

            if response == 'y':
                try:
                    self.find_game_grid()
                    self.initialize_game_grid()
                except:
                    print("I failed to find the window, please try again.")
                    continue
                self.play()
                print("Do you want to try again?(y/n) ", end = "")
                if input() == 'y':
                    self.restart()
                    continue
                else: 
                    break
            
        print("Goodbye!")
        return

        
    def find_game_grid(self, showVision=False):
        """
        Find the minesweeper game squares from a screenshot.

        :param screen: numpy array of the screen
        :return: list of squares representing game board [(x,y,w,h), ...]
        """
        screen = np.array(ImageGrab.grab()) # for the full screen
        
        # Process screenshot to extract contours : pipeline 1 (unscaled window)
        gray = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.Canny(close,170,255,apertureSize = 5)
        kernel = np.ones((3,3),np.uint8)
        edges = cv2.dilate(edges,kernel,iterations = 1)
        kernel = np.ones((5,5),np.uint8)
        edges = cv2.erode(edges,kernel,iterations = 1)

        # # pipeline 2 (for upscaled game window)
        # gray = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
        # blur = cv2.medianBlur(gray, 5)
        # # Rectangular kernel
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=2)
        # edges = cv2.Canny(close,170,255,apertureSize = 5)
        # kernel = np.ones((3,3),np.uint8)
        # edges = cv2.dilate(edges,kernel,iterations = 3)
        # kernel = np.ones((5,5),np.uint8)
        # edges = cv2.erode(edges,kernel,iterations = 2)

        contours, _ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Find which contours have the same dimensions as the game squares
        squares = [] # each square is list of vertices for the detected contour
        for i, c in enumerate(contours):
            x,y,w,h = cv2.boundingRect(c)
            if h != w:
                # only keep squares
                continue
            a = cv2.contourArea(c)
            if h > 12 :
                squares.append((x,y,w,h))
        squares = np.array(squares)

        grid = self.grid_from_squares(squares)
        
        # # Offset grid position because contours detect the inner corner.
        # Set values fail when window is rescaled.
        # grid[:,0] -= 1
        # grid[:,1] -= 1
        
        if showVision:
            # Draw squares on the screen
            if grid is not None:
                for (x,y,w,h) in grid:
                    cv2.rectangle(screen,(x,y),(x+w,y+h),(0,255,0),3)
                Image.fromarray(screen).show()
            else:
                Image.fromarray(edges).show()

        if grid is not None:
            self.game_grid = grid
        else:
            print("Failed to find the game")
    
    
    def grid_from_squares(self, squares):
        """
        Takes a list of squares extracted from a screengrab. Try to extract a grid.
        If a grid cannot be found, return None.

        Screening steps include removing outlier squares and checking that enough
        squares were found. 
        """
        if len(squares) == 0:
            raise Exception("Failed: zero game tiles were detected on the screen")
        
        ### Remove outliers (squares that aren't aligned with other squares)
        # Get frequency of x and y coordinates of all squares
        valsX, countX = np.unique(squares[:,0], return_counts=True)
        valsY, countY = np.unique(squares[:,1], return_counts=True)
        
        # Remove any outlier squares that do not align with at least 5 other squares
        validX = valsX[countX > 5]
        mask = np.isin(squares[:, 0], validX)
        squares = squares[mask, :]
        validY = valsY[countY > 5]
        mask = np.isin(squares[:, 1], validY)
        squares = squares[mask, :]
        
        ### Check if enough squares were found to make a grid
        if squares.shape[0] < 50:
            raise Exception("Failed: not enough game tiles found")
        
        ### Check if the squares form a single complete grid
        valsX = np.unique(squares[:,0])
        valsY = np.unique(squares[:,1])
        
        # Check if every value in the grid is present
        expected = set([(x,y) for x in valsX for y in valsY])
        actual = set([(x,y) for (x,y,w,h) in squares])
        if expected != actual:
            print("Failed: incomplete grid")
            return None

        # Check if x values are evenly spaced
        if not all(np.diff(valsX) == np.diff(valsX)[0]):
            raise Exception("Failed: game tiles are not evenly spaced (x)")
        
        # Check if y values are evenly spaced
        if not all(np.diff(valsY) == np.diff(valsY)[0]):
            raise Exception("Failed: game tiles are not evenly spaced (y)")
        
        # If all checks pass, return the grid
        return squares


    def initialize_game_grid(self):
        """
        Register the game grid and create an internal representation of the game
        """
        # Game window info
        self.game = MineSweeperGame.from_grid(self.game_grid)
        


    def play(self, nmoves = None, verbose = False):
        """
        Play the game until it is won or lost.

        Parameters
        ----------
        nmoves : int, optional
            The maximum number of moves to make. If None, play until the game is won or lost.
        verbose : bool, optional
            If True, print the game state after each move.

        Returns
        -------
        None.
        """
        maxiter = 2000 if nmoves is None else nmoves
        # Startup message
        print("Lets play! Hold s to stop, p to pause, or r to restart")
        for i in range(maxiter):
            # Quit if s is pressed
            if keyboard.is_pressed('s') or keyboard.is_pressed('esc'):
                print("Stopping play")
                break
            # Pause if p is pressed
            if keyboard.is_pressed('p'):
                print("Paused. Press p again to unpause")
                time.sleep(1)
                keyboard.wait('p')
                print("unpaused")
            if keyboard.is_pressed('r'):
                print("Restarting")
                self.restart()
            # Check values of all unexplored cells
            # Check win conditions
            if self.won:
                print("I won!")
                break
            if self.lost:
                print("I lost >:(")
                break

            # Make a move
            if i > 5 and verbose:
                print(self.game)
            move = self.plan_move()
            self.execute_move(move)
            try:
                self.check_cells([move])
            except:
                print("Something went wrong checking cells. I'm quitting.")
                break
            if verbose:
                print(f"Making the move {move}. I think the value there is {self.game.cells[move]}\n")
    

    def restart(self):
        """
        Click the restart button
        """
        mouse.move(self.game.reset_pos[0], self.game.reset_pos[1], duration = 1)
        mouse.double_click()
        # Wait a little bit for screen to catch up?
        time.sleep(1)
        self.lost = False
        self.game.reset_game()

        self.moves_made = set()
        self.last_move = (None, None)

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()
        self.unresolved_conclusions = []

        # List of sentences about the game known to be true
        self.knowledge = []


    def check_cells(self, cells = None, screen = None):
        """ 
        Check a set of cells for changes. If no cells are specified, check all.
        Assumes the game window has not moved.
        Check if won or lost
        """
        # Take a screenshot
        if screen is None:
            screen = ImageGrab.grab()

        # If a single cell is specified, check only that.
        if cells:
            lazy = True
            if len(cells) > 1:
                raise Exception("Can only give one check_cells one target")
        else:
            # Otherwise check all cells
            cells = self.game.get_unclicked_cells()
            lazy = False
        
        for cell in cells:
            # Extract the right part of the screenshot 
            contents = screen.crop(self.game.locate_cell(cell))
            # Check the value of the cell
            val = self.parse_cell(contents)
            # Update the game representation: observed cell values.
            self.game.update_cell(cell, val)
            # Update beliefs about uncovered cell based on observations
            if (val is not np.nan):
                self.add_knowledge(cell, val)
                # If a zero is found during lazy check, then many squares were
                # maybe revealed, so just check the whole grid of unknowns.
                if val == 0 and lazy:
                    self.check_cells(screen = screen)
            else:
                if lazy:
                    raise Exception("Clicked cell detected as na, bad vision or click")
        
        # Check win/lose conditions
        if -1 in self.game.cells:
            self.lost = True
        if (len(self.mines) + len(self.moves_made)) == (self.game.size):
            self.won = True
    

    def parse_cell(self, cell_img):
        """
        Get the value from an image of a cell: e.g. unknown, 0,1,2,3,4,5, mine

        input: a cell from the minesweeper game
        output: the cell value:
            NaN = unknown
            -1 = mine
            -2 = flag
            0-8 = number of mines adjacent to cell

        Possible algorithm: 
        Find screen location of cell
        Clip screen at that location
        if central pixels in square are all uniform grey. 
            if square has a bevel: unknown
            if square has no bevel: zero
        else if central pixels are not uniform:
            if mine, game over
            if not mine, check what the digit is with CNN

        """
        buff = self.game.cell_width // 8 # accounts for scaled game window
        cell_center = cell_img.crop((buff,buff,self.game.cell_width-buff,self.game.cell_height-buff))
        cell_center = np.array(cell_center)

        # Check if the cell is uniform grey (mouse pointer isn't in the screenshot)

        # if R==B==G, cell is fully greyscale. Don't allow black pixels (mine)
        r,g,b = cell_center[:,:,0], cell_center[:,:,1], cell_center[:,:,2]
        if (r == g).all() and (r == b).all() and not 0 in r:
            # Check if the cell has a bevel
            if col_in_img([255, 255, 255], np.array(cell_img)):
                # Cell has a bevel, cell is nan
                return np.nan
            else:
                # Cell has no bevel, it is unknown/unclicked.
                return 0
        else:
            # If any cell has a pitch black pixel and a white pixel, it must be
            # a mine
            if col_in_img([0,0,0], cell_center):
                if col_in_img([255,255,255], cell_center):
                    # Cell is a mine
                    return -1
            # If the cell has a black pixel and a red pixel, and it is not 
            # mine, then it must be a flag 
            if col_in_img([0,0,0], cell_center):
                if col_in_img([255,0,0], cell_center):
                    # Cell is a flag
                    return -2
            # Check for digits
            digit = self.read_digit(cell_center)
            if digit:
                return digit
        # Could not read cell
        return np.nan
    

    def read_digit(self, img):
        """
        This version uses simple heuristics to determine what the digit is.
        Ultimately this will be implemented with a CNN.

        Input: a 16x16 image of a digit
        Output: the digit
        """
        # 1 = [0,0,255]
        # 2 = [0, 128, 0]
        # 3 = [255,0,0]
        # 4 = [0,0,128]
        # 5 = [128,0,0]
        # 6 = [0,128,128]
        # 7 = [0,0,0]
        # 8 = [128,128,128] # unsupported

        img = np.array(img)

        # Check if any pixels are blue
        if col_in_img(img, [0,0,255]):
            return 1
        # Check if any pixels are green
        if col_in_img(img, [0,128,0]):
            return 2
        # Check if any pixels are red
        if col_in_img(img, [255,0,0]):
            return 3
        # Check if any pixels are dark blue
        if col_in_img(img, [0,0,128]):
            return 4
        # Check if any pixels are dark red
        if col_in_img(img, [128,0,0]):
            return 5
        # Check if any pixels are teal
        if col_in_img(img, [0,128,128]):
            return 6


    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        if count == -1 or count == -2:
            # If the move was a mine, then the game is over. If the move was a
            # flag, no new knowledge has been gained.
            return
        # Add this cell to moves made
        self.moves_made.add(cell)
        # Note that this was a safe cell
        self.process_safe(cell)
        # Get neighbours of this cell.
        cells = self.get_neighbours(cell)
        # Represent knowledge as a sentence
        sentence = Sentence(cells, count)
        # give the AI's knowledge about the game board to the sentence (known safes or mines)
        sentence = self.teach(sentence)
        # Add the new sentence to the knowledge base
        self.knowledge.append(sentence)
        # Revise the knowledge base: see if any new conclusions can be made
        self.revise_knowledge()

    def process_mine(self, cell):
        """
        Marks a cell as a mine, and updates all previous knowledge
        sentences containing that cell accordingly
        """
        self.mines.add(cell)
        if not [cell, "mine"] in self.unresolved_conclusions:
            self.unresolved_conclusions.append([cell, "mine"])
        for s in self.knowledge: 
            s.mark_mine(cell)

    def process_safe(self, cell):
        """
        Marks a cell as safe,  and updates all previous knowledge
        sentences containing that cell accordingly
        """
        # If a cell is a mine or safe, it should be marked as safe in
        # all knowledge statements
        if cell not in self.safes:
            self.safes.add(cell)
            self.unresolved_conclusions.append([cell, "safe"])
            for s in self.knowledge:
                s.mark_safe(cell)


    def check_conclusions(self, sentence):
        """
        Conclude that cells in the set are safe or mines, if possible,
        and mark as mine or safe accordingly
        """
        # If count = number of cells, everything is a mine
        if len(sentence.cells) == sentence.count:
            [self.process_mine(c) for c in sentence.cells]
        # If count == 0, everything is safe
        elif sentence.count == 0:
            [self.process_safe(c) for c in sentence.cells]
        # If the sentence is a subset of some other sentence, then you can create a new sentence.
        new_knowledge = []
        for s in self.knowledge:
            # Don't compare a sentence to itself or some other identical sentence
            if sentence.cells == s.cells:
                continue
            if sentence.cells.issubset(s.cells):
                newsentence = Sentence(s.cells.difference(sentence.cells), s.count - sentence.count)
                newsentence = self.teach(newsentence)
                # Check that the sentence doesn't already exist
                if newsentence not in self.knowledge:
                    new_knowledge.append(newsentence)
        if len(new_knowledge) > 0:
            x=1
            [self.knowledge.append(s) for s in new_knowledge]
        
    
    def tidy_knowledgebase(self):
        """
        Remove mines and safes from sentences, and remove any sentences that are empty
        """
        # Loop over knoweldge statements (backwards so iteration doesn't skip entries)
        for i in range(len(self.knowledge)-1, -1, -1): 
            # Purge known mines/safes from sentences
            self.knowledge[i].purge()
            # If any sentences became empty, remove them.
            if len(self.knowledge[i].cells) == 0:
                del self.knowledge[i]
        
    def teach(self, sentence):
        """
        Take a naive sentence and tell it where the known mines or safe squares are.
        """
        [sentence.mark_mine(c) for c in self.mines]
        [sentence.mark_safe(c) for c in self.safes]
        return sentence

    def revise_knowledge(self):
        """
        Check if new safes or mines were identified. If so, go through the knowledge base,
        update things, and try to make new conclusions.
        """
        # Tidy up all sentences (remove mines/safes from sentence, cull empty sentence)
        self.tidy_knowledgebase()
        # Try to make new conclusion from old sentences again
        for s in self.knowledge:
            self.check_conclusions(s)

        # Condition to break recursion 
        if len(self.unresolved_conclusions) == 0:
            return
        # If there is no knowledge, there is nothing to revise...
        if len(self.knowledge) == 0:
            return

        ## Loop over all of the new mines and new safes, and update old sentences 
        while len(self.unresolved_conclusions) > 0:
            newitem = self.unresolved_conclusions.pop()
            for s in self.knowledge:
                if newitem[0] in s.cells:
                    if newitem[1] == "safe":
                        s.mark_safe(newitem[0])
                    if newitem[1] == "mine":
                        s.mark_mine(newitem[0])
        
        # recurse, in case any new mines or safes were added.
        self.revise_knowledge()


    def plan_move(self):
        """
        Decide whether to make a safe move or a random move
        """
        # If there are any known mines that have not been flagged yet, flag them
        unflagged_mines = [m for m in self.mines if m not in self.flags]
        if unflagged_mines:
            self.flag_mine(unflagged_mines[0])

        # Make a safe move if possible, or else make a random move
        unplayed_cells = self.game.get_unclicked_cells()
        unplayed_safes = [cell for cell in unplayed_cells if cell in self.safes]

        if len(unplayed_safes) > 0:
            # Return a safe cell
            return unplayed_safes[0]
        else:
            # Guess a cell
            return self.guess_move()


    def flag_mine(self, cell):
        """
        Flag a mine
        """
        self.flags.add(cell)
        self.execute_move(cell, type = 'right')

    def guess_move(self):
        """
        Make a random move
        """
        # Make a random move
        unplayed_cells = self.game.get_unclicked_cells()
        cells = [c for c in unplayed_cells if c not in self.mines]
        return random.choice(cells)


    def execute_move(self, move, type = 'left', duration = SPEED):
        # convert movement to pixel coordinates
        (x,y) = self.cell_to_pixel(move)
        self.last_move = move

        # move mouse to cell and click
        mouse.move(x,y,duration = duration)
        mouse.click(button = type)
        # Wait a little bit for screen to catch up?
        time.sleep(0.1)


    def cell_to_pixel(self, cell):
        """
        Converts a cell (row, col) to pixel coordinates (x, y)
        """
        x = self.game.xmin + self.game.cell_width * cell[1] + self.game.cell_width/2
        y = self.game.ymin + self.game.cell_height * cell[0] + self.game.cell_height/2
        return (x,y)
        

    def get_neighbours(self, cell):
        """
        Get the neighbours of a target cell.
        """
        neighbours = []
        for i in range(3):
            for j in range(3):
                thisi = cell[0]-1 + i
                thisj = cell[1]-1 + j
                # Only accept values within the board
                if thisi in range(self.game.height) and thisj in range(self.game.width):
                    # Only consider unplayed cells.
                    if (thisi, thisj) not in self.moves_made:
                        neighbours.append((thisi,thisj))
        return neighbours
    

    def show_beliefs(self):
        """
        Show on the screen where it thinks the board is, what the safe cells
        and mines are
        """

        screen = ImageGrab.grab()
        # Get clip of game grid
        box = self.game.get_game_bbox()
        gamewindow = screen.crop((box[0][0], box[0][1], box[1][0], box[1][1]))
        draw = ImageDraw.Draw(gamewindow)
        # Draw rectangles on grid
        for cell in self.game.get_all_cells():
            # Convert cell to pixel coordinates
            (x,y) = self.cell_to_pixel(cell)
            # Get corner of cell instead of middle
            (x,y) = (x - self.game.cell_width/2, y - self.game.cell_height/2)
            # Convert to relative coordinates in the crop
            (x,y) = (x - self.game.xmin, y - self.game.ymin)
            # Colour code cells
            if cell in self.safes:
                col = "green"
            elif cell in self.mines:
                col = "red"
            else:
                col = "blue"

            # Draw rectangles according to colour code
            draw.rectangle(
                [(x+2,y+2), (x+self.game.cell_width-2, y+self.game.cell_height-2)],
                outline = col)            
        # Show image
        gamewindow.show()


    

class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count
        self.mines = set()
        self.safes = set()

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # Self
        return self.mines

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        return self.safes

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.mines.add(cell)

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.safes.add(cell)

    def purge(self):
        """
        Purse marked mines and safes from a statement
        """
        purge_list = []
        for c in self.cells:
            if c in self.safes:
                purge_list.append(c)
            elif c in self.mines:
                purge_list.append(c)
                self.count -= 1
        self.cells.difference_update(purge_list)


def col_in_img(colour, img):
    """
    Check if an image contains any pixel with a specific colour

    Input:
        colour: a list of 3 integers representing the colour to check for
        img: a numpy array representing the image to check
    """
    return (img == np.array(colour)).all(axis=2).any()
