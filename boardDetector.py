import numpy as np
import cv2
from PIL import Image, ImageGrab
from minesweeper import MineSweeperGame

# Find the minesweeper game squares
def find_game(screen, showVision=False):
    """
    Find the minesweeper game squares from a screenshot.

    :param screen: numpy array of the screen
    :return: list of squares representing game board [(x,y,w,h), ...]
    """
    
    # Process screenshot to make it easier to analyze
    gray = cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(blur,170,255,apertureSize = 5)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.erode(edges,kernel,iterations = 1)
    contours, _ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find which contours have the same dimensions as the game squares
    squares = []
    for i, c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        if h == 14 and w == 14 and a > 160 and a < 175 :
            squares.append((x,y,w,h))
    squares = np.array(squares)

    grid = extract_grid(squares)
    
    if showVision:
        # Draw squares on the screen
        if grid is not None:
            for (x,y,w,h) in grid:
                cv2.rectangle(screen,(x,y),(x+w,y+h),(0,255,0),3)
            Image.fromarray(screen).show()
        else:
            Image.fromarray(edges).show()

    if grid is not None:
        return grid
    else:
        print("Failed to find the game")
        return None


def extract_grid(squares):
    """
    Takes a list of squares extracted from a screengrab. Try to extract a grid.
    If a grid cannot be found, return None.

    Screening steps include removing outlier squares and checking that enough
    squares were found. 
    """
    if len(squares) == 0:
        print("Failed: no squares input to grid extractor")
        return None
    
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
        print("Failed: not enough squares found")
        return None
    
    ### Check if the squares form a single complete grid
    valsX = np.unique(squares[:,0])
    valsY = np.unique(squares[:,1])

    # Check if x values are evenly spaced
    if not all(np.diff(valsX) == np.diff(valsX)[0]):
        print("Failed: x values are not evenly spaced")
        return None
    
    # Check if y values are evenly spaced
    if not all(np.diff(valsY) == np.diff(valsY)[0]):
        print("Failed: y values are not evenly spaced")
        return None
    
    # Check if there is the right number of squares for a complete grid
    if len(valsX) * len(valsY) != len(squares):
        print("Failed: not a complete grid")
        return None
    
    # Check if the grid is square
    if len(valsX) != len(valsY):
        print("Failed: grid is not square")
        return None
    
    # Check if every value in the grid is present
    expected = set([(x,y) for x in valsX for y in valsY])
    actual = set([(x,y) for (x,y,w,h) in squares])
    if expected != actual:
        print("Failed: not all squares are present")
        return None
    
    # If all checks pass, return the grid
    return squares


def check_cells(game: MineSweeperGame, cells = None):
    """ 
    Check a set of cells for changes. If no cells are specified, check all.
    Assumes the game window has not moved.
    """
    # If a list of cells is not provided, iterate over all cells. 
    if cells is None:
        cells = [(r,c) for r in range(game.height) for c in range(game.width)]
    
    # Take a screenshot
    screen = ImageGrab.grab()
    # If the value of a cell is unknown, check it
    for cell in cells:
        if np.isnan(game[cell]).any():
            # Extract the right part of the screenshot 
            contents = screen.crop(game.locate_cell(cell))
            # Check the cell for changes
            val = parse_cell(game, contents)
            game.update_cell(cell, val)
                

def parse_cell(game, cell_contents):
    """
    Get the value of a single minesweeper cell: e.g. unknown, 0,1,2,3,4,5, mine

    input: a cell from the minesweeper game
    output: the cell value:
        NaN = unknown
        -1 = flag?
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
    cell_center = cell_contents.crop((1,1,game.cell_width-3,game.cell_height-3))
    cell_center = np.array(cell_center)

    # Check if the cell is uniform grey (mouse pointer isn't in the screenshot)
    if len(np.unique(cell_center)) == 1:
        # Check if the cell has a bevel
        if len(np.unique(cell_contents)) == 2:
            # No bevel, cell is zero
            return 0
        else:
            # Bevel, cell is unknown/unclicked
            return np.nan
    else:
        # Cell is not uniform grey, check if cell is a digit 
        digit = read_digit(cell_center)
        if digit:
            return digit

        # If any cell has a pitch black pixel and a red pixel, assume it is a mine
        if col_in_img([0,0,0], cell_center):
            if col_in_img([255,0,0], cell_center):
                # Cell is a mine
                return -1

    # Could not read cell
    return np.nan

        # If cell is not a mine, it must be a digit. Use CNN to find digit
        # TODO


def read_digit(img):
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
    
    
def col_in_img(colour, img):
    """
    Check if an image contains any pixel with a specific colour

    Input:
        colour: a list of 3 integers representing the colour to check for
        img: a numpy array representing the image to check
    """
    return (img == np.array(colour)).all(axis=2).any()

    
    
    


    
    



# start_square = []
# # Find restart button
# from statistics import median
# for i, c in enumerate(contours):
#     x,y,w,h = cv2.boundingRect(c)
#     if abs(x - median(xpos)) < 20:
#         if y - max(ypos) < 0 and y - max(ypos) > -20: 
#             if abs(w-h) < 5:
#                 start_square.append(c)
#                 break




# # Add new class representing the minesweeper game board
# class MinesweeperBoard:
#     def __init__(self, grid):
#         self.grid = grid
#         self.rows = len(grid)
#         self.cols = len(grid[0])
#         self.size = self.rows * self.cols
#         self.mines = 0
#         self.squares = 0
#         self.squares_left = 0
#         self.mines_left = 0

#     def __repr__(self):
#         return str(self.grid)

#     def __getitem__(self, key):
#         return self.grid[key]

#     def __setitem__(self, key, value):
#         self.grid[key] = value

#     def __iter__(self):
#         return iter(self.grid)

#     def __len__(self):
#         return self.size

#     def __eq__(self, other):
#         return self.grid == other.grid

#     def __ne__(self, other):
#         return not self == other

#     def update(self, screen):
#         # Update/check grid:
#         # Validate that grid hasn't moved/been closed
#         # Check state of each grid square.
#         pass

#     def check(self):
#         # Validate that grid hasn't moved/been closed
#         # Check state of each grid square.
#         pass

#     def output(self):
#         # Output screen (x,y) position and state of every cell in grid
#         pass

#     def get_mines_left(self):
#         return self.mines_left

#     def get_squares_left(self):
#         return self.squares_left

#     def get_squares(self):
#         return self.squares

#     def get_mines(self):
#         return self.mines

#     def get_size(self):
#         return self.size

#     def get_rows(self):
#         return self.rows

#     def get_cols(self):
#         return self.cols

#     def get_grid(self):
#         return self.grid
