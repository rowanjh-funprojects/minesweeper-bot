import numpy as np
import cv2

# Find the minesweeper game squares
def find_game(screen):
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
    
    grid = extract_grid(squares)

    if grid:
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
    ### Remove outliers (squares that aren't aligned with other squares)
    # Get frequency of x and y coordinates of all squares
    valsX, countX = np.unique([x for (x,y,w,h) in squares], return_counts=True)
    valsY, countY = np.unique([y for (x,y,w,h) in squares], return_counts=True)

    # Find outlier squares that do not align with at least 5 other squares
    x_outliers = set([valsX[i] for i, c in enumerate(countX) if c < 5])
    y_outliers = set([valsY[i] for i, c in enumerate(countY) if c < 5])

    # Remove any outlier points
    if x_outliers or y_outliers:
        squares = [
            s for s in squares 
            if s[0] not in x_outliers and s[1] not in y_outliers]
    
    ### Check if enough squares were found to make a grid
    if len(squares) < 50:
        print("Failed: not enough squares found")
        return None
    
    ### Check if the squares form a single complete grid
    valsX = np.unique([x for (x,y,w,h) in squares])
    valsY = np.unique([y for (x,y,w,h) in squares])

    # Check if x values are evenly spaced
    if not np.allclose(np.diff(valsX), valsX[1] - valsX[0]):
        print("Failed: x values are not evenly spaced")
        return None
    
    # Check if y values are evenly spaced
    if not np.allclose(np.diff(valsY), valsY[1] - valsY[0]):
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
