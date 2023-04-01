import itertools
import random
import numpy as np

# Convert the grid to game object class (maybe port into AI brain)
class MineSweeperGame():
    """ 
    Representation of minesweeper game window and game state
    """
    def __init__(self, nrows, ncols, cell_width, cell_height, xmin, ymin):
        # Game state info
        self.height = nrows
        self.width = ncols
        self.cells = np.empty((nrows,ncols))
        self.cells[:] = np.nan

        # Game window info
        self.xmin = xmin # top left corner x
        self.ymin = ymin # top left corner y
        self.cell_height = cell_height
        self.cell_width = cell_width
    
    @classmethod
    def from_grid(cls, grid):
        xvals = np.unique([x for (x,y,w,h) in grid])
        yvals = np.unique([y for (x,y,w,h) in grid])
        r = len(xvals)
        c = len(yvals)
        w = grid[0][2]
        h = grid[0][3]
        xmin = min(xvals)
        ymin = min(yvals)
        return cls(r,c,w,h,xmin,ymin)
    
    def __str__(self):
        return ("Minesweeper Board\n" + 
                f'Window location:{self.get_game_bbox()}\n' + 
                str(self.cells))
    
    def update_cell(self,row,col,new_val):
        self.cells[row][col] = new_val
    
    def get_game_bbox(self):
        return ((self.xmin, self.ymin),
                (self.xmin + self.width * self.cell_width, 
                 self.ymin + self.height * self.cell_height))
