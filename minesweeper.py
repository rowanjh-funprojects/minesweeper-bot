import itertools
import random
import numpy as np

# Convert the grid to game object class (maybe port into AI brain)
class MineSweeperGame():
    """ 
    Representation of minesweeper game window and observed game state
    """
    def __init__(self, nrows, ncols, cell_width, cell_height, xmin, ymin, reset_pos):
        # Game state info
        self.height = nrows
        self.width = ncols
        self.size = nrows * ncols
        self.cells = np.empty((nrows,ncols))
        self.cells[:] = None

        # Game window info
        self.xmin = xmin # top left corner x
        self.ymin = ymin # top left corner y
        self.cell_height = cell_height
        self.cell_width = cell_width
        self.reset_pos = reset_pos # restart button position
    

    def __getitem__(self, index):
        return self.cells[index]


    @classmethod
    def from_grid(cls, grid):
        if grid is None: 
            raise ValueError('Grid is empty')
        xvals = np.unique(grid[:,0])
        yvals = np.unique(grid[:,1])
        nrows = len(yvals)
        ncols = len(xvals)
        xmin = min(xvals)
        ymin = min(yvals)
        xmax = max(xvals)
        ymax = max(yvals)
        cell_width = (xmax-xmin) / (ncols-1)
        cell_height = (ymax-ymin) / (nrows-1)
                
        # Offset grid position because contours detect the inner corner.
        xmin -= cell_width // 16
        xmax -= cell_width // 16
        ymin -= cell_height // 16
        ymax -= cell_height // 16

        # restart button position
        reset_pos = (xmin + (xmax-xmin + cell_width)/2, ymin - cell_width*2)
        return cls(nrows,ncols,cell_width,cell_height,xmin,ymin,reset_pos)
    

    def __str__(self):
        return ("Minesweeper Board\n" + 
                str(self.cells))
    
    def update_cell(self,cell,new_val):
        self.cells[cell] = new_val

    
    def get_game_bbox(self):
        return ((self.xmin, self.ymin),
                (self.xmin + self.width * self.cell_width, 
                 self.ymin + self.height * self.cell_height))
    

    def get_all_cells(self):
        """
        return list of tuples of all cells in the game
        """
        return [(r,c) for r in range(self.height) for c in range(self.width)]


    def get_unclicked_cells(self):
        """
        return list of tuples of unclicked cells
        """
        cells = self.get_all_cells()
        return [cell for cell in cells if np.isnan(self[cell]).any()]

    
    def reset_game(self):
        """
        Reset game state
        """
        self.cells[:] = None


    def locate_cell(self, cell, bbox = True):
        """
        Get screen pixel coordinates of a given cell. Return a bounding box if 
        box = True, or pixel coordinates of top left corner if box = False

        Parameters
        ----------
        cell : tuple
            (row, col) of cell to locate
        bbox : bool, optional
            Return bounding box of cell if True, else return top-left point

        Returns
        -------
        tuple
            Pixel coordinates of cell, (left, top, right, bottom), or (left, top)

        """
        # Check cell is valid
        if cell[0] < 0 or cell[0] >= self.height:
            raise ValueError(f'Row {cell[0]} is out of bounds')
        if cell[1] < 0 or cell[1] >= self.width:
            raise ValueError(f'Column {cell[1]} is out of bounds')

        left = self.xmin + cell[1] * self.cell_width
        top = self.ymin + cell[0] * self.cell_height
        right = left + self.cell_width
        bottom = top + self.cell_height
        if bbox:
            return ((left,top,right,bottom))
        else:
            return (left,top)
        
