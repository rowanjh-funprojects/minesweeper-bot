import random
import mouse
import time

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


class playAgent():
    """
    Minesweeper game player
    Interface: 
    ai.add_knowledge- add knowledge about the game board to the AI (cell, count) 
    ai.make_safe_move- make a safe move
    ai.make_random_move- make a random move

    """

    def __init__(self, game):
        # Game window info
        self.game = game

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()
        self.unresolved_conclusions = []

        # List of sentences about the game known to be true
        self.knowledge = []
    
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
        # If there are no safe moves, make a random move
        if len(self.safes) == 0:
            return self.plan_random_move()
        # If there are safe moves, make one
        else:
            return self.plan_safe_move()

    def plan_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        unplayed_cells = self.get_unplayed_squares()
        safe_choices = [cell for cell in unplayed_cells if cell in self.safes]

        if len(safe_choices) > 0:
            # Return the first safe move that has not yet been played, if possible.
            return random.choice(safe_choices)
        else:
            return None

    def plan_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # If there are no safe moves, pick any square that has not yet 
        # been selected, and isn't a known mine
        unplayed_cells = self.get_unplayed_squares()
        random_choices = [cell for cell in unplayed_cells if cell not in self.mines]
        if len(random_choices) == 0:
            return None
        else:
            return random.choice(random_choices)

    def execute_move(self, move, duration = 0.1):
        # convert movement to pixel coordinates
        (x,y) = self.cell_to_pixel(move)

        # move mouse to cell and click
        mouse.move(x,y,duration = duration)
        mouse.press()
        time.sleep(duration+0.1)
        mouse.release()


    def cell_to_pixel(self, cell):
        """
        Converts a cell (row, col) to pixel coordinates (x, y)
        """
        x = self.game.xmin + self.game.cell_width * cell[0] + self.game.cell_width/2
        y = self.game.ymin + self.game.cell_height * cell[1] + self.game.cell_height/2
        return (x,y)
        
    def get_unplayed_squares(self):
        """
        Returns a list of tuples (row,col) representing cells that have not yet been played
        """
        cells = []
        for i in range(self.game.height):
            for j in range(self.game.width):
                if (i,j) not in self.moves_made:
                    cells.append((i,j))
        return cells
    
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
    
