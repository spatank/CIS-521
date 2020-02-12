############################################################
# CIS 521: Homework 4
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import copy
import numpy as np
import random

############################################################
# Section 1: Sudoku Solver
############################################################

def sudoku_cells():
    cells = []
    for i in range(9): 
        for j in range(9):
            cell = (i,j)
            cells.append(cell)
    return cells

def sudoku_arcs():
    cells = sudoku_cells()
    arcs = []
    for cell_1 in cells:
        for cell_2 in cells:
            if cell_1 == cell_2:
                continue
            if cell_1[0] == cell_2[0]: # same row
                arcs.append((cell_1, cell_2))
                continue
            if cell_1[1] == cell_2[1]: # same column
                arcs.append((cell_1, cell_2))
                continue
            if (cell_1[0]//3) == (cell_2[0]//3): 
                if (cell_1[1]//3) == (cell_2[1]//3): # same block
                    arcs.append((cell_1, cell_2))
    return arcs

def read_board(path):
    with open(path, 'r') as f:
        x = f.readlines()
    array = [row.rstrip('\n') for row in x] # strip newline character
    board = {}
    for i, row_list in enumerate(array):
        for j, col in enumerate(row_list):
            idx_tup = (i, j)
            curr_val = col
            if col == '*': # any of {1,...,9} valid
                board[idx_tup] = set(list(range(1,len(array)+1)))
            else: # value is fixed at initiation   
                board[idx_tup] = set([int(col)])
    return board

class Sudoku(object):
    
    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()
    
    def __init__(self, board):
        self.board = board
    
    def get_values(self, cell):
        board = self.board
        return board[cell]
    
    def remove_inconsistent_values(self, cell_1, cell_2):
        board = self.board
        if (cell_1, cell_2) in self.ARCS:
            domain_1 = board[cell_1]
            domain_2 = board[cell_2]
            if len(domain_2) == 1: # cell_2 is fixed
                if domain_2.issubset(domain_1): 
                    # domain of cell_2 is available for deletion
                    # from domain of cell_1
                    domain_1 = domain_1 - domain_2
                    board[cell_1] = domain_1
                    return True
        return False
    
    def print_board(self):
        board = self.board
        array = np.zeros([9,9])
        for row in range(9):
            for col in range(9):
                idx_tup = (row, col)
                copy_set = copy.deepcopy(board[idx_tup])
                val = copy_set.pop()
                if len(board[idx_tup]) == 1:
                    array[row][col] = int(val)
        print(array)
    
    def infer_ac3(self):
        board = self.board
        arcs = set()
        for arc in self.ARCS:
            arcs.add(arc)
        while arcs:
            arc = arcs.pop()
            cell_1 = arc[0]
            cell_2 = arc[1]
            if self.remove_inconsistent_values(cell_1, cell_2):
                for cell_pair in self.ARCS: 
                    if cell_pair[1] == cell_1:
                        arcs.add(cell_pair)
                        
    def row_cells(self, cell):
        row_cells_domain = set()
        board = self.board
        CELLS = self.CELLS
        for CELL in CELLS:
            if CELL == cell:
                continue
            if CELL[0] == cell[0]:
                if len(board[CELL]) != 1:
                    row_cells_domain = row_cells_domain | board[CELL]
                    # add the domain of CELL to the set containing all row domain
        return row_cells_domain
    
    def col_cells(self, cell):
        col_cells_domain = set()
        board = self.board
        CELLS = self.CELLS
        for CELL in CELLS:
            if CELL == cell:
                continue
            if CELL[1] == cell[1]:
                if len(board[CELL]) != 1:
                    col_cells_domain = col_cells_domain | board[CELL]
                    # add the domain of CELL to the set containing all col domain
        return col_cells_domain
    
    def block_cells(self, cell):
        block_cells_domain = set()
        board = self.board
        CELLS = self.CELLS
        for CELL in CELLS:
            if CELL == cell:
                continue
            if (CELL[0]//3) == (cell[0]//3):
                if (CELL[1]//3) == (cell[1]//3):
                    if len(board[CELL]) != 1:
                        block_cells_domain = block_cells_domain | board[CELL]
                        # add the domain of CELL to the set containing all row domain
        return block_cells_domain
    
    def infer_improved(self):
        board = self.board
        CELLS = self.CELLS
        self.infer_ac3()
        flag = True
        while flag == True:
            flag = False
            for cell in CELLS:
                if len(board[cell]) == 1: # continue if value fixed
                    continue
                # curr_domain = copy.deepcopy(board[cell])
                curr_domain = board[cell]
                row_cells_domain = self.row_cells(cell)
                intersection_domain = curr_domain.intersection(row_cells_domain)
                difference = curr_domain.difference(intersection_domain)
                if len(difference) == 1: # can make decision for cell
                    board[cell] = difference
                    self.board = board
                    self.infer_ac3()
                    flag = True
                    continue
                col_cells_domain = self.col_cells(cell)
                intersection_domain = curr_domain.intersection(col_cells_domain)
                difference = curr_domain.difference(intersection_domain)
                if len(difference) == 1: # can make decision for cell
                    board[cell] = difference
                    self.board = board
                    self.infer_ac3()
                    flag = True
                    continue
                block_cells_domain = self.block_cells(cell)
                intersection_domain = curr_domain.intersection(block_cells_domain)
                difference = curr_domain.difference(intersection_domain)
                if len(difference) == 1: # can make decision for cell
                    board[cell] = difference
                    self.board = board
                    self.infer_ac3()
                    flag = True
                    continue
    
    def select_unassigned_variable(self):
        board = self.board
        CELLS = self.CELLS
        ARCS = self.ARCS
        # minimum remaining values heuristic
        curr_lowest_remaining_values = np.inf
        # initialize the minimum remaining values to Inf
        chosen_cells = []
        for cell in self.CELLS:
            rem_values = len(board[cell])
            if rem_values == 1:
                continue
            if rem_values <= curr_lowest_remaining_values:
                curr_lowest_remaining_values = rem_values
                chosen_cells.append(cell)
        if len(chosen_cells) == 1: # no need for tie-breaking
            choice = chosen_cells[0]
            return choice # the unique cell choice for assigning to
        # maximum degree heuristic
        num_constraints = []
        for cell in chosen_cells:
            count = 0
            for arc in ARCS:
                if len(board[arc[1]]) == 1:
                    continue # partner variable is assigned
                if arc[0] == cell:
                    count += 1
            num_constraints.append(count)
        min_idx = np.argmin(num_constraints)
        choice = chosen_cells[min_idx]
        return choice
    
    def is_solved(self):
        board = self.board
        CELLS = self.CELLS
        for cell in CELLS:
            if len(board[cell]) != 1:
                return False
        return True

    def is_consistent(self):
        sudoku = copy.deepcopy(self)
        board = sudoku.board
        ARCS = sudoku.ARCS
        for arc in ARCS:
            cell_1 = arc[0]
            cell_2 = arc[1]
            if len(board[arc[0]]) == 1: # one cell is fixed
                # Does this make any of the constraints it participates in fail?
                # Check by ensuring that there is at least one value in the domain of the
                # partner cell.
                if not board[arc[1]].difference(board[arc[0]]):
                    return False
        return True
                    
    def backtracking_search(self):
        assignment = self.board
        if self.is_solved():
            return assignment
        cell = self.select_unassigned_variable()
        for value in assignment[cell]:
            sudoku = copy.deepcopy(self)
            sudoku.board[cell] = {value}
            if sudoku.is_consistent():
                sudoku.infer_improved()
                result = sudoku.backtracking_search()
                if result:
                    return result 

    def infer_with_guessing(self):
        self.infer_improved()
        self.board = self.backtracking_search()

############################################################
# Section 2: Dominoes Games
############################################################

def create_dominoes_game(rows, cols):
    board = [[False]*cols for _ in range(rows)]
    return DominoesGame(board)

class DominoesGame(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.optim_move = None # no optimal move known yet, run get_best_move() method

    def get_board(self):
        return self.board

    def reset(self):
        self.board = [[False]*self.cols for _ in range(self.rows)]

    def is_legal_move(self, row, col, vertical):
        # check for bounds violations, and pre-occupations 
        if vertical == True:
            intended_placement = [(row, col), (row + 1, col)]
            for square in intended_placement:
                new_row = square[0]
                new_col = square[1]
                if new_row < 0 or new_row >= self.rows:
                    return False
                if new_col < 0 or new_col >= self.cols:
                    return False
                if self.board[square[0]][square[1]] == True: # occupied
                    return False   
        if vertical == False:
            intended_placement = [(row, col), (row, col + 1)]
            for square in intended_placement:
                new_row = square[0]
                new_col = square[1]
                if new_row < 0 or new_row >= self.rows:
                    return False
                if new_col < 0 or new_col >= self.cols:
                    return False
                if self.board[square[0]][square[1]] == True: # occupied
                    return False  
        return True # otherwise
            
    def legal_moves(self, vertical):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_legal_move(row, col, vertical):
                    yield (row, col)

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical == True:
                intended_placement = [(row, col), (row + 1, col)]
                for square in intended_placement:
                    self.board[square[0]][square[1]] = True
            if vertical == False:
                intended_placement = [(row, col), (row, col + 1)]
                for square in intended_placement:
                    self.board[square[0]][square[1]] = True

    def game_over(self, vertical):
        if not list(self.legal_moves(vertical)):
            return True # game over since no legal moves available
        return False

    def copy(self):
        copy_game = copy.deepcopy(self)
        return copy_game

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            copy_game = self.copy()
            copy_game.perform_move(move[0], move[1], vertical)
            result_tuple = (move, copy_game)
            yield result_tuple

    def get_random_move(self, vertical):
        legal_moves = list(self.legal_moves(vertical))
        return random.choice(legal_moves)
    
    def evaluate(self, vertical):
        moves_curr_player = len(list(self.legal_moves(vertical)))
        moves_opponent = len(list(self.legal_moves(not vertical)))
        return moves_curr_player - moves_opponent
    
    def max_value(self, limit, alpha, beta, vertical, root, leaf_nodes):
        if limit == 0 or self.game_over(vertical):
            leaf_nodes += 1
            return self.evaluate(root), leaf_nodes
        v = -np.inf
        for move, successor_board in self.successors(vertical):
            new_value, leaf_nodes = successor_board.min_value(limit - 1, alpha, beta, 
                                                              not vertical, root, leaf_nodes)
            if new_value > v:
                v = new_value
                self.optim_move = move
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return v, leaf_nodes

    def min_value(self, limit, alpha, beta, vertical, root, leaf_nodes):
        if limit == 0 or self.game_over(vertical):
            leaf_nodes += 1
            return self.evaluate(root), leaf_nodes
        v = np.inf
        for move, successor_board in self.successors(vertical):
            new_value, leaf_nodes = successor_board.max_value(limit - 1, alpha, beta, 
                                                              not vertical, root, leaf_nodes)
            if new_value < v:
                v = new_value
                self.optim_move = move
            beta = min(beta, v)
            if alpha >= beta:
                break
        return v, leaf_nodes
    
    def get_best_move(self, vertical, limit):
        alpha = -np.inf
        beta = np.inf
        root = vertical
        leaf_nodes = 0 
        value, leaf_nodes = self.max_value(limit, alpha, beta, vertical, root, leaf_nodes)
        return self.optim_move, value, leaf_nodes
        
############################################################
# Section 3: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = 25

feedback_question_2 = """
Implementing the alpha-beta pruning was the most challenging aspect. 
It took me a while to realize that the goodness of a value is measured against the value for the player at the root.
"""

feedback_question_3 = """
I liked the detailed instructions for the Sudoku solver. 
The homeworks have been too long for my liking, but they do offer an opportunity to implement in practice the algorithms discussed in class.
"""
