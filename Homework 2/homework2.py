############################################################
# CIS 521: Homework 2
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import collections
import math
import random
import copy


############################################################
# Section 1: N-Queens
############################################################

def num_placements_all(n):
    queens = n
    squares = n**2
    # squares choose queens
    return int(math.factorial(squares)/(math.factorial(queens) * math.factorial(squares - queens)))

def num_placements_one_per_row(n):
    queens = n
    # squares choose queens
    return (queens**queens)

# def n_queens_valid(board):
#     # ROW CHECK 
#     # is implemented by virtue of 
#     # the fact that the i-th index cannot have
#     # two values
#     # COLUMN CHECK
#     if len(set(board)) != len(board):
#         # if condition checks for duplicate entries in board,
#         # if any value appear more than once, that implies that
#         # multiple queens share the same columns
#         return False 
#     # DIAGONAL CHECK
#     n = max(board)
#     queens_exist = []
#     for i, col in enumerate(board):
#         queens_exist.append((i, col))
#     # diagonal elements going down
#     for queen in queens_exist:
#         curr_queen_row = queen[0]
#         curr_queen_col = queen[1]
#         diags_down = []
#         within = True # within bounds of board
#         new_row = curr_queen_row
#         new_col = curr_queen_col
#         while within:
#             new_row += 1
#             new_col += 1
#             if (new_row > n) or (new_col > n):
#                 within = False
#             else:
#                 diags_down.append((new_row, new_col))
#         if [queen_exists for queen_exists in diags_down if queen_exists in queens_exist]:
#             return False # a queen exists diagonally downwards from current queen     
#         diags_up = []
#         within = True # within bounds of board
#         new_row = curr_queen_row
#         new_col = curr_queen_col
#         while within:
#             new_row -= 1
#             new_col += 1
#             if (new_row > n) or (new_col > n):
#                 within = False
#             else:
#                 diags_up.append((new_row, new_col))
#         if [queen_exists for queen_exists in diags_up if queen_exists in queens_exist]:
#             return False # a queen exists diagonally upwards from current queen        
#     return True # if no fail condition was tripped

def n_queens_valid(board):
    if len(set(board)) != len(board):
        return False
    for row_1, col_1 in enumerate(board):
        for row_2, col_2 in enumerate(board):
            if row_1 != row_2:
                if col_1 == col_2: # same queen in two columns
                    return False
                if abs(row_1 - row_2) == abs(col_1 - col_2):
                    return False
    return True

def n_queens_helper(n, board):
    all_valid = set(range(n)) - set(board)
    count = len(all_valid)
    while count > 0:
        copy_board = board.copy()
        elem = all_valid.pop()
        count -= 1
        copy_board.append(elem)
        if n_queens_valid(copy_board):
            yield elem
    
def solve_n_queens(n, row, board, result):
    if row == n:
        yield board.copy()
    else:
        for col in range(n):
            board.append(col)
            if n_queens_valid(board):
                yield from solve_n_queens(n, row + 1, board, result)
            board.pop()
        
def n_queens_solutions(n):
    board = []
    result = []
    row = 0
    return solve_n_queens(n, row, board, result)

############################################################
# Section 2: Lights Out
############################################################

def create_puzzle(rows, cols):
    board = [[False]*cols for _ in range(rows)]
    puzzle = LightsOutPuzzle(board)
    return puzzle
    
class LightsOutPuzzle(object):
    
    def __init__(self, board):
        # initialize 2-D board
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.board = board
    
    def get_board(self):
        return self.board
    
    def perform_move(self, row, col):
        board = self.board
        if row >= 0 and row < self.num_rows and col >= 0 and col < self.num_cols:
            board[row][col] = not board[row][col]
        row_above = row - 1
        col_above = col 
        if row_above >= 0:
            board[row_above][col_above] = not board[row_above][col_above]
        row_below = row + 1
        col_below = col
        if row_below < self.num_rows:
            board[row_below][col_below] = not board[row_below][col_below]
        row_right = row
        col_right = col + 1
        if col_right < self.num_cols:
            board[row_right][col_right] = not board[row_right][col_right]
        row_left = row
        col_left = col - 1
        if col_left >= 0:
            board[row_left][col_left] = not board[row_left][col_left]
        self.board = board
        
    def scramble(self):
        new_self = self.copy()
        for row in range(new_self.num_rows):
            for col in range(new_self.num_cols):
                if random.random() < 0.5:
                    new_self.perform_move(row, col)
        return new_self
                    
    def is_solved(self):
        board = self.board
        need_falses = self.num_rows * self.num_cols
        count = 0
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if not board[row][col]:
                    count += 1
        if count == need_falses:
            return True
        return False
        
    def copy(self):
        copy_puzzle = copy.deepcopy(self)
        return copy_puzzle
    
    def successors(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                move = (row,col)
                temp_puzzle = self.copy()
                temp_puzzle.perform_move(row, col)
                result_tuple = (move, temp_puzzle)
                yield result_tuple
                         
    def find_solution(self):
        node = self
        parent = {}
        moves = ()
        if node.is_solved():
            return moves
        frontier = [node] 
        explored = set()
        while True:
            if not frontier: # frontier empty
                return None
            node = frontier.pop(0) # remove from frontier
            explored.add(tuple(tuple(x) for x in node.board)) # add to explored
            for move, successor in node.successors():
                candidate_child = tuple(tuple(x) for x in successor.board)
                if candidate_child in explored:
                    continue
                if candidate_child in frontier:
                    continue
                parent[tuple(tuple(x) for x in successor.board)] = (move) 
                # this `move' links `parent'-`child'
                if successor.is_solved():
                    # use parent dictionary
                    while successor.board != self.board:
                        moves = (parent[tuple(tuple(x) for x in successor.board)],) + moves 
                        successor.perform_move(moves[0][0], moves[0][1])
                    return list(moves)
                frontier.append(successor)

############################################################
# Section 3: Linear Disk Movement
############################################################

def create_game(length, n):
    state = [-1]*length
    for i in range(n):
        state[i] = i
    game = Game(state)
    return game

class Game(object):
    
    def __init__(self, state):
        self.num_disks = sum(elem is not -1 for elem in state)
        self.length = len(state)
        self.state = state
    
    def get_state(self):
        return self.state
            
    def perform_move(self, loc, jump):
        # jump could be -1, -2, +1, +2
        state = self.state
        disk = state[loc]
        state[loc] = -1
        state[loc + jump] = disk
        self.state = state
    
    def is_solved_identical(self):
        state = self.state
        n = self.num_disks
        check_list = state[-n:]
        if any(elem == -1 for elem in check_list):
            return False
        return True
    
    def is_solved_distinct(self):
        state = self.state
        n = self.num_disks
        length = self.length
        check_list = list(reversed(range(n)))
        while len(check_list) != length:
            check_list.insert(0,-1)
        if state == check_list:
            return True
        else:
            return False
    
    def copy(self):
        copy_game = copy.deepcopy(self)
        return copy_game
    
    def successors(self):
        jumps = [-2, -1, 1, 2]
        for i in range(self.length):
            for jump in jumps:
                temp_game = self.copy()
                if valid_move(i, jump, temp_game):
                    temp_game.perform_move(i, jump)
                    result_tuple = ((i, i + jump),temp_game)
                    yield result_tuple

def valid_move(i, jump, temp_game):
    num_disks = temp_game.num_disks
    length = temp_game.length
    state = temp_game.state
    if jump == -2:
        # ensure there is a disk to leap over
        middle = i + jump + 1
        if middle not in range(length):
            return False
        if state[middle] == -1:
            return False
    if jump == 2:
        # ensure there is a disk to leap over
        middle = i + jump - 1
        if middle not in range(length):
            return False
        if state[middle] == -1:
            return False
    to = i + jump
    if (to < 0) or (to >= length):
        return False
    if state[i] == -1:
        return False
    if state[to] != -1:
        return False
    return True
                    
def solve_identical_disks(length, n):
    g = create_game(length,n)
    node = g.copy()
    parent = {}
    moves = ()
    if node.is_solved_identical():
        return list(moves)
    frontier = collections.deque()
    frontier.append(node) 
    explored = set()
    while True:
        if not frontier: # frontier empty
            return None
        node = frontier.popleft() # remove from frontier: BFS
        explored.add(tuple(node.state)) # add to explored
        for move, successor in node.successors():
            if tuple(successor.state) in explored:
                continue
            if successor in frontier:
                continue
            parent[tuple(successor.state)] = (move) 
            # this `move' links `parent'-`child'
            if successor.is_solved_identical():
                # use parent dictionary
                while successor.state != g.state:
                    moves = (parent[tuple(successor.state)],) + moves 
                    # going backwards, the parent's jump position becomes
                    # child's position, and the jump for the child is the difference
                    # between the parent's initial position and the child's current
                    # position
                    successor.perform_move(moves[0][1], moves[0][0] - moves[0][1])
                return list(moves)
            frontier.append(successor)
            
def solve_distinct_disks(length, n):
    g = create_game(length,n)
    node = g.copy()
    parent = {}
    moves = ()
    if node.is_solved_identical():
        return list(moves)
    frontier = collections.deque() 
    frontier.append(node)
    explored = set()
    while True:
        if not frontier: # frontier empty
            return None
        node = frontier.popleft() # remove from frontier: BFS
        explored.add(tuple(node.state)) # add to explored
        for move, successor in node.successors():
            if tuple(successor.state) in explored:
                continue
            if successor in frontier:
                continue
            parent[tuple(successor.state)] = (move) 
            # this `move' links `parent'-`child'
            if successor.is_solved_distinct():
                # use parent dictionary
                while successor.state != g.state:
                    moves = (parent[tuple(successor.state)],) + moves 
                    # going backwards, the parent's jump position becomes
                    # child's position, and the jump for the child is the difference
                    # between the parent's initial position and the child's current
                    # position
                    successor.perform_move(moves[0][1], moves[0][0] - moves[0][1])
                return list(moves)
            frontier.append(successor)

############################################################
# Section 4: Feedback
############################################################

feedback_question_1 = """
Approximately between 40 and 50 hours.
"""

feedback_question_2 = """
Translating the pseudocode for the search algorithms was tricky for me.
Not having a strong background in data-structures made it harder to quickly
understand what was meant by queues and stacks and popping, etc.
"""

feedback_question_3 = """
I enjoyed the programming rigor. I have had to learn a lot of Python very 
very quickly for this assignment, which is always a plus. I would have loved for this
assignment to be shorter though, perhaps without the solve_distinct_disks() 
implementation.
"""
