############################################################
# CIS 521: Homework 3
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import copy
import random
import numpy as np
import queue
from collections import defaultdict


############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    raw_list = list(range(rows * cols))
    raw_list.pop(0) # remove from the front
    raw_list.append(0) # add to the end
    board = [raw_list[i:i+cols] for i in range(0, len(raw_list), cols)]
    # reshape list of numbers to be a list of lists of numbers
    return TilePuzzle(board)

class TilePuzzle(object):
    
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        zero_tuple = [(i, row.index(0))
                         for i, row in enumerate(board)
                         if 0 in row] # find where the empty tile is
        self.zero_row = [i[0] for i in zero_tuple][0]
        self.zero_col = [i[1] for i in zero_tuple][0]
        
    def __lt__(self, other): # needed for PriorityQueue to work
        return self.heuristic() < other.heuristic()
        
    def get_board(self):
        return self.board
    
    def perform_move(self, direction):
        board = self.board
        zero_row = self.zero_row
        zero_col = self.zero_col
        if direction == "up": # check top bounds
            if (zero_row - 1) >= 0:
                replace_with = board[zero_row - 1][zero_col]
                board[zero_row][zero_col] = replace_with
                board[zero_row - 1][zero_col] = 0
                self.board = board
                self.zero_row = zero_row - 1
                self.zero_col = zero_col
                return True
            else:
                return False
        if direction == "down": # check bottom bounds
            if (zero_row + 1) < self.rows:
                replace_with = board[zero_row + 1][zero_col]
                board[zero_row][zero_col] = replace_with
                board[zero_row + 1][zero_col] = 0
                self.board = board
                self.zero_row = zero_row + 1
                self.zero_col = zero_col
                return True
            else:
                return False
        if direction == "left": # check left bounds
            if (zero_col - 1) >= 0:
                replace_with = board[zero_row][zero_col - 1]
                board[zero_row][zero_col] = replace_with
                board[zero_row][zero_col - 1] = 0
                self.board = board
                self.zero_row = zero_row
                self.zero_col = zero_col - 1
                return True
            else:
                return False
        if direction == "right": # check right bounds
            if (zero_col + 1) < self.cols:
                replace_with = board[zero_row][zero_col + 1]
                board[zero_row][zero_col] = replace_with
                board[zero_row][zero_col + 1] = 0
                self.board = board
                self.zero_row = zero_row
                self.zero_col = zero_col + 1
                return True
            else:
                return False

    def scramble(self, num_moves):
        # shuffle the board from its initial state, which is solved
        moves = ["up", "down", "left", "right"]
        moves_tried = 0
        while moves_tried != num_moves:
            move = random.choice(moves)
            self.perform_move(move)
            moves_tried += 1
            
    def is_solved(self):
        rows = self.rows
        cols = self.cols
        solved_list = list(range(rows * cols))
        solved_list.pop(0)
        solved_list.append(0)
        flat_board = [tile for row in self.board for tile in row]
        # compare current board to what it ought to look like
        if flat_board == solved_list:
            return True
        return False
            
    def copy(self):
        copy_puzzle = copy.deepcopy(self)
        return copy_puzzle
    
    def successors(self):
        moves = ["up", "down", "left", "right"]
        for move in moves:
            copy_self = self.copy()
            if copy_self.perform_move(move):
                result_tuple = (move, copy_self)
                yield result_tuple
                
    def iddfs_helper(self, limit, moves): 
        # base case is when the length of moves equals the depth limit
        # here, either there is a solution, or there is not
        # if there is, then yield it, else do nothing (equal to `cutoff')
        if len(moves) == limit:
            if self.is_solved():
                yield moves
        else:
            for move, successor in self.successors():
                copy_moves = copy.deepcopy(moves)
                copy_moves.append(move)
                yield from successor.iddfs_helper(limit, copy_moves)
                    
    def find_solutions_iddfs(self):
        limit = 0
        solutions = []
        flag = True
        while flag == True: # depth increases until there is a solution
            solutions = list(self.iddfs_helper(limit, solutions))
            if solutions: # solutions are not empty implying depth is good
                for solution in solutions: # to cast the function as a generator
                    yield solution
                flag = False
            else:
                limit += 1 # no solutions found, increase depth and repeat
                
    def heuristic(self):
        manhattan_dist = 0
        num_rows = self.rows
        num_cols = self.cols
        goal_puzzle = create_tile_puzzle(num_rows, num_cols)
        goal_state = goal_puzzle.board # what the board should look like
        for i in range(num_rows):
            for j in range(num_cols):
                if self.board[i][j] != 0:
                    elem = self.board[i][j] # what is the elem
                    elem_row = i # what row is it in
                    elem_col = j # what col is it in
                    index_tuple = [(i, row.index(elem))
                                   for i, row in enumerate(goal_state)
                                   if elem in row]
                    des_row = index_tuple[0][0] # what row should it be in
                    des_col = index_tuple[0][1] # what col should it be in
                    manhattan_dist += (abs(des_row - elem_row) + abs(des_col - elem_col))
        return manhattan_dist
                                   
    def find_solution_a_star(self):
        start = self 
        frontier = queue.PriorityQueue() 
        frontier.put((start.heuristic(), 0, self, ()))
        explored = set()
        while frontier:
            released_tuple = frontier.get() 
            node = released_tuple[2] # gets TilePuzzle object
            moves = released_tuple[3]
            if node.is_solved():
                return list(moves)
            if tuple(tuple(x) for x in node.board) in explored:
                continue
            else:
                explored.add(tuple(tuple(x) for x in node.board)) 
                f_score_curr = released_tuple[0]
                g_score_curr = released_tuple[1]
                for move, successor in node.successors():
                    candidate_child = tuple(tuple(x) for x in successor.board)
                    successor_g_score = g_score_curr + 1
                    successor_f_score = successor_g_score + successor.heuristic()
                    if candidate_child not in explored:
                        frontier.put((successor_f_score, successor_g_score, successor, moves + (move,)))

############################################################
# Section 2: Grid Navigation
############################################################

class GridNavigation(object):
    
    def __init__(self, curr_pos, goal_pos, scene):
        self.num_rows = len(scene)
        self.num_cols = len(scene[0])
        self.scene = scene
        self.curr_pos = curr_pos
        self.goal_pos = goal_pos
        
    def __lt__(self, other):
        return self.heuristic() < other.heuristic()
        
    def get_scene(self):
        return self.scene

    def perform_move(self, dir_tuple):
        curr_row = self.curr_pos[0]
        curr_col = self.curr_pos[1]
        poss_new_row = curr_row + dir_tuple[0]
        poss_new_col = curr_col + dir_tuple[1]
        if (poss_new_row >= 0) and (poss_new_row < self.num_rows):
            if (poss_new_col >= 0) and (poss_new_col < self.num_cols):
                if self.scene[poss_new_row][poss_new_col] == False:
                    self.curr_pos = (poss_new_row, poss_new_col)
                    return True
        return False
            
    def is_solved(self):
        if self.curr_pos == self.goal_pos:
            return True
        return False
            
    def copy(self): # by not copying the scene over and over, makes search go quicker
        return GridNavigation(copy.deepcopy(self.curr_pos), self.goal_pos, self.scene)
    
    def successors(self):
        directions = {"up":(-1,0),
                      "down":(1,0),
                      "left":(0,-1),
                      "right":(0,1),
                      "up-left":(-1,-1), 
                      "up-right":(-1,1),
                      "down-left":(1,-1),
                      "down-right":(1,1)}
        for direction in directions.keys():
            temp_grid = self.copy()
            if temp_grid.perform_move(directions[direction]):
                result_tuple = (direction, temp_grid)
                yield result_tuple
                
    def heuristic(self):
        curr_row = self.curr_pos[0]
        curr_col = self.curr_pos[1]
        goal_row = self.goal_pos[0]
        goal_col = self.goal_pos[1]
        distance = np.sqrt((goal_row - curr_row)**2 + (goal_col - curr_col)**2)
        return distance
    
    def find_solution_a_star(self):
        start = self 
        if start.scene[start.curr_pos[0]][start.curr_pos[1]]:
            # starting at an obstacle
            return None    
        frontier = queue.PriorityQueue() 
        frontier.put((start.heuristic(), 0, start, (start.curr_pos,)))
        explored = set()
        while not frontier.empty():
            released_tuple = frontier.get() 
            node = released_tuple[2] # gets GridNavigation object
            moves = released_tuple[3]
            if node.is_solved():
                return list(moves) 
            if node.curr_pos in explored:
                continue
            else:
                explored.add(node.curr_pos)
                g_score_curr = released_tuple[1]
                for move, successor in node.successors():
                    successor_pos = successor.curr_pos
                    if move in {"up", "down", "left", "right"}:
                        successor_g_score = g_score_curr + 1
                    else:
                        successor_g_score = g_score_curr + np.sqrt(2)
                    successor_f_score = successor_g_score + successor.heuristic()
                    if successor_pos not in explored:
                        frontier.put((successor_f_score, successor_g_score, successor, moves + (successor_pos,)))
        return None
                    
def find_path(curr_pos, goal_pos, scene):
    grid = GridNavigation(curr_pos, goal_pos, scene)
    return grid.find_solution_a_star()

############################################################
# Section 3: Linear Disk Movement, Revisited
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
        
    def __lt__(self, other):
        return self.heuristic() < other.heuristic()
    
    def get_state(self):
        return self.state
            
    def perform_move(self, loc, jump):
        # jump could be -1, -2, +1, +2
        state = self.state
        disk = state[loc]
        state[loc] = -1
        state[loc + jump] = disk
        self.state = state
        
    def is_solved_distinct(self):
        state = self.state
        n = self.num_disks
        length = self.length
        check_list = list(reversed(range(n)))
        while len(check_list) != length:
            check_list.insert(0,-1)
        # compare the current state to the solved state
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
                    
    def heuristic(self):
        state = self.state
        n = self.num_disks
        length = self.length
        heuristic = 0
        goal_state = list(reversed(range(n)))
        while len(goal_state) != length:
            goal_state.insert(0,-1)
        # game state should look like goal state
        # heuristic captures how dissimilar they are
        for disk in range(n):
            curr_idx = state.index(disk)
            goal_idx = goal_state.index(disk)
            heuristic += np.abs(curr_idx - goal_idx)
        return heuristic
        
    def find_solution_a_star(self):
        start = self
        moves = ()
        if start.is_solved_distinct():
            return list(moves)
        frontier = queue.PriorityQueue() 
        frontier.put((start.heuristic(), 0, start, moves))
        explored = set()
        while True:
            if frontier.empty():
                return None
            released_tuple = frontier.get()
            node = released_tuple[2] # Game object
            moves = released_tuple[3] # moves
            if node.is_solved_distinct():
                return list(moves)
            if tuple(node.state) in explored:
                continue
            else:
                explored.add(tuple(node.state))
                f_score_curr = released_tuple[0]
                g_score_curr = released_tuple[1]
                for move, successor in node.successors():
                    candidate_child = tuple(successor.state)
                    successor_g_score = g_score_curr + 1
                    successor_f_score = successor_g_score + successor.heuristic()
                    if candidate_child not in explored:
                        frontier.put((successor_f_score, successor_g_score, successor, moves + (move,)))    

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
 
def solve_distinct_disks(length, n):
    g = create_game(length,n)
    return g.find_solution_a_star()

############################################################
# Section 4: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = 25

feedback_question_2 = """
Managing additions to the frontier for the A* Algorithm was tricky.
Most tests for A* also kept timing out, which was the hardest thing to debug.
"""

feedback_question_3 = """
It would have been nice to merely have to implement the algorithms correctly, 
instead of also having to optimize them for performance. 
"""
