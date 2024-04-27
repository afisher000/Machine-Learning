"""
Created on Fri Dec  8 17:30:33 2023

@author: afisher
"""

import random
import numpy as np
import time
import pickle

class Game2048():
    def __init__(self, N=4, testing=False, strategy='combine', max_depth=3, parameters=None, max_turns=1000):
        self.N = N
        self.nsample = nsample
        self.strategy = strategy
        self.max_depth = max_depth
        self.parameters = parameters
        self.max_turns = max_turns
        self.move_lookup = pickle.load(open('move_lookup.pkl', 'rb'))
        
        
        
        self.simulate_game()
        
    def simulate_Ngames(self, N):
        scores = [g.simulate_game() for _ in range(N)]
        return np.median(scores)
        
    def simulate_game(self):
        # Setup board
        self.initialize_board()

        # Loop over turns
        self.turns = 1

        # print(f'Turn {self.turns}')
        score, direction = self.recursive_search()

        
        
        #While at least one valid board, keep playing
        while score>0 and self.turns<self.max_turns: 
            print(self.board)
            print()
            self.turns += 1
            # print(f'Turn {self.turns}')
            
            # Make move, add random tile
            self.board = self.move(direction)
            self.place_random_tile()
                
            # Decide next direction
            score, direction = self.recursive_search()
            time.sleep(1)
        # print(f'Max tile was {np.max(self.board)}')
        

        return self.turns
    
                    
    def recursive_search(self, board=None, depth=0):
        if board is None:
            board = self.board
            
        # print(f'Depth = {depth}, board is')
        
        max_score = 0
        opt_direction = 'up'
        
        for direction in ['up', 'left', 'right', 'down']:
            # print(f'\tDirection={direction}')
            is_valid, new_board = self.is_valid_move(direction, board)
            
            if is_valid:
                
                # Max depth, evaluate leaf node
                if (depth+1)==self.max_depth:
                    score = self.score_board(new_board)
                    
                # Else, search branch
                else:
                    
                    # Find minimum of sampled space
                    if new_board[0,0]==0:
                        new_board[0,0] == 1
                    else:
                        self.place_random_tile(new_board)
                    score, _ = self.recursive_search(new_board, depth+1)
                
                # Score
                if score>max_score:
                    max_score = score
                    opt_direction = direction
                    
            else:
                # print('\t\tInvalid move')
                pass
            
        # If no valid moves, but not at root, return score for current board. 
        if max_score==0 and depth>0:
            max_score = self.score_board(board)
        
        return max_score, opt_direction
    
    def is_valid_move(self, direction, board=None):
        # Check if move is valid. Return newboard.
        
        if board is None:
            board = self.board
            
        new_board = self.move(direction, board)
        is_valid = not np.array_equal(board, new_board)
        
        return is_valid, new_board
    
    def initialize_board(self):
        self.board = np.zeros( (self.N, self.N), dtype=int)
        self.place_random_tile()
        self.place_random_tile()
        return 
    
    
    def move(self, direction, board=None):
        # Creates copy. Move does not affect input board.
        
        if board is None:
            board_copy = self.board.copy()
        else:
            board_copy = board.copy()
            
        
        if direction == 'left':
            for i in range(self.N):
                board_copy[i] = self.merge(board_copy[i])
                
        elif direction == 'right':
            for i in range(self.N):
                board_copy[i] = self.merge(board_copy[i][::-1])[::-1]
                
        elif direction == 'up':
            for i in range(self.N):
                board_copy[:, i] = self.merge(board_copy[:, i])
                    
        elif direction == 'down':
            for i in range(self.N):
                board_copy[:, i] = self.merge(board_copy[:, i][::-1])[::-1]
                
        return board_copy
    
    
    def place_random_tile(self, board=None):
        if board is None:
            rows, cols = np.where(self.board==0)
        else:
            rows, cols = np.where(board==0)
        
        i, j = random.choice( list(zip(rows, cols)) )
        value = 1 if random.random()<0.9 else 2
        
        if board is None:
            self.board[i,j] = value
        else:
            board[i,j] = value
        return


    def merge_new(self, line):
        return np.array(self.move_lookup[tuple(line)], dtype=int)
        
        
    def merge(self, line):
        result = [0] * len(line)
        index = 0
        for value in line:
            if value != 0:
                if result[index] == 0:
                    result[index] = value
                elif result[index] == value:
                    result[index] += 1 #increment by 1 on logboard
                    index += 1
                else:
                    index += 1
                    result[index] = value
        return result
    

    
    
    def score_board(self, board):
        # Scale to 1,2,3...
        
        # Scores must be greater than 0. 
        if self.strategy == 'simple':
            return 1
        
        elif self.strategy == 'combine':  
            return np.sum(board==0)
        
        elif self.strategy == 'weighted':
            
            zeros = np.sum(board==0)
            filled_0 = int(board[0,0]>2)
            filled_1 = int(board[0,1]>2)
            filled_2 = int(board[0,2]>0)
            # filled_3 = int(board[0,3]>0)
            
            topdiff = np.diff(board[0])
            top_order = -1*topdiff[topdiff<0].sum()
            top_sum = board[0].sum()
            
            weights = np.array([
                [zeros, 1],
                [filled_0, 100], 
                [filled_1, 0],
                [filled_2, 0],
                [top_sum, 0],
                ])
            
            return np.sum( np.prod(weights, axis=1) )
            
            
            
        else:
            raise ValueError(f'Strategy {self.strategy} not recognized')
        

    
    
if __name__=="__main__":
    g = Game2048(strategy='weighted', max_depth=2)
    
    # Train scoring function to match my style of play. Can maybe utilize expectimax brute force searching in that case?
    
    # # Results
    # Simple:   214 (54)
    # Combine:  262 (64)
    # Combine (x2) 450 (80)
    # Combine (x3) 530 (120)
    # 
    
    # Simulates 10 turns in 5.6 ms.
    
    