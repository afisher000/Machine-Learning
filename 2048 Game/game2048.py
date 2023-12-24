# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:30:33 2023

@author: afisher
"""

import random
import numpy as np
import time

class Game2048():
    def __init__(self, N=4, printout=False, strategy='simple', max_depth=4):
        self.N = N
        self.printout = printout
        self.strategy = strategy
        self.max_depth = max_depth
        self.simulate_game()
        return
        
    def simulate_game(self):
        # Setup board
        self.board = self.initialize_board()
        self.place_random_tile()
        self.place_random_tile()

        # Loop over turns
        self.turns = 0
        direction = self.decide_direction();
        while direction!='gameover':
            self.board = self.move(direction)
            self.place_random_tile()
            
            # Printout for testing
            if self.printout and self.turns<50:
                print(direction)
                self.print_board()
                
            direction = self.decide_direction();
            self.turns += 1
                    
        print(f'Max tile was {np.max(self.board)}')
        return self.turns
    
    def decide_direction(self):
        max_score = 0
        opt_direction = 'up'
        for direction in ['up', 'left', 'right', 'down']:
            is_valid, new_board = self.is_valid_move(direction)
            if is_valid:
                # new_board[new_board==0] = 1 #lock random with ones
                move_score = self.recursive_search(new_board)
                
                if move_score>max_score:
                    max_score = move_score
                    opt_direction = direction
        
        if max_score==0:
            return 'gameover'
        else:
            return opt_direction
                    
    def recursive_search(self, board, depth=1):
        max_score = 0
        for direction in ['up', 'left', 'right', 'down']:
            new_board = self.move(direction, board)
            if (depth+1)==self.max_depth:
                score = self.score_board(new_board)
            else:
                score = self.recursive_search(new_board, depth+1)
            
            if score>max_score:
                max_score = score
        
        return max_score
    
    
    def score_board(self, board):
        # Scale to 1,2,3...
        logboard = board.copy()
        logboard[logboard==0] = 1
        logboard = np.log2(logboard)

        # Scores must be greater than 0. 
        if self.strategy == 'simple':
            return 1
        
        elif self.strategy == 'combine':  
            return np.sum(logboard==0)
        
        elif self.strategy == 'weighted':
            
            zeros = np.sum(board==0)
            filled_0 = int(logboard[0,0]>0)
            filled_1 = int(logboard[0,1]>0)
            filled_2 = int(logboard[0,2]>0)
            filled_3 = int(logboard[0,3]>0)
            
            topdiff = np.diff(logboard[0])
            top_order = -1*topdiff[topdiff<0].sum()
            
            top_sum = logboard[0].sum()
            
            weights = np.array([
                [filled_0, 1], 
                [filled_1, 1],
                [filled_2, 1],
                [filled_3, 1],
                [top_order, 1],
                [top_sum, 1],
                ])
            
            return np.sum( np.prod(weights, axis=1) )
            
            
            
        else:
            raise ValueError(f'Strategy {self.strategy} not recognized')
        
        
        
    def final_score(self):
        final_score = 0
        values, counts = np.unique(self.board, True)
        for value, count in zip(values, counts):
            final_score += count * value * (np.log2(value)-1)
        return final_score
    
    def initialize_board(self):
        return np.zeros( (self.N, self.N), dtype=int)
    
    def print_board(self):
        for row in self.board:
            print(" ".join("{:5}".format(cell) for cell in row))
        print()

    
    def place_random_tile(self):
        rows, cols = np.where(self.board==0)
        
        i, j = random.choice( list(zip(rows, cols)) )
        value = 2 if random.random()<0.9 else 4
        self.board[i,j] = value
        return
    
    def is_valid_move(self, direction):
        new_board = self.move(direction)
        is_valid = not np.array_equal(self.board, new_board)
        return is_valid, new_board
            
        
        
    def move(self, direction, board=None):
        # Use self.board by default
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
    
    def merge(self, line):
        result = [0] * len(line)
        index = 0
        for value in line:
            if value != 0:
                if result[index] == 0:
                    result[index] = value
                elif result[index] == value and value!=1:
                    result[index] *= 2
                    index += 1
                else:
                    index += 1
                    result[index] = value
        return result
    
    
    
if __name__=="__main__":
    g = Game2048(printout=True, strategy='weighted')
    
    # Print statistics
    if False:
        scores = [g.simulate_game() for _ in range(10)]
        median = np.median(scores)
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        print(f'{g.strategy}: Score = {median} +/- {(q3-q1)/2}')
        
        
    # # Results
    # Simple:   214 (54)
    # Combine:  262 (64)
    # Combine (x2) 450 (80)
    # Combine (x3) 530 (120)
    # 
    
    