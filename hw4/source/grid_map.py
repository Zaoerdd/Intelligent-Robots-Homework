import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

class grid_map:
    def __init__(self, map_matrix=None, reward_matrix=None, state_prob=1, start_index=(2, 2), goal_index=(16, 16)):
        self.map_matrix = map_matrix

        self.state_space = map_matrix.shape
        self.action_space = [ (1, 0), (-1, 0), (0, 1), (0, -1) ]
        self.state_prob = state_prob
        self.reward_matrix = reward_matrix
        self.reward_bound = -5

        self.start_index = start_index
        self.goal_index = goal_index

    def step(self, cur_state_index, action_index):
        if cur_state_index[0] == self.goal_index[0] and cur_state_index[1] == self.goal_index[1]:
            return cur_state_index, 0, self.state_prob, True

        action = self.action_space[action_index]
        done = False

        next_x = cur_state_index[0] + action[0]
        next_y = cur_state_index[1] + action[1]

        if next_x > self.state_space[0] - 1:
            next_x = self.state_space[0] - 1
            reward = self.reward_bound
            done = False
        
        elif next_x < 0:
            next_x = 0
            reward = self.reward_bound
            done = False
        
        elif next_y > self.state_space[1] - 1:
            next_y = self.state_space[1] - 1
            reward = self.reward_bound 
            done = False

        elif next_y < 0:
            next_y = 0
            reward = self.reward_bound 
            done = False
        else:        
            reward = self.reward_matrix[next_x, next_y]
            done = False

        if next_x == self.goal_index[0] and next_y == self.goal_index[1]:
            done = True

        next_state = (next_x, next_y)

        return next_state, reward, self.state_prob, done

    def set_path(self, index):
        self.map_matrix[index[0], index[1], :] = 255

    def show_map(self):
        if plt is None:
            raise ModuleNotFoundError("matplotlib is required to display the grid map.")
        plt.imshow(self.map_matrix / 255 )
        plt.show()

    def draw_map(self, time=0.0001):
        if plt is None:
            raise ModuleNotFoundError("matplotlib is required to draw the grid map.")
        plt.imshow(self.map_matrix / 255)
        plt.pause(time) 
