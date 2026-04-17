import numpy as np
import queue
from math import sqrt

class Astar:
    def __init__(self):
        # graph node: x, y, cost, parent
        self.cur_vis = []   # current visited node list
        self.next_vis = []  # next visit node list

    def find_path(self, graph, start_pos, goal_pos):
        
        '''
        graph: the instance of the class defined in the grid_graph file
        start_pos: The start node consist of four parts: (x, y, cost, parent node)
        goal_pos: The goal node consist of four parts: (x, y, cost, parent node) 
        '''

        print('start to Astar search') 

        self.cur_vis = []
        self.next_vis = []

        frontier = queue.PriorityQueue()  # priority queue for algorithm to explore current points.
        frontier.put((0, 0, start_pos))  # put the priority and position

        self.next_vis.append([start_pos.x, start_pos.y])
        
        cost_so_far = np.full((graph.width, graph.height), np.inf) # cost matrix from start point to this point
        cost_so_far[start_pos.x, start_pos.y] = 0
        final_node = None
        expand_count = 0
        cur_vis_set = set()
        next_vis_set = {(start_pos.x, start_pos.y)}

        while not frontier.empty():
            # please complete this part for the homework question1 
            # each node has four parts: x position, y position, the cost so far, the parent node. Utilize the parent node, the path can be generated

            # parts: (1) get current node with priority (using frontier.get()[1])  
            # (2) check whether current node is the goal node (using graph.node_equal) 
            # (3) explore the neighbors of current node (using graph.neighbors)
            # (4) if the neighbor node is not in the next_vis (and cur_vis): put that in the frontier with priority and append that in the next_vis.  (priority= cost_so_far + heuristic)
            # (5) if the neighbor node is in the next_vis: check whether cost so far is less than that in the next_vis to determine whether put it in the frontier
            # (6) return the node when it is in the goal position.
            _, _, current = frontier.get()
            current_key = (current.x, current.y)

            if current_key in cur_vis_set:
                continue

            cur_vis_set.add(current_key)
            self.cur_vis.append([current.x, current.y])

            if graph.node_equal(current, goal_pos):
                final_node = current
                break

            for neighbor in graph.neighbors(current):
                step_cost = sqrt((neighbor.x - current.x) ** 2 + (neighbor.y - current.y) ** 2)
                new_cost = cost_so_far[current.x, current.y] + step_cost

                if new_cost < cost_so_far[neighbor.x, neighbor.y]:
                    cost_so_far[neighbor.x, neighbor.y] = new_cost
                    new_node = graph.node_tuple(neighbor.x, neighbor.y, new_cost, current)
                    priority = new_cost + self.heuristic(new_node, goal_pos)

                    expand_count += 1
                    frontier.put((priority, expand_count, new_node))

                    if (neighbor.x, neighbor.y) not in next_vis_set:
                        next_vis_set.add((neighbor.x, neighbor.y))
                        self.next_vis.append([neighbor.x, neighbor.y])

        print('search done')

        return final_node, self.cur_vis

    def heuristic(self, node1, node2, coefficient=1):
        # please complete the heuristic function for the homework question1  (related to the distance to the goal)
        return coefficient * sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
        

    def generate_path(self, final_node):
        # utilize the node to generate the path. 

        path = [ [final_node.x, final_node.y] ]
        
        while final_node.parent is not None:
            path.append( [final_node.parent.x, final_node.parent.y] )
            final_node = final_node.parent
        
        return path





    

