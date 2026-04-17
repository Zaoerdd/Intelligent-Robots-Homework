import numpy as np
from math import sqrt, inf


class dynamic_window_approach:
    def __init__(self, vx_range, vy_range, accelerate=0.5, time_interval=0.1, predict_time=1, graph=None):

        self.vx_range = vx_range
        self.vy_range = vy_range
        self.acce = accelerate
        self.time_int = time_interval
        self.graph = graph
        self.pre_time = predict_time
        
    def cal_vel(self, cur_pose, goal_pose, current_vel, v_gain=1, g_gain=1, o_gain=1, a_gain=1, astar_path=None):
        
        '''
        v_gain, g_gain, o_gain, a_gain: the gain for the velocity cost, goal cost, obstacle cost, astar cost, you should modify the gain for you own task.
        cur_pose: current pose with x, y; 2*1 vector
        goal_pose: goal pose with x, y; 2*1 vector
        current_vel: current velocity; 2*1 vector
        v_gain, g_gain=1, o_gain, a_gain: the scalar value
        '''
       
        cost_list = []
        vel_pair_list = []
        pre_traj_list = []

        cur_pose = np.array(cur_pose, dtype=float).reshape(2, 1)
        goal_pose = np.array(goal_pose, dtype=float).reshape(2, 1)
        current_vel = np.array(current_vel, dtype=float).reshape(2, 1)

        # Calculate the admissable velocity
        min_vx, max_vx, min_vy, max_vy = self.search_space(current_vel)
        vx_samples = np.arange(min_vx, max_vx + 1e-9, 0.1)
        vy_samples = np.arange(min_vy, max_vy + 1e-9, 0.1)

        if vx_samples.size == 0:
            vx_samples = np.array([min_vx])

        if vy_samples.size == 0:
            vy_samples = np.array([min_vy])

        for vx in vx_samples:
            for vy in vy_samples:
                pre_traj = self.predict_traj(cur_pose, vx, vy) # predict the trajectory under current velocity
                cost = self.cost_function(goal_pose, vx, vy, pre_traj, vel_cost_gain=v_gain, goal_cost_gain=g_gain, obstacle_cost_gain=o_gain)  # the object(cost) function for you to complete. you should complete this function for the homework question2

                if astar_path is not None:
                    astar_cost = self.astar_cost(pre_traj, astar_path) # complete the astar_cost function for question3
                    cost += a_gain * astar_cost

                cost_list.append(cost)
                vel_pair_list.append([vx, vy])
                pre_traj_list.append(pre_traj)

        min_cost_index = cost_list.index(min(cost_list))
        dwa_vel = vel_pair_list[min_cost_index]
        dwa_traj = pre_traj_list[min_cost_index]

        return np.c_[dwa_vel], dwa_traj

    def search_space(self, current_vel):
        cur_vx = float(current_vel[0, 0])
        cur_vy = float(current_vel[1, 0])

        min_vx = max(self.vx_range[0], cur_vx - self.time_int * self.acce)
        max_vx = min(self.vx_range[1], cur_vx + self.time_int * self.acce)

        min_vy = max(self.vy_range[0], cur_vy - self.time_int * self.acce)
        max_vy = min(self.vy_range[1], cur_vy + self.time_int * self.acce)

        return min_vx, max_vx, min_vy, max_vy

    def predict_traj(self, cur_pose, vx, vy):
        # predict the trajectory of current velocity
        
        pre_traj = []
        cur_vel = np.array( [ [vx], [vy] ] ) 
        cur_pose = np.array(cur_pose, dtype=float).reshape(2, 1)
        # print(cur_vel)
        i = 0
        while i < self.pre_time:
            next_pos = cur_pose + cur_vel * self.time_int
            i = i + self.time_int
            pre_traj.append(next_pos)
            cur_pose = next_pos
        
        return pre_traj

    def cost_function(self, goal_pose, vx, vy, pre_traj, vel_cost_gain=1, goal_cost_gain=1, obstacle_cost_gain=1):
        # Calculate the cost of current pair of vx vy
        # you should complete the function for question2. (HintL the summary cost of the following functions)
        # The cost include three parts: (1) the cost related to velocity, larger velocity is better, 10%
        # (2) cost realted to the goal, the closer position to the goal is better, 30%
        # (3) cost related to the obstacle, move away from the obstacle is better, 40%

        vel_cost = self.vel_cost(vx, vy)
        goal_cost = self.cost_to_goal(pre_traj, goal_pose)
        obstacle_cost = self.cost_to_obstacle(pre_traj)

        cost = vel_cost_gain * vel_cost + goal_cost_gain * goal_cost + obstacle_cost_gain * obstacle_cost
        return cost

    def vel_cost(self, vx, vy):
        # you should complete the function for question2
        # the cost function about the velocity cost (hint: maximize the velocity is better, you can use the norm of this velocity)
        speed = sqrt(vx ** 2 + vy ** 2)
        max_speed = sqrt(self.vx_range[1] ** 2 + self.vy_range[1] ** 2)
        return max_speed - speed

    def cost_to_goal(self, pre_traj, goal):
        # you should complete the function for question2
        # the closer position to the goal is better (hint: use the position of the final point in the pre_traj to judge. )
        final_pos = np.array(pre_traj[-1], dtype=float).reshape(2, 1)
        goal = np.array(goal, dtype=float).reshape(2, 1)
        return float(np.linalg.norm(final_pos - goal))
        

    def cost_to_obstacle(self, pre_traj):
        # you should complete the function for question2
        # the cost to the avoid the obstacles, move away from the obstacle is better 
        # (hint: the minimum distance between the predicted trajectory and obstacle in grid map, 
        # you can use the below function point_to_obstalce to calculate the distance with each point)
        min_distance = inf
        for point in pre_traj:
            distance = self.point_to_obstalce(point)

            if distance <= max(self.graph.xy_reso[0, 0], self.graph.xy_reso[1, 0]):
                return inf

            min_distance = min(min_distance, distance)

        return 1.0 / (min_distance + 1e-6)

        
    def point_to_obstalce(self, point):
        # the distance between current point and the obstacle depending on the grid map 

        max_x = self.graph.width * self.graph.xy_reso[0, 0]
        max_y = self.graph.height * self.graph.xy_reso[1, 0]

        if point[0, 0] < 0 or point[1, 0] < 0 or point[0, 0] >= max_x or point[1, 0] >= max_y:
            return 0.0

        index_x, index_y = self.graph.pose_to_index(point[0, 0], point[1, 0])

        if self.graph.grid_map[index_x, index_y] != 0:
            return 0.0

        temp_x = (self.graph.obstacle_index[0] - index_x) * self.graph.xy_reso[0, 0]
        temp_y = (self.graph.obstacle_index[1] - index_y) * self.graph.xy_reso[1, 0] 

        dis_list = [ sqrt(x**2 + y**2) for x, y in zip(temp_x, temp_y)]
        distance = min(dis_list)

        return distance

    def astar_cost(self, pre_traj, astar_path):
        # you should complete the function for question3
        # related to the distance between the positions in pre_traj and points in astar_path, 20%
        if astar_path is None or len(astar_path) == 0:
            return 0

        path_points = np.array([self.graph.index_to_pose(*index) for index in astar_path], dtype=float)
        distance_sum = 0.0

        for point in pre_traj:
            point_xy = np.array([point[0, 0], point[1, 0]], dtype=float)
            distances = np.linalg.norm(path_points - point_xy, axis=1)
            distance_sum += float(np.min(distances))

        return distance_sum / len(pre_traj)
        

        

        

        




