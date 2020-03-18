# Breadth First Search algorithm
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from matplotlib import cm

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

class A_star_Graph:
    def __init__(self, limit_x, limit_y, discr, cost_values, goal, start, goal_radius = 0.3, forward_length=1.65, rotation_cost_factor = 1.46, forward_cost_factor = 1, plot = False, all_actions = None, forward_actions = None):
        self.limit_x = limit_x
        self.limit_y = limit_y
        self.discr = discr
        self.cost = cost_values     
        self.goal = goal
        self.start = start
        self.n_cell = int(forward_length/discr + 1)
        self.goal_radius = goal_radius
        self.forward_cost_factor = forward_cost_factor
        self.rotation_cost_factor = rotation_cost_factor
        self.plot = plot
        #### Action Definition
        if all_actions is not None:
            self.all_actions = all_actions
        else:
            l = forward_length
            self.all_actions = {'s0': (l,None),
                            
                            'fm': (2/math.pi*l,l/(2/math.pi*l)), 'fm1': (-2/math.pi*l,l/(-2/math.pi*l)),
                            'f0': (2/math.pi*l+0.1,l/(2/math.pi*l+0.1)), 'f1': (-2/math.pi*l-0.1,l/(-2/math.pi*l-0.1)),
                            'f2': (2/math.pi*l+0.2,l/(2/math.pi*l+0.2)), 'f3': (-2/math.pi*l-0.2,l/(-2/math.pi*l-0.2)),
                            'f4': (2/math.pi*l+0.5,l/(2/math.pi*l+0.5)), 'f5': (-2/math.pi*l-0.5,l/(-2/math.pi*l-0.5)),
                            'f6': (2/math.pi*l+1.0,l/(2/math.pi*l+1.0)), 'f7': (-2/math.pi*l-1.0,l/(-2/math.pi*l-1.0)),
                            'f8': (2/math.pi*l+1.8,l/(2/math.pi*l+1.8)), 'f9': (-2/math.pi*l-1.8,l/(-2/math.pi*l-1.8)),
        
                            'r0': (None,20*math.pi/180), 'r1': (None,40*math.pi/180), 
                            'r2': (None,60*math.pi/180), 'r3': (None,80*math.pi/180),
                            'r4': (None,100*math.pi/180), 'r5': (None,120*math.pi/180), 
                            'r6': (None,140*math.pi/180), 'r7': (None,160*math.pi/180),
                            'r8': (None,180*math.pi/180), 'r9': (None,-20*math.pi/180), 
                            'r10': (None,-40*math.pi/180), 'r11': (None,-60*math.pi/180),
                            'r12': (None,-80*math.pi/180), 'r13': (None,-100*math.pi/180), 
                            'r14': (None,-120*math.pi/180), 'r15': (None,-140*math.pi/180), 'r16': (None,-160*math.pi/180)
                            }
        if forward_actions is not None:
            self.forward_actions = forward_actions
        else:
            self.forward_actions = {'s0': (l,None),
                            
                            'fm': (2/math.pi*l,l/(2/math.pi*l)), 'fm1': (-2/math.pi*l,l/(-2/math.pi*l)),
                            'f0': (2/math.pi*l+0.1,l/(2/math.pi*l+0.1)), 'f1': (-2/math.pi*l-0.1,l/(-2/math.pi*l-0.1)),
                            'f2': (2/math.pi*l+0.2,l/(2/math.pi*l+0.2)), 'f3': (-2/math.pi*l-0.2,l/(-2/math.pi*l-0.2)),
                            'f4': (2/math.pi*l+0.5,l/(2/math.pi*l+0.5)), 'f5': (-2/math.pi*l-0.5,l/(-2/math.pi*l-0.5)),
                            'f6': (2/math.pi*l+1.0,l/(2/math.pi*l+1.0)), 'f7': (-2/math.pi*l-1.0,l/(-2/math.pi*l-1.0)),
                            'f8': (2/math.pi*l+1.8,l/(2/math.pi*l+1.8)), 'f9': (-2/math.pi*l-1.8,l/(-2/math.pi*l-1.8))
                            }
    
    # Heuristic Function of A_star: Euclidian Distance is chosen   
    def heuristic(self,goal,state):
        state_c = self.disc2cont(state)
        (xs,ys,ths) = state_c
        (xf,yf) = goal
        return math.sqrt((xs-xf)**2 + (ys-yf)**2)
    
    def check_forward_actions(self, x_v, y_v, length, r, dth):
        # Test all points are within boundaries
        if np.any(abs(x_v) > self.limit_x) or np.any(abs(y_v) > self.limit_y):
            return None, None
        # Test all points do not cross untraverable cells
        x_cell_prev = None
        y_cell_prev = None
        x_cell_new = None
        y_cell_new = None
        cost_action = 0
        num_cell_action = 0
        for i_xv, i_yv in zip(x_v, y_v):
            x_cell_new = int(np.floor((i_xv-(-self.limit_x))/self.discr))
            y_cell_new = int(np.floor((i_yv-(-self.limit_y))/self.discr))
            if x_cell_new != x_cell_prev or y_cell_new != y_cell_prev:
                cost_cell = self.cost[y_cell_new,x_cell_new]
                if cost_cell >=255:
                    return None, None
                num_cell_action +=1
                cost_action += cost_cell
            x_cell_prev = x_cell_new
            y_cell_prev = y_cell_new
        
        # If I am here I passed both tests (i.e. the trajectory is safe)
            
        # I check if I cross the goal circle
        if np.any(np.linalg.norm([x_v-self.goal[0], y_v-self.goal[1]], axis = 0) < self.goal_radius):
            end = 'g'
        else:
            end = 's'
            
        # The total cost over an action is defined as the average cost of the
        # traversed cells, and normalized wrt the trajectory lenght (i.e, min cost = 0, max cost = length)
        avg_cost = cost_action/num_cell_action
        final_cost = length*avg_cost/255.0
        
        return final_cost, end
    
    def check_rotation_actions(self, dth, state_c):
        (xd,yd,thd) = self.cont2disc(state_c)
        # Cost of rotation is given by the cost of the cell occupied by the centre of mass
        # and normalized wrt the rotation angle and a weight rotation_cost_factor
        # (i.e., min cost = 0, max cost = abs(dth)*rotation_cost_factor)
        if self.cost[yd,xd] >=255:
            return None, None
        else:
            final_cost = abs(dth)*self.cost[yd,xd]/255.0 * self.rotation_cost_factor
            end = 's'
            return final_cost, end
    
    # Check validity of actions (don't cross unsafe cells, and don't go out of map boundaries)
    # and compute next state, cost, and end value for valid actions
    def passable(self, current_state, action_id, all_actions = True):
        # Conversion from discrete to continuous
        (xi, yi, th_i) = self.disc2cont(current_state)
        
        if all_actions:
            (r, dth) = self.all_actions[action_id]
        else:
            (r, dth) = self.forward_actions[action_id]
        
        # Trajectory is discretised in n_cell points and
        # New state (state_f) and infos (final_cost, end) are computed
        if r is not None: #no point turn rotations
            if dth is not None: # no straight lines
                et = np.linspace(th_i,th_i+dth,self.n_cell);
                x_v = xi - r*math.cos(th_i) + r*np.cos(et)
                y_v = yi - r*math.sin(th_i) + r*np.sin(et)
                length = r*dth
            else: # straight lines
                et = np.linspace(th_i,th_i,self.n_cell)
                x_v = np.linspace(xi,xi-r*np.sin(th_i), self.n_cell)
                y_v = np.linspace(yi,yi+r*np.cos(th_i), self.n_cell)
                length = r
            final_cost, end = self.check_forward_actions(x_v,y_v, length, r, dth)
            # Conversion from continous to discrete of new state
            state_f = self.cont2disc((x_v[-1],y_v[-1],et[-1]))  
        else: #point turn rotations
            final_cost, end = self.check_rotation_actions(dth, (xi, yi, th_i))
            th_f = th_i + dth
            state_f = self.cont2disc((xi,yi,th_f))
            
        if final_cost is None:
            return None, None
        else:
            return state_f, (action_id, end, final_cost)
    
    # Find Valid Neighbors states from current state
    def neighbors(self, current_state, all_actions = True):
        # Compute Neighbors states and info based on Rover actions
        new_state = []
        info = []
        
        # I consider rotation actions only for the staring position (parent is None) 
        if all_actions:
            actions = self.all_actions
        else:
            actions = self.forward_actions
            
        for action_id in actions.keys():
            ns, inf = self.passable(current_state, action_id, all_actions)
            if ns is not None:
                new_state.append(ns)
                info.append(inf)
                        
        return new_state, info
    
    # Conversion from discrete to continuous    
    def disc2cont(self, s):
        (xd,yd,thd) = s
        xi = round(xd*self.discr - self.limit_x, 3)
        yi = round(yd*self.discr - self.limit_y, 3)
        th_i = round(thd*math.pi/180, 3)
        return (xi, yi, th_i)
    
    # Conversion from continuous to discrete
    def cont2disc(self, s):
        (xc,yc,thc) = s
        xd = int(np.floor((xc-(-self.limit_x))/self.discr))
        yd = int(np.floor((yc-(-self.limit_y))/self.discr))
        if thc<0:
            thd = int((thc+2*math.pi)*180/math.pi)
        elif thc > 2*math.pi:
            thd = int((thc-2*math.pi)*180/math.pi)
        else:
            thd = int(thc*180/math.pi)
        return (xd,yd,thd) 
    
    # Reconstruct Path once search is done
    def obtain_path(self, final_s):
        s = final_s
        s_cont = self.disc2cont(s)
        actions = []
        states = []
        costs = []
        states.append(s_cont)
        # Loop backward to find all state, action, and cost parents from final state
        while True:
            value = self.came_from.get(s)
            if value[0] is not None:
                actions.append(value[1][0])
                costs.append(value[1][2])
                s = value[0]
                s_cont = self.disc2cont(s)
                states.append(s_cont)   
            else:
                break
        actions.reverse()
        states.reverse()
        costs.reverse()
        return states, actions, costs
    
    # Plot of found path
    def plot_path_cost_map(self):
        # Number of Cells for Traversability Map
        num_points_x = int(self.limit_x*2/self.discr + 1)
        num_points_y = int(self.limit_y*2/self.discr + 1)
        x_tr = np.linspace(-self.limit_x,self.limit_x,num_points_x)
        y_tr = np.linspace(-self.limit_y,self.limit_y,num_points_y)
        Z= self.cost
        X, Y = np.meshgrid(x_tr,y_tr)
        
        fig = plt.figure(figsize=(15,15))
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_title("A* Path", fontsize = 35)
        ax.tick_params(labelsize=40)
        ax.set_xlabel("X [m]", fontsize = 35)
        ax.set_ylabel("Y [m]", fontsize = 35, rotation = 0, va= "bottom", labelpad = 25)
        im = ax.pcolormesh(X,Y,Z, cmap=cm.coolwarm)
        cb = fig.colorbar(im, ax =ax)
        cb.ax.tick_params(labelsize=40)
        cb.set_label("Cost", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
        
        # Draw a Circle around the target
        circle1 = plt.Circle((self.goal[0], self.goal[1]), self.goal_radius, color='g')
        ax.add_artist(circle1)
        # Plot of trajectories
        for state, action in zip(self.states, self.actions):
            ax.plot(state[0], state[1], 'o', c = 'b')
            vx = -np.sin(state[2])*0.4
            vy = np.cos(state[2])*0.4
            plt.arrow(state[0],state[1],vx,vy, head_width=0.1, head_length=0.2, color = 'r' )
            
            x_v, y_v, _ = self.compute_trajectory(state, action)
            ax.plot(x_v, y_v, color = "orange")
        state = self.states[-1]
        ax.plot(state[0], state[1], 'o', c = 'b')
        vx = -np.sin(state[2])*0.4
        vy = np.cos(state[2])*0.4
        plt.arrow(state[0],state[1],vx,vy, head_width=0.1, head_length=0.2, color = 'r' )
        
        plt.show()
    
    def compute_trajectory(self, current_state, action_id):
        (xi,yi,th_i) = current_state
        (r, dth) = self.all_actions[action_id]
        # Trajectory is discretised in n_cell points
        if r is not None: #no point turn rotations
            if dth is not None: # no straight lines
                et = np.linspace(th_i,th_i+dth,self.n_cell);
                x_v = xi - r*math.cos(th_i) + r*np.cos(et)
                y_v = yi - r*math.sin(th_i) + r*np.sin(et)
            else: # straight lines
                et = np.linspace(th_i,th_i,self.n_cell)
                x_v = np.linspace(xi,xi-r*np.sin(th_i), self.n_cell)
                y_v = np.linspace(yi,yi+r*np.cos(th_i), self.n_cell)
        else: #point turn rotations
            et = np.linspace(th_i,th_i+dth,self.n_cell);
            x_v = np.linspace(xi,xi, self.n_cell)
            y_v = np.linspace(yi,yi, self.n_cell)
        
        return x_v, y_v, et
                  
    # Main Search Function
    def search(self, plot = None):
        if plot is None:
            plot = self.plot
        # Conversion from continuous to discrete for start
        start = self.cont2disc(self.start)
        # Initialise priority queue with start state    
        self.frontier = PriorityQueue()
        self.frontier.put(start, 0)
        # Initialise parents and costs dictionaries with start values
        self.came_from = {}
        cost_so_far = {}
        self.came_from[start] = (None,(None,'s',None))
        cost_so_far[start] = 0
        # Result Variables if A* succeds
        self.states = []
        self.actions = []
        self.costs = []
        self.find = False
        
        # A* loop until I find the optimal path or my queue is empty and I didnt find a path
        while not self.frontier.empty() and not self.find:
            # Pick state with highest priority from queue
            current = self.frontier.get()
            # Check for optimal path found (this happens if a goal state is at the top priority of the frontier)
            if current in self.came_from:
                (parent, info) = self.came_from[current]
                if info[1] == 'g':
                    self.find = True
                    break
            # For my specific implementation I decide to consider rotation actions only when
            # I am at the start state, then I only consider forward actions
            if current == start:
                all_actions = True
            else:
                all_actions = False
            # Compute list of all next states (tot_next) and infos (tot_info_next: action_id, end, cost)
            # for all actions from the current state
            tot_next, tot_info_next = self.neighbors(current, all_actions)
            # Update all next states costs and priorities according to A* algorithm
            for next, info_next in zip(tot_next, tot_info_next):
                new_cost = cost_so_far[current] + info_next[2]
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(self.goal,next)
                    self.frontier.put(next, priority)
                    self.came_from[next] = (current, info_next)
        
        if self.find:        
            self.states, self.actions, self.costs = self.obtain_path(current)
            if plot:
                self.plot_path_cost_map()
    