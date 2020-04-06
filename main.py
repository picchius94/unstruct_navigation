import terrain_generator as tg
import traversability_test as tt
import A_star_lattice as astar
import plot_path as pp
from func_timeout import func_timeout, FunctionTimedOut
import math

def main():
    #########################################
    ######## Parameter Definition ###########
    #########################################
    # Graphs yes or no
    plot = True
    ### Map Parameters
    map_size = 8
    discr = 0.0625
    terrain_type = "scattered_sharp"
    ### Robot and Traversability Parameters
    width = 0.61
    length = 0.58
    rover_clearance = 0.3
    max_pitch = 20
    residual_ratio = 5
    non_traversable_threshold = 245
    ### Path Planning Parameters
    start = (0,0,0)
    goal = (2,2)
    goal_radius = 0.3
    # Max seconds for A star algorithm
    TIMEOUT = 5.0
    # Cost multipliers for Forward and Rotation actions
    rotation_cost_factor = 1.46
    forward_cost_factor = 1
    # Action Definition. Each action is defined by two parameters.
    # If the first is None, the second is the angle (in rad) of a point turn rotation
    # If the second is None, the first is the length (in m) of a straight line
    # If both parameters are defined, the first is the radius (in m), and the second is the angle (in rad) of the arc of a circle
    forward_length = 1.65
    l = forward_length
    all_actions = {'s0': (l,None),
                            
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
    forward_actions = {'s0': (l,None),
                            
                            'fm': (2/math.pi*l,l/(2/math.pi*l)), 'fm1': (-2/math.pi*l,l/(-2/math.pi*l)),
                            'f0': (2/math.pi*l+0.1,l/(2/math.pi*l+0.1)), 'f1': (-2/math.pi*l-0.1,l/(-2/math.pi*l-0.1)),
                            'f2': (2/math.pi*l+0.2,l/(2/math.pi*l+0.2)), 'f3': (-2/math.pi*l-0.2,l/(-2/math.pi*l-0.2)),
                            'f4': (2/math.pi*l+0.5,l/(2/math.pi*l+0.5)), 'f5': (-2/math.pi*l-0.5,l/(-2/math.pi*l-0.5)),
                            'f6': (2/math.pi*l+1.0,l/(2/math.pi*l+1.0)), 'f7': (-2/math.pi*l-1.0,l/(-2/math.pi*l-1.0)),
                            'f8': (2/math.pi*l+1.8,l/(2/math.pi*l+1.8)), 'f9': (-2/math.pi*l-1.8,l/(-2/math.pi*l-1.8))
                            }
    
    #########################################
    ##### Generate OpenSimplex Map ##########
    #########################################
    terrain = tg.OpenSimplex_Map(map_size, discr, terrain_type=terrain_type, plot = plot)
    terrain.sample_generator()
    #########################################
    #### Compute Traversability Analysis ####
    #########################################
    cost_map = tt.Traversability_Map(map_size, discr, plot = plot, width = width, 
                                     length = length, rover_clearance = rover_clearance,
                                     max_pitch = max_pitch, residual_ratio = residual_ratio,
                                     non_traversable_threshold = non_traversable_threshold)
    cost_map.analysis(Z = terrain.Z)
    #########################################
    ############# A* algorithm ##############
    #########################################
    # Setting parameters
    path = astar.A_star_Graph(cost_map.map_size_tr/2, cost_map.map_size_tr/2, discr, 
                              cost_map.tot, goal, start, goal_radius = goal_radius, 
                              forward_length = forward_length,
                              rotation_cost_factor = rotation_cost_factor, 
                              forward_cost_factor = forward_cost_factor, plot = plot,
                              all_actions = all_actions, forward_actions = forward_actions)
    # Searching for path
    # If A star takes more than TIMEOUT secs A* return without a path
    try:
        func_timeout(TIMEOUT, path.search)
    except FunctionTimedOut:
        # I didn't make it on time
        print("A* couldn't make it in {} s!".format(TIMEOUT))
    except Exception as e:
        # Any other possible exception
        print(e)
    if path.find:
        print("A path has been found in {} actions".format(len(path.actions)))
        print("States (x,y,th): {}".format(path.states))
        print("Actions: {}".format(path.actions))
        print("Action Costs: {}".format(path.costs))
        print("Total Cost: {}".format(sum(path.costs)))
        pp.plot(map_size, discr, terrain.Z, width, length, path.states, path.actions, goal, goal_radius, all_actions)
    else:
        print("No path has been found. Check start {} and goal {} postions".format(start, goal))
    
    
if __name__ == "__main__":
    main()