from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.patches as patches

def plot(map_size, discr, Z, width, length, states, actions, goal, goal_radius, all_actions):
    DEM_size = int(map_size/discr +1)
    x = np.linspace(-map_size/2,map_size/2,num=DEM_size)
    y = np.linspace(-map_size/2,map_size/2,num=DEM_size)
    X , Y = np.meshgrid(x,y)
    
    # Plot Map
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_title("Path on Map", fontsize = 35)
    ax.tick_params(labelsize=40)
    ax.set_xlabel("X [m]", fontsize = 35)
    ax.set_ylabel("Y [m]", fontsize = 35, rotation = 0, va= "bottom", labelpad = 25)
    im = ax.pcolormesh(X,Y,Z, cmap=cm.coolwarm)
    cb = fig.colorbar(im, ax =ax)
    cb.ax.tick_params(labelsize=40)
    cb.set_label("Z [m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    
    # Draw a Circle around the target
    circle1 = plt.Circle((goal[0], goal[1]), goal_radius, color='g')
    ax.add_artist(circle1)
    # Plot of trajectories
    for state, action in zip(states, actions):
        # Plot Centre of mass coordinate and direction
        ax.plot(state[0], state[1], 'o', c = 'b')
        vx = -np.sin(state[2])*0.4
        vy = np.cos(state[2])*0.4
        plt.arrow(state[0],state[1],vx,vy, head_width=0.1, head_length=0.2, color = 'r' )
        # Plot trajectory
        x_v, y_v, _ = compute_trajectory(state, action, all_actions)
        ax.plot(x_v, y_v, color = "orange")
        # Plot Robot
        xbl, ybl = bottom_left_wheel(state, width, length)
        ax.add_patch(patches.Rectangle((xbl,ybl), width, length, angle = state[2]*180/math.pi, alpha = 0.4))
    # Plot last Centre of mass coordinate and direction
    state = states[-1]
    ax.plot(state[0], state[1], 'o', c = 'b')
    vx = -np.sin(state[2])*0.4
    vy = np.cos(state[2])*0.4
    plt.arrow(state[0],state[1],vx,vy, head_width=0.1, head_length=0.2, color = 'r' )
    # Plot last Robot
    xbl, ybl = bottom_left_wheel(state, width, length)
    ax.add_patch(patches.Rectangle((xbl,ybl), width, length, angle = state[2]*180/math.pi, alpha = 0.4))
    
    plt.show()

def compute_trajectory(current_state, action_id, all_actions):
    n_cell = 20
    (xi,yi,th_i) = current_state
    (r, dth) = all_actions[action_id]
    # Trajectory is discretised in n_cell points
    if r is not None: #no point turn rotations
        if dth is not None: # no straight lines
            et = np.linspace(th_i,th_i+dth,n_cell);
            x_v = xi - r*math.cos(th_i) + r*np.cos(et)
            y_v = yi - r*math.sin(th_i) + r*np.sin(et)
        else: # straight lines
            et = np.linspace(th_i,th_i,n_cell)
            x_v = np.linspace(xi,xi-r*np.sin(th_i), n_cell)
            y_v = np.linspace(yi,yi+r*np.cos(th_i), n_cell)
    else: #point turn rotations
        et = np.linspace(th_i,th_i+dth,n_cell);
        x_v = np.linspace(xi,xi, n_cell)
        y_v = np.linspace(yi,yi, n_cell)
    
    return x_v, y_v, et

def bottom_left_wheel(state, width, length):
    (XC,YC,Theta) = state
    pcenter = np.array([[XC],[YC]])
    rcenter = np.matrix(((np.cos(Theta), -np.sin(Theta)), (np.sin(Theta), np.cos(Theta))))
    pbl=pcenter+rcenter*np.array([[-width/2],[-length/2]])
    xbl=np.asscalar(pbl[0])
    ybl=np.asscalar(pbl[1])
    return xbl,ybl