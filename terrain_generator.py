import random
from opensimplex import OpenSimplex
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Class Parameters of Simplex Algorithm
class OpenSimplex_Map(object):
    def __init__(self, map_size, discr, terrain_type = "scattered_sharp", plot = False):
        self.plot = plot
        #### Map Parameters
        self.map_size = map_size
        self.discr = discr
        self.DEM_size = int(map_size/discr +1)
        self.x = np.linspace(-map_size/2,map_size/2,num=self.DEM_size)
        self.y = np.linspace(-map_size/2,map_size/2,num=self.DEM_size)
        self.Z = np.empty((self.DEM_size,self.DEM_size), dtype=np.float32)
        
        #### Opensimplex Parameters
        # Percentage Map with Mountains
        self.perc_obstacles = 0.0
        # Percentage Smoothnes Transition from Mountains to Plain i.e interpolation from mountains to plains
        self.smooth_transition = 0.0
        # Distribution over axis of Mountains (0 = Only one big mountain, 1 = a lot of small sharp mountains)
        self.distribution_x = 0.0
        self.distribution_y = 0.0
        # Maximum height of Mountains
        self.height_obstacles = 0.0
        # Mountain Smoothness (0 = very Sharp mountain, 1 = very smooth/flat mountain)
        self.smooth_mountains = 0.0
        # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
        self.smooth_plains = 0.0
        # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
        self.slopes_variation = 0.0
        # Maximum Slope Variation (0 = no variation, 1 = max variation, i.e. 15% of map_size)
        self.max_slope_variation = 0.0

        if terrain_type == "mountain_crater":
            self.mountain_crater()
        elif terrain_type == "smooth":
            self.smooth()
        elif terrain_type == "rough":
            self.rough()
        elif terrain_type == "wavy":
            self.wavy()
        elif terrain_type == "scattered_sharp":
            self.scattered_sharp()
        else:
            print("Not valid terrain type: {}".format(terrain_type))
        
    # Functions for setting Simplex Parameters        
    def mountain_crater(self):  
        # Percentage Map with Mountains
        self.perc_obstacles = random.uniform(0.15,0.2)
        # Percentage Smoothnes Transition from Mountains to Plain i.e interpolation from mountains to plains
        self.smooth_transition = random.uniform(0.2,1)
        # Distribution over axis of Mountains (0 = Only one big mountain, 1 = a lot of small sharp mountains)
        self.distribution_x = random.uniform(0.05,0.1)
        self.distribution_y = random.uniform(0.05,0.1)
        if random.random() < 0.5:
            # Maximum height of Mountains
            self.height_obstacles = random.uniform(2,6)
        else:
            # Maximum height of Mountains
            self.height_obstacles = random.uniform(-6,-2)
        # Mountain Smoothness (0 = very Sharp mountain, 1 = very smooth/flat mountain)
        self.smooth_mountains = random.uniform(1,6)
        # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
        self.smooth_plains = random.uniform(0.3,1)
        # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
        self.slopes_variation = random.uniform(0.05,0.5)
        # Maximum Slope Variation (0 = no variation, 1 = max variation, i.e. 15% of map_size)
        self.max_slope_variation = 0.3
    
    def smooth(self):   
        # Percentage Map with Mountains
        self.perc_obstacles = random.uniform(0.05,0.1)
        # Percentage Smoothnes Transition from Mountains to Plain i.e interpolation from mountains to plains
        self.smooth_transition = 0.3
        # Distribution over axis of Mountains (0 = Only one big mountain, 1 = a lot of small sharp mountains)
        self.distribution_x = 0.85
        self.distribution_y = 0.85
        # Maximum height of Mountains
        self.height_obstacles = 0.5
        # Mountain Smoothness (0 = very Sharp mountain, 1 = very smooth/flat mountain)
        self.smooth_mountains = 0.2
        # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
        self.smooth_plains = random.uniform(0.4,0.9)
        # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
        self.slopes_variation = random.uniform(0.08,0.8)
        # Maximum Slope Variation (0 = no variation, 1 = max variation, i.e. 15% of map_size)
        self.max_slope_variation = random.uniform(0.08,0.4)
            
    def rough(self):  
        # Percentage Map with Mountains
        self.perc_obstacles = random.uniform(0.1,0.15)
        # Percentage Smoothnes Transition from Mountains to Plain i.e interpolation from mountains to plains
        self.smooth_transition = 0.1
        # Distribution over axis of Mountains (0 = Only one big mountain, 1 = a lot of small sharp mountains)
        self.distribution_x = random.uniform(0.7,0.85)
        self.distribution_y = random.uniform(0.7,0.85)
        # Maximum height of Mountains
        self.height_obstacles = random.uniform(0.5,1.5)
        # Mountain Smoothness (0 = very Sharp mountain, 1 = very smooth/flat mountain)
        self.smooth_mountains = 0.1
        # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
        self.smooth_plains = random.uniform(0.03,0.05)
        # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
        self.slopes_variation = random.uniform(0.03,0.06)
        # Maximum Slope Variation (0 = no variation, 1 = max variation, i.e. 15% of map_size)
        self.max_slope_variation = random.uniform(0.05,0.5)
       
    def wavy(self): 
        # Percentage Map with Mountains
        self.perc_obstacles = 0.1
        # Percentage Smoothnes Transition from Mountains to Plain i.e interpolation from mountains to plains
        self.smooth_transition = 0.1
        # Distribution over axis of Mountains (0 = Only one big mountain, 1 = a lot of small sharp mountains)
        self.distribution_x = 0.7
        self.distribution_y = 0.7
        # Maximum height of Mountains
        self.height_obstacles = 1.
        # Mountain Smoothness (0 = very Sharp mountain, 1 = very smooth/flat mountain)
        self.smooth_mountains = 0.1
        if random.random() < 0.5:
            # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
            self.smooth_plains = random.uniform(0.3,0.35)
            # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
            self.slopes_variation = random.uniform(0.45,0.55)
        else:
            # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
            self.smooth_plains = random.uniform(0.4,0.7)
            # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
            self.slopes_variation = random.uniform(0.8,1)
        
        # Maximum Slope Variation (0 = no variation, 1 = max variation, i.e. 15% of map_size)
        self.max_slope_variation = 0.5
        
    def scattered_sharp(self):  
        # Percentage Map with Mountains
        self.perc_obstacles = random.uniform(0.1,0.17)
        # Percentage Smoothnes Transition from Mountains to Plain i.e interpolation from mountains to plains
        self.smooth_transition = 0.3
        # Distribution over axis of Mountains (0 = Only one big mountain, 1 = a lot of small sharp mountains)
        self.distribution_x = random.uniform(0.85,0.9)
        self.distribution_y = random.uniform(0.85,0.9)
        # Maximum height of Mountains
        self.height_obstacles = random.uniform(0.3,2.5)
        # Mountain Smoothness (0 = very Sharp mountain, 1 = very smooth/flat mountain)
        self.smooth_mountains = random.uniform(0.1,0.2)
        # Plains Smoothness (0 = very frequent variations, 1 = very rare variations)
        self.smooth_plains = 0.4
        # Slope Variation (0 = no variation from point to point, 1 = maximum variation from point to point)
        self.slopes_variation = 0.5
        # Maximum Slope Variation (0 = no variation, 1 = max variation, i.e. 15% of map_size)
        self.max_slope_variation = 0.4
        
    # Generator of one dataset sample       
    def sample_generator(self, plot = None):
        if plot is None:
            plot = self.plot
        # Generate random seed for Simplex Algorithm
        tmp = OpenSimplex(int(random.random() * 10000))
        # Generate Map
        Z = np.empty((self.DEM_size,self.DEM_size), dtype=np.float32)
        for index_y, y_i in enumerate(self.y):
            for index_x, x_i in enumerate(self.x):
                w = self.interp_curve(tmp,x_i,y_i)
                p = self.plains(tmp,x_i,y_i)
                m = self.mountains(tmp,x_i,y_i)    
                Z[index_y][index_x] = (p*w) + (m*(1-w))
        self.Z = Z
        
        if plot:
            self.plot_3D()
            self.plot_colormesh()
        
    
    def interp_curve(self, tmp, xi, yi):
        value = (tmp.noise2d(xi/((1/self.map_size-self.map_size)*self.distribution_x+self.map_size), yi/((1/self.map_size-self.map_size)*self.distribution_y+self.map_size)) + 1) / 2.0
        start = self.perc_obstacles - 0.2*self.smooth_transition 
        end = self.perc_obstacles + 0.2*self.smooth_transition 
        if value < start:
            return 0.0
        if value > end:
            return 1.0
        return (value-start)*(1/(end-start))
    
    def plains(self, tmp, xi, yi):
        value = (tmp.noise2d(xi/((self.map_size-1/self.map_size)*self.smooth_plains+1/self.map_size), yi/((self.map_size-1/self.map_size)*self.smooth_plains+1/self.map_size)) + 1) *self.max_slope_variation*0.15*self.map_size
        value = value**self.slopes_variation - 0.8
        return value
    
    def mountains(self, tmp, xi, yi):
        value = (tmp.noise2d(xi/((self.map_size-1/self.map_size)*self.smooth_mountains+1/self.map_size), yi/((self.map_size-1/self.map_size)*self.smooth_mountains+1/self.map_size)) + 1) / 2.0
        return value*self.height_obstacles
    
    # Plot Functions
    def plot_colormesh(self):
        X, Y = np.meshgrid(self.x,self.y)
        fig = plt.figure(figsize=(15,15))
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_title("OpenSimplex Map", fontsize = 35)
        ax.tick_params(labelsize=40)
        ax.set_xlabel("X [m]", fontsize = 35)
        ax.set_ylabel("Y [m]", fontsize = 35, rotation = 0, va= "bottom", labelpad = 25)
        im = ax.pcolormesh(X,Y,self.Z, cmap=cm.coolwarm)
        cb = fig.colorbar(im, ax =ax)
        cb.ax.tick_params(labelsize=40)
        cb.set_label("Z [m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
        plt.show()    
    def plot_3D(self):
        X, Y = np.meshgrid(self.x,self.y)
        plt.figure(figsize=(15,15))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X,Y,self.Z, cmap=cm.coolwarm)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), self.Z.max()-self.Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.Z.max()+self.Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
        # Set camera view rotation angles
        ax.view_init(azim=-60,elev=40)
        plt.show()
        # print("Elevation Map")
        # print("")
        # print("Max Z: {}".format(self.Z.max()))
        # print("")
        # print("Min Z: {}".format(self.Z.min()))
        # print("")
        # print("Mean Z: {}".format(self.Z.mean()))
        # print("")