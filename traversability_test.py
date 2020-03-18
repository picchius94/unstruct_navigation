import numpy as np
import math
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Class Cost Maps Traversability Analysis      
class Traversability_Map(object):
    def __init__(self, map_size, discr, Z = None, plot = False, width = 0.835, length = 1.198, rover_clearance = 0.108, max_pitch = 36, residual_ratio = 5, non_traversable_threshold = 245):
        self.plot = plot
        #### Map Parameters
        self.map_size = map_size
        self.discr = discr
        self.DEM_size = int(map_size/discr +1)
        self.x = np.linspace(-map_size/2,map_size/2,num=self.DEM_size)
        self.y = np.linspace(-map_size/2,map_size/2,num=self.DEM_size)
        
        #### Parameters for Traversability Analysis
        # Rover width in meters
        self.width = width
        # Rover length in meters
        self.length = length
        # Rover radius in meters
        self.radius = math.sqrt((length/2)**2+(width/2)**2)
        # Number of Cells for Traversability Map (traversability cannot be computed for points closer than
        # the rover radius to the map border)
        self.x_tr = self.x[map_size/2- abs(self.x)>=self.radius]
        self.y_tr = self.y[map_size/2- abs(self.y)>=self.radius]
        self.map_size_tr = max(self.x_tr)
        # Rover clearance from ground
        self.rover_clearance = rover_clearance
        # Max Tolerated Pitch Angle
        self.max_pitch = max_pitch
        # residual_ratio has been selected experimentally to obtain a proportion between residual and rover clearance
        self.residual_ratio = residual_ratio
        # If at least one of tests is over non_traversable_threshold, cell is non-traversable
        self.non_traversable_threshold = non_traversable_threshold
        
        #### Cost Maps
        self.roughness = np.empty((len(self.y_tr),len(self.x_tr)), dtype = np.uint8)
        self.slopes = np.empty((len(self.y_tr),len(self.x_tr)), dtype = np.uint8)
        self.obstacles = np.empty((len(self.y_tr),len(self.x_tr)), dtype = np.uint8)
        self.tot = np.empty((len(self.y_tr),len(self.x_tr)), dtype = np.uint8)
        
        if Z is not None:
            self.analysis(Z,plot)
        
    # Function for Traversability Analysis of a Z elevation Map
    def analysis(self, Z, plot = None):
        if plot is None:
            plot = self.plot
            
        # For all the valid point of the map (i.e, which can contain a circle of the rover radius within the map)
        for i_x, xi in enumerate(self.x_tr):
            for i_y, yi in enumerate(self.y_tr):
                # Loop to collect all points of the circle for one centre point (xi,yi)
                points_x = []
                points_y = []
                points_z = []
                xp = self.x[abs(self.x-xi)<=self.radius]
                for x_cel in xp:
                    yp = self.y[(self.y-yi)**2<=self.radius**2-(x_cel-xi)**2]
                    mask_y = np.floor((yp-(-self.map_size/2))/self.discr).astype(int)
                    z_cells = Z[mask_y,int(np.floor((x_cel-(-self.map_size/2))/self.discr))]
                    points_x.extend([x_cel]*len(yp))
                    points_y.extend(yp)
                    points_z.extend(z_cells)      
                # Fit Plane which better approximates the points
                tmp_A = []
                tmp_b = []
                for i in range(len(points_x)):
                    tmp_A.append([points_x[i], points_y[i], 1])
                    tmp_b.append(points_z[i])
                b = np.matrix(tmp_b).T
                A = np.matrix(tmp_A)
                fit = (A.T * A).I * A.T * b  # i.e the 3 parameters a,b,c of the plane ax + by + c = z
                errors = b - A * fit
                
                # Traversability Costs
                # Roughness Test
                residual = np.linalg.norm(errors)
                # residual_ratio has been selected experimentally to obtain a proportion between residual and rover clearance
                roughness_ratio = (residual/self.residual_ratio)/self.rover_clearance
                r_value = int(min([1, roughness_ratio])*255)
                self.roughness[i_y][i_x] = r_value
                
                # Inclination Test
                # I define a unitary vector, thus: nz = cos(Pitch)
                nz = 1/math.sqrt(fit[0]**2+fit[1]**2+1)
                slopes_ratio = np.arccos(nz)*180/math.pi/self.max_pitch
                sl_value = int(min([1, slopes_ratio])*255)
                self.slopes[i_y][i_x] = sl_value        
                
                # Obstacle Test
                # Check zpoint are lower than the plane for that point + rover belly
                obstacles_ratio = errors.max()/self.rover_clearance
                ob_value = int(min([1, obstacles_ratio])*255)
                self.obstacles[i_y][i_x] = ob_value
                
    #            # Step Test
    #            # Difference between maximum and minimum points in the terrain patch
    #            steps_ratio = (b.max()-b.min())/rover_clearance 
    #            st_value = int(min([1, steps_ratio])*255)
    #            tr_map.steps[i_y][i_x] = st_value
                
                # Total Cost Map: if at least one of tests is over non_traversable_threshold cell is non-traversable, otherwise average
                if np.any(np.array([r_value, sl_value, ob_value]) > self.non_traversable_threshold):
                    self.tot[i_y][i_x] = 255
                else:
                    self.tot[i_y][i_x] = int(np.mean([r_value, sl_value, ob_value]))
        
        if plot:        
            # self.plot_3D('obstacles')
            self.plot_colormesh('obstacles')
            # self.plot_3D('slopes')
            self.plot_colormesh('slopes')
            # self.plot_3D('roughness')
            self.plot_colormesh('roughness')
            # self.plot_3D('tot')
            self.plot_colormesh('tot')
            
    def plot_colormesh(self, feature):
        if feature == 'obstacles':
            Z = self.obstacles
        elif feature == 'slopes':
            Z = self.slopes
        elif feature == 'roughness':
            Z = self.roughness
        elif feature == 'tot':
            Z = self.tot
        else:
            print("Traversability Plot Option not valid")
            return -1
        Z = Z/255.0
        X, Y = np.meshgrid(self.x_tr,self.y_tr)
        fig = plt.figure(figsize=(15,15))
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_title("Cost {}".format(feature), fontsize = 35)
        ax.tick_params(labelsize=40)
        ax.set_xlabel("X [m]", fontsize = 35)
        ax.set_ylabel("Y [m]", fontsize = 35, rotation = 0, va= "bottom", labelpad = 25)
        im = ax.pcolormesh(X,Y,Z, cmap=cm.coolwarm)
        cb = fig.colorbar(im, ax =ax)
        cb.ax.tick_params(labelsize=40)
        cb.set_label("Cost", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
        plt.show()                 
    def plot_3D(self, feature):
        if feature == 'obstacles':
            Z = self.obstacles
        elif feature == 'slopes':
            Z = self.slopes
        elif feature == 'roughness':
            Z = self.roughness
        elif feature == 'tot':
            Z = self.tot
        else:
            print("Traversability Plot Option not valid")
            return -1
        Z = Z/255.0
         
        # Plot of Feature Traversability Map        
        X_tr, Y_tr = np.meshgrid(self.x_tr,self.y_tr)
        plt.figure(figsize=(15,15))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X_tr,Y_tr,Z)
        ax.set_title("Cost {}".format(feature), fontsize = 35)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        # Plot of Non-plannable areas
        x1 = self.x[self.x > self.map_size/2 - self.radius]
        x2 = self.x[self.x < self.radius - self.map_size/2]
        y1 = self.x[self.y > self.map_size/2 - self.radius]
        y2 = self.x[self.y < self.radius - self.map_size/2]
        X1,Y1 = np.meshgrid(x1,self.y)
        X2,Y2 = np.meshgrid(x2,self.y)
        X3,Y3 = np.meshgrid(self.x,y1)
        X4,Y4 = np.meshgrid(self.x,y2)
        ax.plot_surface(X1,Y1,np.zeros((self.DEM_size,x1.shape[0])), color='g')
        ax.plot_surface(X2,Y2,np.zeros((self.DEM_size,x2.shape[0])), color='g')
        ax.plot_surface(X3,Y3,np.zeros((y1.shape[0],self.DEM_size)), color='g')
        ax.plot_surface(X4,Y4,np.zeros((y2.shape[0],self.DEM_size)), color='g')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([self.x.max()-self.x.min(), self.y.max()-self.y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.x.max()+self.x.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.y.max()+self.y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
        #Set camera view rotation angles
        ax.view_init(azim=-60,elev=50)
        plt.show()
        # print("{}".format(feature))
        # print("")
        # print("Max value: {}".format(Z.max()))
        # print("")
        # print("Min value: {}".format(Z.min()))
        # print("")
        # print("Mean value: {}".format(Z.mean()))
        # print("")