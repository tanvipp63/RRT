#REF
#https://www.youtube.com/watch?v=OXikozpLFGo

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import random
import shapely
import time




class treeNode():
    def __init__(self, locationX:float, locationY:float):
        """
        Creates a new node in tree

        locationX: x coordinate
        locationY: y coordinate
        children: children of node
        parent: parent of node
        """
        self.locationX = locationX
        self.locationY = locationY
        self.children = []
        self.parent = None

    def equals(self, node1):
        """
        Method for tree node that compares two nodes and returns if they are equal or not

        Params:
        node1: node to compare

        Outputs:
        bool - equal or not equal
        """
        if (self.locationX == node1.locationX):
            if (self.locationY == node1.locationY):
                return True
        return False
    
    def toArray(self):
        """
        Turns node into array

        Outputs:
        node: node as a (1,2) array
        """
        node = np.array([self.locationX, self.locationY])
        return node
    

class RRTAlgorithm():
    def __init__(self, start:np.ndarray, goal:np.ndarray, numIterations:int, grid:np.ndarray, stepSize:float):
        """
        Starts RRT and specifies start and goal.

        start: np.array of start node [X, Y]
        goal: np.array of goal node [X,Y]
        numIterations: number of iterations, limited to 200
        grid: occupancy grid
        stepSize or rho: rho i.e. length of each branch
        nearestNode: nearest node to current node
        path_distance: total path distance
        nearestDist: distance to nearest node
        numWaypoints:  number of waypoints
        Waypoints: the waypoints as a list

        """
        self.randomTree = treeNode(start[0], start[1])
        self.goal = treeNode(goal[0], goal[1])
        self.nearestNode = None
        self.iterations = min(numIterations, 200)
        self.grid = grid
        self.rho = stepSize
        self.path_distance = 0
        self.nearestDist = 10000
        self.numWaypoints = 0
        self.Waypoints = []


    def sampleAPoint(self):
        """
        Sample a random point within grid limits

        Outputs:
        x: x coordinate of random point
        y: y coordinate of random point
        """

        x = random.randint(1, self.grid.shape[1]-1)
        y = random.randint(1, self.grid.shape[0]-1)

        return np.array([[x], [y]])

    def steerToPoint(self, locationStart:treeNode, locationEnd:np.ndarray):
        """
        Steer a distance stepsize from start to end location

        Params:
        locationStart, locationEnd: start and end node

        Outputs:
        newPoint: point that was steered to
        None: if unit vector is 0
        """
        norm = self.unitVector(locationStart, locationEnd)
        if norm.all() == None:
            return np.array([[None], [None]])
        path = norm * self.rho

        newX = locationStart.locationX + path[0,0]
        newY = locationStart.locationY + path[1,0]

        #Boundary conditions
        if newX >= self.grid.shape[1]:
            newX = self.grid.shape[1] - 1
        if newX < 0:
            newX = 0
        if newY >= self.grid.shape[0]:
            newY = self.grid.shape[0] - 1
        if newY < 0:
            newY = 0

        newPoint = np.array([[newX], [newY]])

        return newPoint

    def isInObstacleOld(self, locationStart:treeNode, locationEnd:np.ndarray):
        """
        Check if an obstacle lies between the start node and end point of the edge
        """
        
        norm = self.unitVector(locationStart, locationEnd)
        pathPoint = np.zeros((2,1))
        for i in range(self.rho):
            pathPoint[0,0] = locationStart.locationX + i * norm[0,0]
            pathPoint[1,0] = locationStart.locationY + i * norm[1,0]

            x = round(pathPoint[0,0])
            y = round(pathPoint[1,0])

            #Boundary conditions
            if x >= self.grid.shape[1]:
                x = self.grid.shape[1] - 1
            if y >= self.grid.shape[0]:
                y = self.grid.shape[0] - 1
                        
            if self.grid[y][x] == 1:
                return True
        return False

    def bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def isInObstacle(self, locationStart: treeNode, locationEnd: np.ndarray):
        x0, y0 = round(locationStart.locationX), round(locationStart.locationY)
        x1, y1 = round(locationEnd[0,0]), round(locationEnd[1,0])


        path_points = list(self.bresenham(x0, y0, x1, y1))

        for x, y in path_points:
            #Boundary conditions
            if x >= self.grid.shape[1]:
                x = self.grid.shape[1] - 1
            if y >= self.grid.shape[0]:
                y = self.grid.shape[0] - 1
            
            if self.grid[y][x] == 1:
                    return True
        return False



    def unitVector(self, locationStart:treeNode, locationEnd:np.ndarray):
        """
        Find unit vector between 2 points which form a vector

        Params:
        locationStart: starting node
        locationEnd: end point

        Outputs:
        norm: unit vector
        None: if unit vector is 0
        """
        x = locationEnd[0,0] - locationStart.locationX
        y = locationEnd[1,0] - locationStart.locationY

        #Check if x & y are not 0 i.e. no random point is same as nearest point
        if (x == 0) and (y == 0):
            return np.array([[None], [None]])

        magnitude = np.sqrt(x**2 + y**2)
        norm = np.array([[x], [y]])/magnitude
        return norm

    def findNearest(self, root:treeNode, point:np.ndarray):
        """
        Find the nearest node from a given unconnected point (Euclidean distance)
        Use recursion

        Params:
        root: root of tree
        point: point to find nearest node to
        """
        if not root:
            return

        #Get current distance between node and point
        currDist = self.distance(root, point)

        #If current node is closer, update nearest node and distance
        if currDist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = currDist
        
        #Recursively check each child node
        for child in root.children:
            self.findNearest(child, point)
                    

    def distance(self, node1:treeNode, point:np.ndarray):
        """
        Find euclidean distance between a node and an XY point
        
        node1: node 
        XY point: another point
        """
        distance = np.sqrt((node1.locationX - point[0,0])**2 + (node1.locationY - point[1,0])**2)
        return distance

    def goalFound(self, point:np.ndarray):
        """
        Check if the goal has been reached within step size
        """
        return self.distance(self.goal, point) <= self.rho

    def addChild(self, locationX:float, locationY:float):
        """
        Add the point to the nearest node and add goal when reached

        locationX: x coordinate of node to add
        locationY: y coordinate of node to add
        """

        if (self.goalFound(np.array([[locationX], [locationY]]))):
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            child = treeNode(locationX, locationY)
            self.nearestNode.children.append(child)
            child.parent = self.nearestNode

    def resetNearestValues(self):
        """
        Reset nearestNode and nearestDistance
        """
        self.nearestNode = None
        self.nearestDist = 10000

    def retraceRRTPath(self, goal:treeNode):
        """
        Trace the path from goal to start.
        Updates waypoints accordingly

        Params:
        goal: goal tree node
        """
        #Terminate if goal is reached
        if goal.equals(self.randomTree):
            return
        
        #Update parameters
        self.numWaypoints += 1
        self.Waypoints.insert(0, goal.toArray())
        self.path_distance += self.rho
        self.retraceRRTPath(goal.parent)

class robot():
    def __init__(self, x:np.ndarray):
        """
        Robot is 1m wide and 1.5m long, and rectangular in shape
        The x axis is pointing right positive, y axis pointing up positive
        x is a (3,1) array with the pose of the robot. The axis is located at the centre of the robot.
        th is given in degrees
        """
        self.x = x[0,0]
        self.y = x[1,0]
        self.th = np.deg2rad(x[2,0])
        self.updateCorners()
    
    def updateCorners(self):
        """
        Updates the corners of the robot based on its current pose.
        """
        # Define the robot corners in the local frame
        corners = np.array([[-0.5, 0.75], [0.5, 0.75], [-0.5, -0.75], [0.5, -0.75]]).T

        # Create the rotation matrix
        RE2 = np.array([[np.cos(self.th), -np.sin(self.th)], [np.sin(self.th), np.cos(self.th)]])

        # Apply the rotation and translation to the corners
        corners = RE2 @ corners
        corners[0, :] += self.x
        corners[1, :] += self.y

        # Update the corner points
        self.c1 = corners[:, 0:1]
        self.c2 = corners[:, 1:2]
        self.c3 = corners[:, 3:4]
        self.c4 = corners[:, 2:3]        

    def move(self, newX:float, newY:float):
        """
        Gets the positions of the corners of the robot.
        c1 is top left corner
        c2 is top right corner
        c4 is bottom left corner
        c3 is bottom right corner
        """
        self.x = newX
        self.y = newY
        self.updateCorners()

    def rotate(self, newth:float):
        """
        Applies rotation matrix to get corners at new orientation
        """
        self.th = np.deg2rad(newth)
        self.updateCorners()
    
    def getCorners(self):
        """
        Returns array of plottable corners for robot
        """
        corners = [self.c1, self.c2, self.c3, self.c4]
        cornersPlot = []
        for i in range(4):
            cornersPlot.append((corners[i][0,0], corners[i][1,0]))
        cornersPlot.append((corners[0][0,0], corners[0][1,0]))
        return cornersPlot

def robotFollower(start:np.ndarray):
        """
        Plots the path of the robot along waypoints
        """
        #Init robot
        mu = np.array([[start[0]], [start[1]], [0]])
        rob = robot(mu)
        rob = shapely.Polygon(rob.getCorners())
        x, y = rob.exterior.xy
        ax.plot(x, y, 'r-')
        plt.pause(0.1)
    
def runRRT(start:np.ndarray, goal:np.ndarray, numIterations: int, grid:np.ndarray, stepSize:float, rrt:RRTAlgorithm, success:bool):
    """
    Runs rrt algorithm

    Params:
    start goal: arrays (1,2) for start and end coordinates
    numIterations: number of iterations
    grid: occupancy grid
    stepSize: step size of graph
    rrt: tree
    success: bool for whether goal was found

    Output:
    success: True if goal is found, else false
    """
    for count in range (rrt.iterations):
        #Reset
        rrt.resetNearestValues()

        #Find a sample point and nearest node
        xrand = rrt.sampleAPoint()
        plt.plot(xrand[0,0], xrand[1,0], 'mo')
        rrt.findNearest(rrt.randomTree, xrand)
        print("Rand point", xrand[0,0], xrand[1,0])
        print("nearest", rrt.nearestNode.locationX, rrt.nearestNode.locationY)
        
        #Attempt to get to new point with new edge
        newPoint = rrt.steerToPoint(rrt.nearestNode, xrand)
        print("New Point", newPoint[0,0], newPoint[1,0])
        invalidNewPoint = newPoint.all() == None

        if not invalidNewPoint:
            #Check if new edge is valid
            if not rrt.isInObstacle(rrt.nearestNode, newPoint):
                rrt.addChild(newPoint[0,0], newPoint[1,0])

                #Plot new edge
                ax.plot([rrt.nearestNode.locationX, newPoint[0,0]], [rrt.nearestNode.locationY, newPoint[1,0]], 'yo', linestyle = "--")
                plt.pause(0.10)

                if rrt.goalFound(newPoint):
                    rrt.addChild(goal[0], goal[1])
                    success = True
                    break
    return success    

if __name__ == "__main__":
    #Specify inputs
    grid = np.load('cspace.npy') 
    start = np.array([1.0, 1.0]) #Adjust these coords
    startNode = treeNode(1.0, 1.0)
    goal = np.array([15, 17.5])
    numIterations = 100
    stepSize = 1.5
    #goal region
    goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill = False)

    #Plot
    fig = plt.figure("RRT Algorithm")
    plt.imshow(grid, cmap='binary')
    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'bo')
    ax = fig.gca()
    ax.add_patch(goalRegion)
    plt.xlabel('X-axis $(m)$')
    plt.ylabel('Y-axis $(m)$')

    #Call RRT algorithm
    rrt = RRTAlgorithm(start, goal, numIterations, grid, stepSize)
    success = False #flag to detect if path was successful or not
    success = runRRT(start, goal, numIterations, grid, stepSize, rrt, success)

    if success:
        rrt.retraceRRTPath(rrt.goal)
        rrt.Waypoints.insert(0,startNode.toArray())

        #Plot final path
        for i in range(len(rrt.Waypoints) - 1):
            ax.plot([rrt.Waypoints[i][0], rrt.Waypoints[i+1][0]], [rrt.Waypoints[i][1], rrt.Waypoints[i+1][1]], 'go', linestyle = "--")
            plt.pause(0.1)

        #Robot follower
        robotFollower(start)

    else:
        print("Path not found")
    plt.show()    


