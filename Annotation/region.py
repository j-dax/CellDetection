"""
Created by Lanting Li, Matthew Piazza and Daania Kalam.

This class aims to help with save and access the information of one region.
"""
import cv2
import numpy as np
from math import floor

_DEBUG = True

def makeRegion(pmap, roi, id: int = 0, selected: bool = False,
                text: str = "", visited: bool = None, correct: bool = None):
        temp = Region([0], [0], id, selected, text, visited, correct)
        temp.bottom, temp.left, temp.top, temp.right = roi
        temp.pmap = pmap
        temp.pmap_contours()

        return temp

class Region:
    # cannot default construct xvertices/yvertices as lists are mutable
    #       default arguments are evaluated once, not each time a class object is created
    def __init__(self, xvertices: [int], yvertices: [int], id: int = 0, selected: bool = False,
                text: str = "", visited: bool = None, correct: bool = None):
        self._id = id
        self._selected = selected
        self._xvertices = xvertices
        self._yvertices = yvertices
        self._set_bounds()
        self._set_center()
        self._visited = visited # Initialized as None, should be change to True/False later
        self._correct = correct # Initialized as None, should be change to True/False later
 
    @property
    def visited(self):
        return self._visited

    @visited.setter
    def visited(self, visited):
        self._visited = visited

    def set_visited(self, visited):
        self._visited = visited

    @property
    def correct(self):
        return self._correct

    @correct.setter
    def correct(self, correct):
        self._correct = correct

    def set_correct(self, correct):
        self._correct = correct
        
    @property
    def vertices(self):
        return zip(self._xvertices, self._yvertices)

    @property
    def xvertices(self):
        return self._xvertices

    @property
    def yvertices(self):
        return self._yvertices

    @property
    def id(self):
        return self._id

    @property
    def visited(self):
        return self._visited

    @vertices.setter
    def vertices(self, xvertices, yvertices):
        self._xvertices = xvertices
        self._yvertices = yvertices

    def _set_bounds(self):
        # find and return bottom-left x,y and top-right x,y
        self.left = min(self._xvertices)
        self.right = max(self._xvertices)
        self.bottom = min(self._yvertices)
        self.top = max(self._yvertices)

    def get_bounds(self):
        # returns bottom-left, top-right in x,y fomat
        if "left" not in self.__dict__:
            self._set_bounds()
        return [self.left, self.bottom,
                self.right, self.top]

    def _set_center(self):
        self.center = ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def get_center(self):
        if "center" not in self.__dict__:
            self._set_center()
        return self.center

    def tostr(self):
        output = '<Region ID=\"{}\" visited=\"{}\" correct=\"{}\">\n'.format(self._id, str(self._visited), str(self._correct))
        output += '    <Attributes/>\n'
        output += '    <Vertices>\n'
        for x, y in self.vertices:
            output += f'        ["{x:.6f}\" Y=\"{y:.6f}\],\n'
        output += '    </Vertices>\n</Region>'

        return output

    def pmap_contours(self):
        '''
            generate contours from the current probability map
            update region's xs and ys at the end
        '''
        # generate a simplistic version of a neural network's pmap
        if "pmap" not in self.__dict__:
            self.create_pmap_from_contours()
        threshold = 0.5
        mask = np.where(self.pmap >= threshold, 1, 0).astype(np.bool)
        contours ,_ = cv2.findContours((255*mask).astype(np.uint8),
                                    cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0][:,0]
        self._xvertices = contours[:,0] + self.left
        self._yvertices = contours[:,1] + self.bottom

    def create_pmap_from_contours(self):
        '''
            given contours and no probability map,
            create an np.array() from the roi
            for each point in the contour, set the pmap[y,x] > threshold
            apply gaussian smoothing over the new pmap
        '''
        x = int(self.right - self.left) + 1
        y = int(self.top - self.bottom) + 1
        # print("(%d, %d)" % (x, y))
        pmap = np.zeros((x,y)) + 0.1
        # take the contours and add them to a 2d
        assert len(self._xvertices) == len(self._yvertices), "the lengths of vertices must be equal"
        for i in range(len(self._xvertices)):
            x = int(self.right - self._xvertices[i])
            y = int(self.top - self._yvertices[i])

            pmap[x,y] = 0.6

        return pmap

    def modify_pmap(self, plot_point):
        '''
            this function is responsible for differentiating between positive and negative clicks
            if a click is outside of the current contours, it is a positive click
        '''
        # transform global coordinate to local pmap coordinate
        start_point = (int(plot_point[0] - self.left), int(plot_point[1] - self.bottom))
        if "pmap" not in self.__dict__:
            self.create_pmap_from_contours()
            
        # TODO: if out of bounds, add rows/columns to pmap
        # self.enlarge_pmap()

        if _DEBUG:
            print(start_point)
        threshold = 0.5
        if self.point_inside(start_point):
            if _DEBUG:
                print("inside")
            self.quick_slice(start_point, threshold)
        else:
            if _DEBUG:
                print("outside")
            self.quick_append(start_point, threshold)

        self.pmap_contours()

    def point_inside(self, point):
        '''
            determine if a point is inside or on the contour
            else false
        '''
        x1,y1,x2,y2 = self.quick_points(point)
        return ((x1 > point[0] > x2) or (x2 > point[0]) > x1) and \
            ((y1 > point[0] > y2) or (y2 > point[0]) > y1)

    def enlarge_pmap(self, new_bounds, prev_bounds):
        '''
            supply new bounds, expand pmap as necessary
            # FIXME: be sure to modify the region's bounds.
        '''
        bottom, left, top, right = new_bounds
        # fill in row by row, starting at the top
        # for _ in range(self.pmap.shape[0]):
        #     for _ in range(self.pmap.shape[1]):
        #         pass

    def find_closest(self, point_index, pmap, threshold=0.5):
        '''
            find the index of the closest point to @from_point
        '''
        distance = lambda p1, p2: (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        # set to first available value in xs/ys
        min_distance = distance(point_index, (len(pmap) - point_index[0], len(pmap[0] - point_index[1])))
        min_index = (0, 0)
        for i in range(len(pmap)):
            for j in range(len(pmap[i])):
                this_distance = distance(point_index, (i,j))
                if  pmap[min_index] < threshold or \
                        (pmap[i,j] >= threshold and \
                        this_distance < min_distance):
                    min_distance = this_distance
                    min_index = (i,j)
        return min_index

    def quick_points(self, point):
        ''' find the 4 points in cardinal direction from a given point'''
        threshold = 0.5
        x = point[0]
        y = point[1]
        xs = self.pmap[y,:]
        ys = self.pmap[:, x]
        rightmost = leftmost = x
        for i in range(point[1] + 1, len(xs)):
            if xs[i] > threshold:
                rightmost = i
                break
        for i in range(point[1] - 1, -1, -1):
            if xs[i] > threshold:
                leftmost = i
                break

        topmost = bottommost = y
        for i in range(y + 1, len(ys)):
            if ys[i] > threshold:
                topmost = i
                break
        for i in range(y - 1, -1, -1):
            if ys[i] > threshold:
                bottommost = i
                break
        return leftmost, bottommost, rightmost, topmost

    def quick_append(self, point, threshold):
        # find tangential lines from given point
        min_point, max_point = self.find_tangents(point, threshold)
        # add lines to pmap
        where = np.where(self.pmap[:,-1] > threshold)[0]
        if len(where) > 0:
            min_point = (point[0], where[0])
        for pt in zip(*self.fill_line(point, min_point)):
            self.pmap[pt[::-1]] = threshold
        for pt in zip(*self.fill_line(point, max_point)):
            self.pmap[pt[::-1]] = threshold


    def quick_slice(self, point, threshold):
        '''
            point is expected to be inside of the natural contours of 
            the region's probability map
            map 4 points from cardinal directions
            find the 2 points that form the smallest area
            lower the values of the pmap for that area
            fill in coarse lines
        '''
        # compare point to corner
        y = int(self.right - self.left)
        x = int(self.top - self.bottom)
        candidates = [[0, 0], [x, 0], [0, y], [x, y]]
        area_by_points = lambda p1, p2: abs(p1[0] - p2[0]) * abs(p1[1] - p2[1])
        
        p = candidates[0]
        area = area_by_points(point, p)
        # minimize the area affected
        for cand in candidates:
            this_area = area_by_points(point, cand)
            if 0 < this_area < area:
                area = this_area
                p = cand

        # reduce probability in this area
        for y in range(min(p[0], point[0]), max(p[0], point[0]) + 1):
            for x in range(min(p[1], point[1]), max(p[1], point[1]) + 1):
                self.pmap[x,y] *= 0.66
        
        # set a new point 1 unit closer to the center
        x /= 2.0
        y /= 2.0
        point = list(point)
        point[0] += -1 if point[0] > x else 1
        point[1] += -1 if point[1] > y else 1
        # find the corresponding x and y from our new point
        # draw two new lines representing the new connection
        x1, y1, x2, y2 = self.quick_points(point)
        # setup minimal probability lines to ensure the contour remains enclosed
        x = x2 if abs(point[0] - x1) > abs(point[0] - x2) else x1
        for pt in zip(*self.fill_line(point, (x, point[1]))):
            self.pmap[pt[::-1]] = threshold

        y = y2 if abs(point[1] - y1) > abs(point[1] - y2) else y1
        for pt in zip(*self.fill_line(point, (point[0], y))):
            self.pmap[pt[::-1]] = threshold

    def find_tangents(self, from_point, threshold):
        '''
            find the two points which maximize and minimize 
            the slope of a line to the given point by index.
            Returns the two points found from this process
                 min_point, max_point
        '''
        slope = lambda p1, p2 : (p2[1] - p1[1]) / (p2[0] - p1[0]) if \
            (round(abs(p2[0] - p1[0]), 3) > 0.000) else self.top - self.bottom
        arctan = lambda p1, p2: np.arctan([slope(p1,p2)])[0]
        max_angle = arctan(from_point, from_point)
        min_point = max_point = from_point
        for y in range(len(self.pmap)):
            for x in range(len(self.pmap[y])):
                this_point = (x,y)
                this_angle = arctan(this_point, from_point)
                if self.pmap[max_point[::-1]] < threshold or \
                    (self.pmap[this_point[::-1]] >= threshold and \
                        -np.pi/4 >= np.sin(this_angle) > np.pi/4 and \
                        max_angle < this_angle): # -pi/2 0
                    max_angle = this_angle
                    max_point = this_point
        if _DEBUG:
            print("min", min_point, "\nmax", max_point)
                
        return min_point, max_point

    def fill_line(self, start_point, end_point, x_step = 1):
        '''
            returns two np.arrays representing the x's and y's
            from the start point to the end point
        '''
        if _DEBUG:
            print("start", start_point, "\nend", end_point)

        slope_coefficient = lambda p1, p2 : (p2[1] - p1[1]) / (p2[0] - p1[0]) if \
            (round(abs(p2[0] - p1[0]), 3) > 0.000) else self.top - self.bottom
        slope = slope_coefficient(start_point, end_point)
        
        if start_point[0] == end_point[0]: # vertical
            print("vertical")
            ys = np.arange(min(start_point[1],end_point[1]), max(start_point[1],end_point[1]), step=x_step, dtype=np.uint32)
            xs = np.zeros(ys.shape, dtype=np.uint32) + start_point[0]
        elif start_point[1] == end_point[1]: # horizontal
            print("horizontal")
            xs = np.arange(min(start_point[0],end_point[0]), max(start_point[0],end_point[0]), step=x_step, dtype=np.uint32)
            ys = np.zeros(xs.shape, dtype=np.uint32) + start_point[1]
        elif abs(start_point[0] - end_point[0]) > abs(start_point[1] - end_point[1]): # favoring more x values
            print("sloped, favoring x")
            y_intercept = lambda x, y, slope: y - slope * x
            slope_intercept = lambda x : slope * x + y_intercept(start_point[0], start_point[1], slope)
            xs = np.arange(min(start_point[0],end_point[0]), max(start_point[0],end_point[0]), step=x_step, dtype=np.uint32)
            ys = np.array(slope_intercept(xs), dtype=np.uint32)
        else: # favoring more y values
            print("sloped, favoring y")
            x_intercept = lambda x, y, slope: x - y / slope
            slope_intercept = lambda y : y / slope + x_intercept(start_point[0], start_point[1], slope)
            ys = np.arange(min(start_point[1],end_point[1]), max(start_point[1],end_point[1]), step=x_step, dtype=np.uint32)
            xs = np.array(slope_intercept(ys), dtype=np.uint32)
           
        exclude_out_of_bounds = np.where(ys >= self.top - self.bottom)[0]
        if exclude_out_of_bounds.shape[0] > 0:
            ys = np.delete(ys, exclude_out_of_bounds)
            xs = np.delete(xs, exclude_out_of_bounds)
        if _DEBUG:
            print("found xs", xs, "\nfound ys", ys)
        return xs, ys

if __name__ == "__main__":
    # test data
    data = np.array([[619, 161.200000],[619.200000, 161],[619.400000, 160.800000],[619.600000, 160.400000],[619.800000, 160.200000],[620, 160],[620.200000, 159.600000],[620.400000, 159.400000],[620.600000, 159],[620.800000, 158.800000],[621, 158.600000],[621, 158.400000],[621.200000, 158.200000],[621.200000, 158],[621.200000, 157.800000],[621.400000, 157.600000],[621.400000, 157.400000],[621.400000, 157.200000],[621.200000, 157],[621.200000, 156.800000],[621.200000, 156.600000],[621, 156.200000],[621, 156],[621, 155.800000],[620.800000, 155.600000],[620.800000, 155.200000],[620.600000, 154.800000],[620.400000, 154.600000],[620.400000, 154.400000],[620.400000, 154.200000],[620.200000, 153.400000],[620, 153],[620, 152.800000],[619.800000, 152.600000],[619.800000, 152.400000],[619.600000, 152],[619.400000, 151.800000],[619.400000, 151.600000],[619.200000, 151.400000],[619, 151.200000],[618.800000, 151],[618.600000, 150.800000],[618.400000, 150.800000],[618.200000, 150.600000],[618, 150.600000],[617.800000, 150.400000],[617.600000, 150.400000],[617.400000, 150.200000],[617.200000, 150],[617, 149.800000],[616.800000, 149.600000],[616.600000, 149.400000],[616.400000, 149.400000],[616, 149.200000],[615.600000, 149],[615.400000, 149],[615.200000, 149],[615, 149],[614.600000, 148.800000],[614.400000, 148.800000],[614.200000, 148.800000],[614, 149],[613.800000, 149],[613.600000, 149],[613.400000, 149],[613.200000, 149.200000],[612.800000, 149.400000],[612.600000, 149.600000],[612.400000, 149.600000],[612.200000, 149.800000],[612, 150],[611.800000, 150.200000],[611.400000, 150.400000],[611.200000, 150.600000],[611, 150.800000],[610.800000, 151],[610.600000, 151.200000],[610.600000, 151.400000],[610.400000, 151.600000],[610.200000, 151.800000],[610, 152],[609.800000, 152.200000],[609.600000, 152.600000],[609.400000, 152.800000],[609.200000, 153],[609, 153.200000],[608.800000, 153.400000],[608.600000, 153.800000],[608.400000, 154],[608.200000, 154.200000],[608, 154.400000],[607.800000, 154.600000],[607.800000, 154.800000],[607.600000, 155],[607.400000, 155.200000],[607.200000, 155.400000],[607.200000, 155.600000],[607, 155.800000],[607, 156],[607, 156.200000],[606.800000, 156.800000],[606.600000, 157],[606.600000, 157.200000],[606.600000, 157.400000],[606.400000, 157.600000],[606.200000, 157.800000],[606, 158],[605.800000, 158.200000],[605.600000, 158.600000],[605.600000, 158.800000],[605.600000, 159],[605.600000, 159.200000],[605.600000, 159.400000],[605.600000, 159.600000],[605.600000, 159.800000],[605.400000, 160.200000],[605.400000, 160.400000],[605.400000, 160.600000],[605.400000, 160.800000],[605.200000, 161],[605.200000, 161.400000],[605.200000, 161.600000],[605, 161.800000],[605, 162],[605, 162.200000],[605, 162.400000],[605, 162.600000],[605, 162.800000],[605, 163],[605.200000, 163.200000],[605.200000, 163.400000],[605.200000, 163.600000],[605.200000, 163.800000],[605.200000, 164],[605.200000, 164.400000],[605, 164.800000],[605, 165],[604.800000, 165.200000],[604.600000, 165.400000],[604.600000, 165.600000],[604.600000, 165.800000],[604.600000, 166],[604.400000, 166.400000],[604.400000, 166.600000],[604.400000, 166.800000],[604.400000, 167],[604.400000, 167.200000],[604.400000, 167.400000],[604.400000, 167.600000],[604.400000, 167.800000],[604.600000, 168],[604.800000, 168.200000],[604.800000, 168.400000],[604.800000, 168.600000],[604.800000, 168.800000],[604.800000, 169],[604.800000, 169.200000],[605, 169.400000],[605, 169.600000],[605, 169.800000],[605.200000, 170],[605.200000, 170.200000],[605.200000, 170.400000],[605.200000, 170.600000],[605.200000, 171],[605.200000, 171.200000],[605.400000, 171.600000],[605.400000, 172],[605.600000, 172.400000],[605.600000, 172.600000],[605.600000, 172.800000],[605.800000, 173],[606, 173.200000],[606.200000, 173.200000],[606.400000, 173.200000],[606.600000, 173.200000],[606.800000, 173.200000],[607, 173.200000],[607.200000, 173.200000],[607.400000, 173.200000],[607.600000, 173.200000],[608, 173.200000],[608.600000, 173.200000],[608.800000, 173.200000],[609, 173],[609.200000, 173],[609.400000, 173],[609.600000, 173],[609.800000, 173],[610, 173],[610.200000, 173],[610.600000, 173],[611.200000, 173],[611.400000, 173],[611.800000, 173],[612, 173],[612.200000, 173],[612.400000, 173],[613, 172.800000],[613.400000, 172.400000],[613.800000, 172],[614, 172],[614.200000, 172],[614.400000, 172],[614.800000, 171.800000],[615, 171.600000],[615.200000, 171.400000],[615.400000, 171.200000],[615.600000, 171.200000],[615.800000, 171],[616, 170.800000],[616.200000, 170.600000],[616.400000, 170.400000],[617, 170],[617.200000, 169.800000],[617.400000, 169.600000],[617.600000, 169.400000],[617.800000, 169.200000],[618, 169],[618, 168.800000],[618.200000, 168.400000],[618.200000, 168.200000],[618.400000, 168],[618.400000, 167.800000],[618.400000, 167.600000],[618.400000, 167.200000],[618.400000, 167],[618.400000, 166.800000],[618.600000, 166.400000],[618.600000, 166.200000],[618.600000, 166],[618.600000, 165.600000],[618.600000, 165.400000],[618.800000, 165],[618.800000, 164.800000],[618.800000, 164.600000],[618.800000, 164.400000],[618.800000, 164.200000],[618.800000, 164],[619, 163.800000],[619, 163.600000],[619, 163.400000],[619, 163.200000],[619, 163],[619.200000, 162.600000],[619.200000, 162.400000],[619.200000, 162.200000],[619.200000, 162],[619.200000, 161.800000],[619, 161.600000],[619, 161.400000],[619, 161.200000],[619.200000, 161]])
    region = Region(data[:, 0], data[:, 1])
    output = region.create_pmap_from_contours()
    region.pmap = output
    print(region.pmap.shape)
    output = region.modify_pmap((23+region.left, 1+region.bottom))
    # region.modify_pmap((region.right, region.top), 0)
    import matplotlib.pyplot as plt
    plt.imshow(region.pmap)
    plt.show()
