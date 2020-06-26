"""
Created by Lanting Li.
Jan 26, 2020

This class aims to help with save and access the information of one annotation.
"""
import xml.dom.minidom as minidom 
from .region import Region


class Annotation:
    def __init__(self, regions: [Region], id: int = 0, name: str = "",
                 visible: bool = True, selected: bool = False):
        self._id = id
        self._name = name
        self._selected = selected
        self._visible = visible
        self._selected = selected
        self._regions = regions

    def __len__(self):
        return len(self._regions)

    @property
    def regions(self):
        return self._regions

    @regions.setter
    def regions(self, regions):
        self._regions = regions

    @property 
    def xvertices(self):
        x_verts = []
        for region in self._regions:
            x_verts.append(region.xvertices)
        return x_verts

    @property
    def id(self):
        return self._id

    @property
    def yvertices(self):
        y_verts = []
        for region in self._regions:
            y_verts.append(region.yvertices)
        return y_verts

    def _set_bounds(self):
        # find and return bottom-left x,y and top-right x,y
        self.left = min(self.regions[0]._xvertices)
        self.right = max(self.regions[0]._xvertices)
        self.top = max(self.regions[0]._yvertices)
        self.bottom = min(self.regions[0]._yvertices)
        for region in self._regions:
            min_x, min_y, max_x, max_y = region.get_bounds()
            self.left = min(min_x, self.left)
            self.bottom = min(min_y, self.bottom)
            self.right = max(max_x, self.right)
            self.top = max(max_y, self.top)

    def update_bounds(self, min_x, min_y, max_x, max_y):
        # Use for update bounds with input from MS-COCo box
        self.left = min_x
        self.bottom = min_y
        self.right = max_x
        self.top = max_y

    def get_bounds(self):
        # returns bottom-left, top-right in x,y fomat
        if "left" not in self.__dict__:
            self._set_bounds()
        return (self.left, self.bottom,
                self.right, self.top)

    def _set_center(self):
        self.center = ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def get_center(self):
        if "center" not in self.__dict__:
            self._set_center()
        return self.center

    def all_bounds(self):
        regions = []
        for region in self._regions:
            regions.append({'bounds': region.get_bounds()})
        return regions

    def region_bounds(self, index: int):
        _, min_y, _, max_y = list(self.get_bounds())
        region_bounds = list(self._regions[index].get_bounds())
        # if the bounds come from an np.array,
        #   they will not be encoded properly for the api
        #   cast them to python ints
        bounds = [0,0,0,0]
        bounds[0] = int(region_bounds[0])
        bounds[1] = int(max_y - region_bounds[1] + min_y)
        bounds[2] = int(region_bounds[2])
        bounds[3] = int(max_y - region_bounds[3] + min_y)
        return {'bounds': bounds}

    def modify_region(self, index, point):
        _, min_y, _, max_y = list(self.get_bounds())
        point = list(point)
        point[1] = max_y - point[1] + min_y
        self._regions[index].modify_pmap(point)

    def delete_region(self, index: int):
        del self._regions[index]

    def __str__(self):
        output = '<Annotation Id=\"{}\" Visible=\"{}\" Selected=\"{}\">\n'.format(str(self._id), str(self._visible), str(self._selected))
        output += '    <Regions>\n'
        for region in self._regions:
            region_output = region.tostr()
            region_output = "\n".join(['        '+line for line in region_output.split("\n")])
            output += region_output
        output += "\n\t</Regions>\n</Annotation>"
        return output


def parse_annotation(annot) -> "Annotation":
    """
    Input annotation information in xml.dom.minidom.Element format, and parse and convert it to Annotation class.
    :param annot: Ought to be xml.dom.minidom.Element
    :return: an Annotation object
    """
    if not isinstance(annot, minidom.Element):
        raise(TypeError("Input annot information is not in xml.dom.minidom.Element format."))

    attributes = ['Id', 'Name', 'ReadOnly', 'NameReadOnly', 'LineColorReadOnly', 'Incremental', 'Type', 'LineColor',
                  'Visible', 'Selected', 'MarkupImagePath', 'MacroName']
    attr_list = []
    # Check if this annotation has all required attribute information.
    # If not, raise AttributeError with related information.
    for attribute in attributes:
        if annot.hasAttribute(attribute):
            attr_list.append(annot.getAttribute(attribute))
        else:
            raise(AttributeError("This annotation has no {} information.".format(attribute)))

    annotation = Annotation(*attr_list)

    return annotation