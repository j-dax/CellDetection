from os import path
from random import random
import numpy as np
from skimage.io import imread

from bokeh.plotting import figure, show, output_file
from bokeh.io import curdoc, show
from bokeh.embed import components
from bokeh.models import Range1d, ColumnDataSource, MultiLine, Plot, Grid, LinearAxis, Patches
from bokeh.palettes import Spectral11
from bokeh.models.tools import WheelZoomTool

from Bokeh.bokeh_palette import create_region_palette, create_palette
from Bokeh.bokeh_tooling import add_tools
from Annotation.annotation import Annotation
from Annotation.region import Region
from bokeh.layouts import gridplot


def load_bokeh_in_new_page(html: 'os.path', image_name: 'os.path'):
    '''
        it is possible to save the generated html file by using output_file(<[filepath]/filename>)
    '''
    output_file(html)
    p = figure(x_range=(0, 1), y_range=(0, 1))
    p.image_url(url=[image_name], x=0, y=1, w=0.8, h=0.6)
    show(p)


def create_bokeh_components(image_name):
    # Create the main plot
    # bounds restricts the view from leaving a certain area. A tight view would be the image dimensions
    #   below, I've set them to allow additional panning/scrolling passed by 0.2 units
    #   with an initial view from 0,0 bottom-left to 1,1 top-right
    plot = figure(x_range=Range1d(0, 1, bounds=(-0.2, 2.2)), y_range=Range1d(0, 1, bounds=(-1.2, 1.2)))
    plot.image_url(url=[image_name], x=0, y=1, w=2, h=2)
    # TODO: interpolate the image on zoom (reload focused area with better quality)
    #   perhaps https://stackoverflow.com/questions/55545343/plot-image-with-interpolation-in-python-bokeh-like-matplotlib#answer-59458256
    # plot.toolbar.active_scroll = WheelZoomTool()

    # returns js-script, div
    return bokeh_from_figure(plot)
    

def multiplot(x_coords: "num[[]]", y_coords: "num[[]]"):
    ''' takes two 2D arrays of x coordinates and y coordinates
            and returns a figure with the representation
    '''
    # sets unique coloring for each region
    palette = create_palette(len(x_coords))

    plot = figure(width=500, height=300)
    plot.multi_line(xs=x_coords, ys=y_coords, line_color=palette, line_width=5)
    return plot


def build_multiplot(annotations: [Annotation], tool_icon_dir, imagepath, img_dims: list = None):
    '''
        Takes annotations and renders their contours
        on top of a corresponding image
    '''
    plot = None

    x_coords = []
    y_coords = []
    lineNames = []
    palette = []

    if not img_dims:
        image = imread(imagepath)
        img_dims = image.shape[:2]

    min_x, min_y, max_x, max_y = annotations[0].get_bounds()
    for annot in annotations:
        x1, y1, x2, y2 = annot.get_bounds()
        min_x = min(x1, min_x)
        min_y = min(y1, min_y)
        max_x = max(x2, max_x)
        max_y = max(y2, max_y)

        x_coords.extend(annot.xvertices)
        y_coords.extend(annot.yvertices)
        for i in range(len(annot.regions)):
            lineNames.append("Annotation{}".format(i))

        palette += create_region_palette(annot.regions)
        
        # define plot dimensions, restrict view using bounds=(...)
        annot.plot = figure(
                x_range=Range1d(0, img_dims[0], bounds=(0, img_dims[1])), 
                y_range=Range1d(0, img_dims[1], bounds=(0, img_dims[0])),
                toolbar_location="above")
        plot = annot.plot
    
    # bokeh plotting flips the y-axis, invert to match the image
    x_coords = np.array(x_coords)
    y_coords = max_y - np.array(y_coords) + min_y
    
    # order matters, post image, write overlay after image
    ## testing raw image input
    # image = cv2.imread(imagepath)
    # imageCDS = ColumnDataSource(data=dict(image=image))
    # plot.image(image='image', x=0, y=img_dims[0], dw=img_dims[1], dh=img_dims[0], source=imageCDS)
    # FIXME: currently disappears whenever the plot automoves through our tooling
    plot.image_url(url=dict(value=imagepath),
                    x=0, y=img_dims[0], w=img_dims[1], h=img_dims[0])
    
    # add each line, one at a time, so that they can be referenced later by name
    for i in range(len(x_coords)):
        plot.patch(x = x_coords[i], y = y_coords[i], line_color = palette[i], line_width = 5, alpha = .9, fill_alpha = 0.0 , name = lineNames[i])
        #plot.add(source, glyph)
    # output_file('patches.html')
    # show(plot)
    add_tools(plot, tool_icon_dir)

    return plot


def bokeh_image():
    '''
        testing image loading
    '''
    min_x = min_y = 0
    max_x = max_y = 1
    plot = figure(x_range=Range1d(min_x, max_x, bounds=(min_x, max_x)),
                y_range=Range1d(min_y, max_y, bounds=(min_y, max_y)),
                toolbar_location="above")
    plot.image_url(url=dict(value='http://127.0.0.1:5000/static/img.jpg'), x=min_x, y=max_y, w=max_x, h=max_y)
    return plot


def bokeh_from_figure(plot):
    ''' takes a bokeh plot and return a js-script and div'''
    plot.sizing_mode = "scale_both"
    plot.match_aspect = True
    return components(plot)