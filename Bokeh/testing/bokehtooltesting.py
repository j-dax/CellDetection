import xml.dom.minidom
import numpy as np

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, CustomAction, CustomJS
from bokeh.plotting import figure

output_file('tool.html')

#gets the annotations from xml file
def getRegions(xmlDoc):
    doc = xml.dom.minidom.parse(xmlDoc)
    doc = doc.documentElement
    annotations = doc.getElementsByTagName("Annotation")

    # region type: xml.dom.minidom.Element
    for annot in annotations:
        if annot.hasAttribute("Id"):
            regions = annot.getElementsByTagName("Regions")
            regions = regions[0].getElementsByTagName("Region")
    return regions

def getRegionBounds(region):
    #get x min and x max
    vertices = region.getElementsByTagName("Vertex")
    x = np.array([float(vertices[i].getAttribute("X")) for i in range(len(vertices))])
    y = np.array([float(vertices[i].getAttribute("Y")) for i in range(len(vertices))])

    min_x = np.amin(x)
    max_x = np.amax(x)
    min_y = np.amin(y)
    max_y = np.amax(y)

    return {'min_x':min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

regions = getRegions("C:/Users/Will/Desktop/LAHacks19/191-LeicaBiosystems/Static/TCGA-18-5592-01Z-00-DX1.xml")

i = 0
region = regions[i]

print(getRegionBounds(region))


# cant get the js to work
# not sure if it is the route or the js itself that is not working
jscode = '''
alert("Moved to next annotation")

$.ajax({
    type : "POST",
    url: "/updateplot",
    x_min: xmin,
    x_max: xmax,
    y_min: ymin,
    y_max: ymax,
    success: alert("???"),
    error: alert("broken")
});
'''

nextAnnotation = CustomAction(name = "Next Annotation",
                              icon="C:/Users/Will/Desktop/LAHacks19/191-LeicaBiosystems/Templates/img/anteater.jpeg",
                              callback = CustomJS(args=dict(xmin =  getRegionBounds(region)['min_x'], xmax =  getRegionBounds(region)['max_x'], ymin =  getRegionBounds(region)['min_y'], ymax = getRegionBounds(region)['max_y']), code=jscode))


from flask import Flask, render_template, request

#import requests

app = Flask(__name__)

@app.route("/updateplot", methods = ['POST', 'GET'])
def updateplot():
    if request.method == 'POST':
        x_min = request.args.get('x_min')
        x_max = request.args.get('x_max')
        y_min = request.args.get('y_min')
        y_max = request.args.get('y_max')

        figure.x_range.factors = (x_min, x_max)
        figure.y_range.factors =  (y_min, y_max)

approvalcode = '''
var answer = window.confirm("Keep Annotation?")
if (answer) {
    alert("Saved")
} else {
    var answer2 = window.confirm("Delete Annotation?")
    if (answer2) {
        alert("Deleted")
        y = xmlDoc.getElementsByTagName("Region")[0];
        xmlDoc.documentElement.removeChild(y);
    } else {
        alert("Annotation edit")
        //edit the annotation
        }
}
'''

yesnotool = CustomAction(name = "Annotation Approval",
                         icon="C:/Users/Will/Desktop/LAHacks19/191-LeicaBiosystems/Templates/img/anteater.jpeg",
                         callback = CustomJS(args=dict(), code=approvalcode))


# Things I need to do:
# Be able to know what annotation is currently being looked at
# When cancel is hit the ID of the current annotation should be deleted (if Okay nothing is changed)
# The next tool should move to and center the view around the next annotation region
# Maybe create a back tool which does what next does in reverse

#get the coordinates for the annotation currently looked at
#plug the coordinates into the JS script to:
#   -center the image around a region
#   -switch to next region
#   -delete that section of xml


source = ColumnDataSource(data=dict(x=[], y=[]))

plot = figure(x_range=(0, 10), y_range=(0, 10))
plot.title.text = "Drag to draw on the plot"
plot.line('x', 'y', source=source)
plot.add_tools(nextAnnotation, yesnotool)

show(plot)