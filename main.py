'''
    This file is the access point where Flask sets up routing

    Run the flask server with 
        python main.py
    
        this command runs the flask server in debug mode, so it will reload whenever a change is detected

    Current limitations:
        Too much functionality is server-side,
            this prevents the client from updating 
'''

from bokeh.plotting import figure

# access to api calls
from flask import Flask, render_template, request, session
from flask_cors import CORS
from flask import send_from_directory, jsonify, redirect, url_for, send_file
from flask.helpers import flash
from werkzeug.utils import secure_filename
from os import path, getcwd
from re import search
import sys

#creates paths to necessary files
sys.path.append(path.join(getcwd(), "Bokeh"))
sys.path.append(path.join(getcwd(), "Annotation"))
sys.path.append(path.join(getcwd(), "mask"))

# flask configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'tif', 'tiff', 'png'}
app = Flask(__name__)
UPLOAD_FOLDER = path.join(path.dirname(__file__), 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# FIXME: remove key from this file, load from private (not in repo) file
app.secret_key = b'i\\Fkf/co0Rg$S"Wnmk.TiS_5'
CORS(app)

# FIXME: move annotation management to client side
# Store in session
annotations = None
current_region = current_annotation = -1

network = None

# TODO: adding magic byte checking to mitigate risk of false extensions
def file_allowed(filename: str) -> bool:
    if filename == "":
        return False
    regx = search(r'(.*)(\.)(.*)', filename)
    extension = regx.group(3)
    return str.lower(extension) in ALLOWED_EXTENSIONS


@app.route("/")
def default():
    '''
        a redirect for when a user navigates to 127.0.0.1:5000/
    '''
    return redirect(url_for('test'))


#allows user to select files to upload to application for annotation
@app.route('/upload', methods=["POST", "GET"])
def upload():
    '''
        use POST to upload a file to the server for processing
        only allowing certain extensions.
    '''
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == "":
            flash('No selected file')
            return redirect(request.url)

        # File is allowed to be processed by server
        if file_allowed(file.filename):
            session["filename"] = secure_filename(file.filename)
            filepath = path.join(app.config['UPLOAD_FOLDER'], session["filename"])
            file.save(filepath)
            # begin annotating image
            return redirect(url_for("annotate", filename=session["filename"]))
            # return redirect(url_for('uploaded', filename=filename))
        else:
            flash('Invalid file type')
        # basepath = os.path.dirname(__file__)
    return render_template("upload.html")


@app.route('/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/test")
def test():
    '''
        renders a bokeh plot using XML-parsing of annotations
    '''
    from Annotation.XML import parse_XML
    from Bokeh.flask_bokeh import bokeh_from_figure, build_multiplot    
    global annotations, current_region, current_annotation
    current_region = 0
    current_annotation = 0
    
    filename = 'TCGA-18-5592-01Z-00-DX1.png'
    session["filename"] = filename
    image_path = lambda base, fn: path.join(base, "static/%s" % fn)

    annotation_xml = 'TCGA-18-5592-01Z-00-DX1.xml'
    annotations = parse_XML(image_path(getcwd(), annotation_xml))
    STATIC_PATH = "http://127.0.0.1:5000/"

    TOOL_IMAGE_DIR = path.join(getcwd(), "templates/img")
    plot = build_multiplot(annotations, TOOL_IMAGE_DIR, image_path(STATIC_PATH, filename))
    script, div = bokeh_from_figure(plot)
    return render_template("base/bokeh.html", script=script, div=div)


@app.route("/updated")
def updated():
    '''
        Uses session tracking to match the user's current file
        renders the updated bokeh plot.
    '''
    from Annotation.XML import parse_XML
    from Bokeh.flask_bokeh import bokeh_from_figure, build_multiplot
    global annotations
    TOOL_IMAGE_DIR = path.join(getcwd(), 'templates/img')
    
    if "filename" not in session:
        session["filename"] = 'TCGA-18-5592-01Z-00-DX1.png'
    filename = session["filename"]

    image_path = lambda base, fn: path.join(base, "static/%s" % fn)

    IMAGE_PATH = image_path("http://127.0.0.1:5000/", filename)
    print(IMAGE_PATH)
    plot = build_multiplot(annotations, TOOL_IMAGE_DIR, IMAGE_PATH)
    script, div = bokeh_from_figure(plot)
    return render_template("base/bokeh.html", script=script, div=div)

@app.route("/annotate")
def annotate():
    '''
        Uses session tracking to reference the client's file
        Create a bokeh plot that uses an image and generates contours 
            from a set of probability maps
        Ideally, the neural network would only be initialized once
        makeRegion(...) creates a region with an additional class variable to hold the
            probability map
    '''
    if "filename" not in session:
        session["filename"] = 'TCGA-18-5592-01Z-00-DX1.png'
    filename = session["filename"]

    from img_rcnn import load_image
    from Bokeh.flask_bokeh import bokeh_from_figure, build_multiplot
    from Annotation.region import Region, makeRegion
    from Annotation.annotation import Annotation
    
    global annotations, current_region, current_annotation, network
    if not network:
        initialize_network()
    current_region = 0
    current_annotation = 0

    image_path = lambda base, fn: path.join(base, "static/{}".format(fn))
    IMAGE_PATH = image_path(getcwd(), filename)
    TOOL_IMAGE_DIR = path.join(getcwd(), "templates/img")

    image = load_image(IMAGE_PATH)
    
    results = network.get_results([image])[0]
    annotations = []
    regions = []
    pmap = results["pmaps"]
    roi = results["rois"]
    for i in range(len(pmap)):
        regions.append(makeRegion(pmap[i], roi[i]))
        
    annotations.append(Annotation(regions))
    # the image cannot be accessed locally due to cross-origin policy,
    # send it back through a static link from the server
    IMAGE_PATH = image_path("http://127.0.0.1:5000/", filename)
    script, div = bokeh_from_figure(build_multiplot(annotations, TOOL_IMAGE_DIR, IMAGE_PATH, image.shape))
    return render_template("base/bokeh.html", script=script, div=div)


@app.route("/net")
def net():
    '''
        Create a bokeh plot that uses an image and generates contours 
            from a set of probability maps
        makeRegion(...) creates a region with an additional class variable to hold the
            probability map
        Known bug: neural network fails to load a second image
            e.g. accessing this page again causes the tf to throw an error 
            >>> tensorflow.python.framework.errors_impl.InvalidArgumentError: Tensor input_image:0,
                specified in either feed_devices or fetch_devices was not found in the Graph
    '''
    import img_rcnn
    from Bokeh.flask_bokeh import bokeh_from_figure, build_multiplot
    from Annotation.region import Region, makeRegion
    from Annotation.annotation import Annotation
    if "filename" not in session:
        session["filename"] = 'TCGA-18-5592-01Z-00-DX1.png'
    filename = session["filename"]
    # Ideally, annotations,current_region,current_annotation would be client-side variables
    #       Addition option, store them as session attributes from flask
    global annotations, current_region, current_annotation, network
    if not network:
        initialize_network()
    current_region = 0
    current_annotation = 0

    IMAGE_PATH = path.join(getcwd(), "static/TCGA-18-5592-01Z-00-DX1.png")
    TOOL_IMAGE_DIR = path.join(getcwd(), "templates/img")

    image = img_rcnn.load_image(IMAGE_PATH)
    
    results = network.get_results([image])[0]
    annotations = []
    regions = []
    pmap = results["pmaps"]
    roi = results["rois"]
    for i in range(len(pmap)):
        regions.append(makeRegion(pmap[i], roi[i]))
        
    annotations.append(Annotation(regions))
    # the image cannot be accessed locally due to cross-origin policy,
    # send it back through a static link from the server
    IMAGE_PATH = "http://127.0.0.1:5000/static/TCGA-18-5592-01Z-00-DX1.png"
    script, div = bokeh_from_figure(build_multiplot(annotations, TOOL_IMAGE_DIR, IMAGE_PATH, image.shape))
    return render_template("base/bokeh.html", script=script, div=div)


@app.route("/bbox")
def bbox():
    '''
        similar to /net
        instead of using contours, shows bounding boxes around targets
    '''
    from img_rcnn import load_image
    from Bokeh.flask_bokeh import bokeh_from_figure, build_multiplot
    from Annotation.region import Region
    from Annotation.annotation import Annotation
    
    global annotations, current_region, current_annotation, network
    current_region = 0
    current_annotation = 0
    if "filename" not in session:
        session["filename"] = 'TCGA-18-5592-01Z-00-DX1.png'
    filename = session["filename"]
    IMAGE_PATH = path.join(getcwd(), "static/img.jpg")
    TOOL_IMAGE_DIR = path.join(getcwd(), "templates/img")

    if not network:
        initialize_network()
    
    image = load_image(IMAGE_PATH)
    results = network.get_results([image])[0]

    regions = []
    for roi in results["rois"]:
        bottom, left, top, right = roi
        # set up points that represent this bounding box
        # FIXME: coords are shifted down from the neural net
        #        will look at it again when contours are complete
        xs = [left, left, right, right, left]
        ys = [top, bottom, bottom, top, top]
        reg = Region(xs,ys)
        regions.append(reg)
    annotations = [Annotation(regions)]

    # the image cannot be accessed locally due to cross-origin policy,
    # send it back through a static link from the server
    IMAGE_PATH = "http://127.0.0.1:5000/static/img.jpg"
    script, div = bokeh_from_figure(build_multiplot(annotations, TOOL_IMAGE_DIR, IMAGE_PATH, image.shape))
    return render_template("base/bokeh.html", script=script, div=div)


@app.route("/api/v0.1/region/<annot_index>/<region_index>")
def region_bounds(annot_index, region_index):
    '''
        testing interface to see a given region
    '''
    return jsonify(annotations[int(annot_index)].region_bounds(int(region_index)))


def get_all_regions():
    '''
        create a dict that will be passed as a JSON object
    '''
    annot_list = []
    if annotations is not None:
        for annot in annotations:
            annot_list.append({'id': annot.id,'regions':annot.all_bounds()})
    return {'annotations': annot_list}


@app.route("/api/v0.1/region/<annot_index>")
def all_regions_in_annotation():
    '''
        testing interface to see all regions
    '''
    return jsonify(get_all_regions)


@app.route("/api/v0.1/approve/<boolean>")
def approve(boolean: str):
    '''
        interface to update the current region as approved/rejected
    '''
    from Bokeh.bokeh_palette import update_color
    global current_region, current_annotation, annotations
    region = annotations[current_annotation].regions[current_region]
    if boolean == "true": # http passes this as a string
        region.set_correct(True)
    else:
        region.set_correct(False)
    region.set_visited(True)
    
    update_color(annotations[current_annotation].plot, region, current_region)
    return jsonify({"index": current_region, "bool": boolean})


@app.route("/api/v0.1/prev")
def prev():
    '''
        interface to change the bokeh view port
        to the hover over the previous Region
    '''
    global current_region, current_annotation
    if (current_annotation == -1 or current_region == -1):
        return jsonify({'index': -1})

    current_region = (current_region - 1)%len(annotations[current_annotation])
    return region_bounds(current_annotation, current_region)


@app.route("/api/v0.1/next")
def next():
    '''
        interface to change the bokeh view port
        to the hover over the next Region
    '''
    global current_region, current_annotation
    if (current_annotation == -1 or current_region == -1):
        return jsonify({'index': -1})
        
    current_region = (current_region + 1) % len(annotations[current_annotation])
    return region_bounds(current_annotation, current_region)


@app.route("/api/v0.1/current")
def current():
    # unused
    return jsonify({'index': current_region})


@app.route("/api/v0.1/saveXML")
def saveXML():
    '''
        interface to send client the xml associated with this 
            Annotation and [Region] set.
    '''
    from Annotation.XML import annotations_tostr
    global annotations
    if annotations:
        # TODO: generate a random file name
        filename = "temp.xml"
        filepath = app.config['UPLOAD_FOLDER'] + "/" + filename
        xml = annotations_tostr(annotations)
        with open(filepath, "w") as f:
            f.write(xml)
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({"status": 304, "reason": "annotations not allocated"})


@app.route("/api/v0.1/sendClick/<x>/<y>")    
def sendClick(x, y):
    '''
        interface to receive client clicks
        and update the focused region
    '''
    global annotations, current_annotation, current_region
    if annotations:
        try:
            x = int(float(x))
            y = int(float(y))
            annotations[current_annotation].modify_region(current_region, (x,y))
            return jsonify({"status": 200})
        except:
            return jsonify({"status": 500, "message": "modify_pmap has failed"})
    return jsonify({"status": 400, "message": "annotations not allocated"})


def initialize_network():
    '''
        load the neural network into a global variable
            bug:    the network seems to fail whenever it processes
                    a second image
    '''
    from img_rcnn import initialize_network
    LOG_DIR = path.join(getcwd(), "logs")
    H5_PATH = path.join(getcwd(), "nucleus.h5")
    global network
    network = initialize_network(H5_PATH, LOG_DIR)

if __name__ == '__main__':
    app.run(debug=True)
