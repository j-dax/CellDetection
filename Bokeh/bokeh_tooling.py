from bokeh.models import CustomAction, CustomJS, TapTool, PolyEditTool, PolyDrawTool
from bokeh.plotting import figure

from os import path

def create_nav_tool(plot: figure, name: str, http: str, iconpath: str):
    '''
        This tool allows the user to move the view port between
            region contours
        plot is passed to add the tool to the Bokeh div
    '''
    tool_js = '''
    var region_min_x = 0, region_min_y = 0,
        region_max_x = 0, region_max_y = 0;
    
    fetch("%s")
    .then((response)=>{
        return response.json();
    }).then((data)=>{
        var padding = 1;
        region_min_x = data["bounds"][0];
        region_min_y = data["bounds"][1];
        region_max_x = data["bounds"][2];
        region_max_y = data["bounds"][3];
        
        var x_start = region_min_x - padding;
        var x_end = region_max_x + padding;
        var y_start = region_min_y + padding;
        var y_end = region_max_y - padding;
        
        x_range.setv({"start": x_start, "end": x_end});
        y_range.setv({"start": y_start, "end": y_end});
    }).catch((err)=> console.log(err));
    ''' % (http)
    return CustomAction(name = name, icon = iconpath,
        callback = CustomJS(args=dict(x_range = plot.x_range, y_range = plot.y_range),
                            code=tool_js))


def create_approval_tool(plot: figure, name: str, urlApprove: str,
                        urlNext: str, iconpath: str):
    '''
        This tool allows the user to approve/reject the focused region
            following action, move to the next contour
        plot is passed to add the tool to the Bokeh div
    '''
    tool_js = """
    // calls approve/reject
    fetch("%s")
    .then((response)=>{
        return response.json();
    }).catch((err)=> console.log(err));

    var region_min_x = 0, region_min_y = 0,
        region_max_x = 0, region_max_y = 0,
        window_max_y = 0, window_min_y = 0;

    fetch("%s")
    .then((response)=>{
        return response.json();
    }).then((data)=>{
        var padding = 1;
        region_min_x = data["bounds"][0];
        region_min_y = data["bounds"][1];
        region_max_x = data["bounds"][2];
        region_max_y = data["bounds"][3];
        
        var x_start = region_min_x - padding;
        var x_end = region_max_x + padding;
        var y_start = region_min_y + padding;
        var y_end = region_max_y - padding;

        x_range.setv({"start": x_start, "end": x_end});
        y_range.setv({"start": y_start, "end": y_end});
    }).catch((err)=> console.log(err));
    """ % (urlApprove, urlNext)
    return CustomAction(name = name, icon = iconpath,
        callback = CustomJS(args=dict(x_range = plot.x_range, y_range = plot.y_range),
                            code=tool_js))


def save_xml_tool(name, http, iconpath):
    '''
        This tool allows the user to download the current XML-representation
            of the Annotation/[Regions]
        plot is passed to add the tool to the Bokeh div
    '''
    tool_js = '''
    console.log("save button clicked");
    // calling this url is enough to make the link download
    var http = "%s";
    console.log(http);
    window.open(http, "_blank");
    ''' % (http)
    return CustomAction(name = name, icon = iconpath,
        callback = CustomJS(args=dict(),
            code=tool_js))


def create_navigate_to_updated_button(image):
    '''
        A tool to reload the plot with updated contours/colors
        plot is passed to add the tool to the Bokeh div
    '''
    tool_js = '''
    window.open("http://localhost:5000/updated");
    '''
    return CustomAction(name= 'updated', icon = image, callback = CustomJS(args=dict(), code = tool_js))


def create_poly_edit_tool(plot: figure):
    p1 = plot.patches([], [], fill_alpha=0.4)
    c1 = plot.circle([], [], size=10, color='red')
    return PolyEditTool(renderers = [p1], vertex_renderer = c1)


def create_poly_draw_tool(plot: figure):
    p1 = plot.patches([], [], fill_alpha=0.4)
    return PolyDrawTool(renderers = [p1])

def tap_tool_callback(plot: figure):
    '''
        This callback is always active.
        
        Clicking on the plot updates contours
            see Region.modify_pmap

        plot is passed to add the tool to the Bokeh div
    '''
    callback = CustomJS(args=dict(), code="""
        var target = "api/v0.1/sendClick/" + cb_obj.x.toString() + "/" + cb_obj.y.toString();
        console.log(target);
        fetch(target)
            .catch(err=>console.log(err))
    """)
    plot.js_on_event('tap', callback)


def add_tools(plot: figure, img_directory: path):
    '''
        configure which additional tools are shown on the Bokeh page
    '''
    api = "http://localhost:5000/api/v0.1/"
    next_path = path.join(img_directory, "Right.png")
    prev_path = path.join(img_directory, "Left.png")
    approve_path = path.join(img_directory, "Check.png")
    reject_path = path.join(img_directory, "Cross.png")
    update_path = path.join(img_directory, "eye.jpeg")
    plot.add_tools(
        create_navigate_to_updated_button(update_path),
        create_nav_tool(plot, "Previous Annotation", api + "prev", prev_path),
        create_nav_tool(plot, "Next Annotation", api + "next", next_path), 
        create_approval_tool(plot, "Approval Tool", api + 'approve/true', api + "next", approve_path),
        create_approval_tool(plot, "Reject Tool", api + 'approve/false', api + "next", reject_path),
        create_poly_edit_tool(plot),
        create_poly_draw_tool(plot)
    )

    tap_tool_callback(plot)