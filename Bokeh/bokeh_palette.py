'''
    Color management for Bokeh
'''

from Annotation.region import Region

def create_palette(num_colors: int) -> "palette":
    # returns a palette with exactly the number of colors asked for
    palette = []
    while len(palette) < num_colors:
        palette.append('#808080')
    return palette[:num_colors]


def create_region_palette(regions: [Region]):
    palette = []
    for region in regions:
        palette.append(region_color(region))
    return palette


def region_color(region):
    green = '#93c47d'
    red = '#cc4125'
    yellow = '#ffd966'
    grey = '#999999'
    if not region.visited:
        return yellow
    elif not region.correct:
        return red
    elif (region.visited and region.correct):
        return green
    else: # this shouldn't happen and indicates a problem with the region
        return grey


def update_color(plot, region, index):
    plot.select("Annotation{}".format(index)).glyph.line_color = region_color(region)
