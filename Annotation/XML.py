import xml.dom.minidom
import numpy as np
from .annotation import Annotation
from .region import Region


def parse_XML(xmlfile):
    doc = xml.dom.minidom.parse(xmlfile)
    doc = doc.documentElement
    annotations = doc.getElementsByTagName("Annotation")

    # region type: xml.dom.minidom.Element
    found_annotations = []
    for annot in annotations:
        # if annot.hasAttribute("Id"):
        #     print(annot.getAttribute("Id"))
        found_regions = []
        regions = annot.getElementsByTagName("Regions")
        regions = regions[0].getElementsByTagName("Region")
        for region in regions:
            # print(len(region.getElementsByTagName("Vertex")))
            # if region.hasAttribute("Id"):
            #     print("Region #{}:".format(region.getAttribute("Id")))
            #     print("    Length: {}".format(region.getAttribute("Length")))
            #     print("    Zoom: {}".format(region.getAttribute("Zoom")))

            vertices = region.getElementsByTagName("Vertex")
            x = np.array([float(vertices[i].getAttribute("X")) for i in range(len(vertices))])
            y = np.array([float(vertices[i].getAttribute("Y")) for i in range(len(vertices))])
            region_id = region.getElementsByTagName("Id")
            length = region.getElementsByTagName("Length")
            text = region.getElementsByTagName("Text")
            # print(x)
            # print(y)
            found_regions.append(Region(id=region_id, text=text, xvertices=x, yvertices=y))
        annotation_id = annotations[0].getElementsByTagName("Id")
        annotation_name = annotations[0].getElementsByTagName("name")
        found_annotations.append(Annotation(id=annotation_id, name=annotation_name, regions=found_regions))
    return found_annotations

def write_XML(annotations, file):
    with open(file, 'w') as w:
        w.write(str(annotations))

def annotations_tostr(annotations):
    output = '<Annotations>\n'
    for annot in annotations:
        annot_output = str(annot)
        output += '\n'.join(['    '+line for line in annot_output.split("\n")])
    output += '\n</Annotations>'
    return output


if __name__ == "__main__":
    # Change the following path to the example xml file in Static
    found_annotations = parse_XML('/Users/lantingli/Desktop/191-LeicaBiosystems/Static/TCGA-18-5592-01Z-00-DX1.xml')
    write_XML(found_annotations, "/Users/lantingli/Desktop/191-LeicaBiosystems/Static/output.xml")