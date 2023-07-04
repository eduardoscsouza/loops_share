import os
import sys
import shutil
from glob import glob
from math import floor
from multiprocessing import Pool
from xml.etree import ElementTree

import numpy as np
import imageio.v3 as imageio
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon, Point
from skimage.segmentation import flood_fill

sys.path.append(os.path.join(os.environ["LOOPS_PATH"], "src"))
from utils import *



LINE_WIDTH = 3
LINE_COLOR = 255

NEIGHBOURS_SIZE = 5

ACCEPTED_LABELERS = ["anna", "isabel", "cristina"]
UNNAMED_LABELER = "unnamed"
UNION_LABELER = "union"
INTERSECTION_LABELER = "intersection"



def get_xy(pos):
    return int(pos.attrib["X"]), int(pos.attrib["Y"])

def get_ij(pos):
    return int(pos.attrib["Y"]), int(pos.attrib["X"])

def draw_region(mask, region, line_color, line_witdh):
    drawn_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(drawn_mask)
    for edge, next_edge in zip(region[:-1], region[1:]):
        draw.line((*get_xy(edge), *get_xy(next_edge)), fill=line_color, width=line_witdh)
    draw.line((*get_xy(region[-1]), *get_xy(region[0])), fill=line_color, width=line_witdh)

    return np.array(drawn_mask)

def fill_regions(mask, edges, polys, fill_color, neighbours_size):
    edges = np.array(edges)
    neighbours = get_array_of_indices((neighbours_size, neighbours_size)) - floor(neighbours_size / 2)

    init_poss = np.unique(np.concatenate([edge + neighbours for edge in edges], axis=0), axis=0)
    init_poss = [tuple(pos) for pos in init_poss if (all(pos >= 0) and all(pos < mask.shape))]
    init_poss = [pos for pos in init_poss if ((mask[pos] == 0) and any([poly.contains(Point((pos[1], pos[0]))) for poly in polys]))]
    for pos in init_poss:
        if mask[pos] == 0:
            mask = flood_fill(mask, pos, fill_color)

    return mask

def get_mask(regions, shape, dtype):
    polys, edges = [], []
    mask = np.zeros(shape, dtype=dtype)
    for region in regions:
        mask = draw_region(mask, region, LINE_COLOR, LINE_WIDTH)
        polys += [Polygon([get_xy(pos) for pos in region])]
        edges += [get_ij(pos) for pos in region]

    np.putmask(mask, mask>(LINE_COLOR/2.0), LINE_COLOR)
    np.putmask(mask, mask<=(LINE_COLOR/2.0), 0)
    mask = fill_regions(mask, edges, polys, LINE_COLOR, NEIGHBOURS_SIZE)

    return mask



def get_xml_masks(xml, img_path):
    masks = {}

    try:
        root = ElementTree.parse(xml).getroot()
    except ElementTree.ParseError:
        print_error(xml, "XML Badly Formatted")
        return masks

    img = imageio.imread(img_path)
    shape, dtype = img.shape[:-1], img.dtype
    del img

    for labeler in root:
        labeler_name = labeler.attrib["Name"].lower().strip()
        if (labeler_name == ""):
            labeler_name = UNNAMED_LABELER
        elif labeler_name not in ACCEPTED_LABELERS:
            print_error(xml, "Unrecognized Labeler \"{}\"".format(labeler_name))
            continue

        if labeler_name in masks.keys():
            print_error(xml, "Multiple Masks for Labeler \"{}\". Ignoring Latter Masks".format(labeler_name))
            continue

        try:
            assert labeler.tag == "Annotation"
            assert labeler[0].tag == "Attributes"
            assert labeler[1].tag == "Regions"
            assert labeler[1][0].tag == "RegionAttributeHeaders"
        except AssertionError:
            print_error(xml, "Wrong Labeler Format")
            continue

        if any([(attributes.attrib["Name"].lower().strip() == "normal") for attributes in labeler[0]]):
            print_error(xml, "Labeler \"{}\" Classified as \"Normal\". Ignoring this Labeler".format(labeler_name))
            continue

        regions = []
        for region in labeler[1][1:]:
            try:
                assert region.tag == "Region"
                assert region[0].tag == "Attributes"
                assert region[1].tag == "Vertices"
            except AssertionError:
                print_error(xml, "Wrong Region Format")
                continue

            if any([(attributes.attrib["Name"].lower().strip() == "normal") for attributes in region[0]]):
                print_error(xml, "Region in Labeler \"{}\" Classified as \"Normal\". Ignoring this Region".format(labeler_name))
                continue

            regions += [region[1]]

        if len(regions) > 0:
            masks[labeler_name] = get_mask(regions, shape, dtype)
        else:
            print_error(xml, "Labeler \"{}\" Found in XML Without Regions".format(labeler_name))

    if (UNNAMED_LABELER in masks.keys()) and (list(masks.keys()) != [UNNAMED_LABELER]):
        print_error(xml, "XML Has Both Named and Unnamed Labelers")

    if len(masks) > 0:
        masks_list = list(masks.values())
        masks[UNION_LABELER] = (LINE_COLOR * np.logical_or.reduce(masks_list)).astype(dtype)
        masks[INTERSECTION_LABELER] = (LINE_COLOR * np.logical_and.reduce(masks_list)).astype(dtype)

        for labeler in ACCEPTED_LABELERS:
            if labeler not in masks.keys():
                print_error(xml, "No Regions From Labeler \"{}\"".format(labeler))

    else:
        print_error(xml, "No Masks Found")

    return masks

def process_xml(xml, input_dir, output_dir):
    img_path = get_xml_matching_imgs(xml)
    if len(img_path) > 0:
        if len(img_path) > 1:
            print_error(xml, "Multiple Matching Images Found")
        img_path = img_path[0]
        masks = get_xml_masks(xml, img_path)
        if len(masks) > 0:
            sample_name = get_sample_name(xml)
            sample_class = get_sample_class(xml, input_dir)

            output_img_path = os.path.join(output_dir, sample_class, "imgs", os.path.basename(img_path))
            if os.path.isfile(output_img_path):
                print_error(xml, "File \"{}\" Already Exists. Overwriting".format(output_img_path))
            shutil.copyfile(img_path, output_img_path)

            for labeler in masks.keys():
                output_mask_path = os.path.join(output_dir, sample_class, "masks", labeler, "{}.png".format(sample_name))
                if os.path.isfile(output_mask_path):
                    print_error(xml, "File \"{}\" Already Exists. Overwriting".format(output_mask_path))
                imageio.imwrite(output_mask_path, masks[labeler])

    else:
        print_error(xml, "No Matching Images Found")



if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) >= 2 else os.path.join(os.environ["LOOPS_PATH"], "data", "filtered")
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(os.environ["LOOPS_PATH"], "data", "processed")

    xmls = glob(os.path.join(input_dir, "**", "*.xml"), recursive=True)
    sample_classes = set([get_sample_class(xml, input_dir) for xml in xmls])
    for sample_class in sample_classes:
        os.makedirs(os.path.join(output_dir, sample_class, "imgs"), exist_ok=True)
        for labeler in (ACCEPTED_LABELERS + [UNNAMED_LABELER, UNION_LABELER, INTERSECTION_LABELER]):
            os.makedirs(os.path.join(output_dir, sample_class, "masks", labeler), exist_ok=True)

    shutil.copytree(os.path.join(input_dir, NORMAL_CLASS), os.path.join(output_dir, NORMAL_CLASS))
    with Pool(os.cpu_count()) as pool:
        pool.starmap(process_xml, [(xml, input_dir, output_dir) for xml in xmls])
