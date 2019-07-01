import os
import cv2
import numpy as np
import xml.etree.cElementTree as ET
from lxml import etree
import re
import shutil


### FUNCTIONS
def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir, 'Annotations')

        new_path = img_path.replace(INPUT_DIR, new_path, 1)
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths

def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = DATASET_NAME
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)
    return xml_path



def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def append_bb(ann_path, line):
    class_name, xmin, ymin, xmax, ymax, truncated = line

    tree = ET.parse(ann_path)
    annotation = tree.getroot()

    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = class_name
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = truncated
    ET.SubElement(obj, 'difficult').text = '0'

    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = xmin
    ET.SubElement(bbox, 'ymin').text = ymin
    ET.SubElement(bbox, 'xmax').text = xmax
    ET.SubElement(bbox, 'ymax').text = ymax

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, ann_path)

def makebboxes(xcenter,ycenter,radius):
    xmin = xcenter - radius
    xmax = xcenter + radius
    ymin = ycenter - radius
    ymax = ycenter + radius
    return xmin, ymin, xmax, ymax



INPUT_DIR = '../Data/Mixin/Orig'
OUTPUT_DIR = '../Data/Mixin/Video_1299'
DATASET_NAME = 'video_1299'
IMAGE_PATH_LIST = []
RADIUS = 10

annotation_formats = {'PASCAL_VOC' : '.xml'}
jpg_list = [f for f in os.listdir(INPUT_DIR) if 'jpg' in f]

ANN = os.path.join(OUTPUT_DIR,'PASCAL_VOC','Annotations') 
JPEG_PATH = os.path.join(OUTPUT_DIR,'PASCAL_VOC','JPEGImages')
IMAGESETS_PATH = os.path.join(OUTPUT_DIR,'PASCAL_VOC','ImageSets','Main')
IMAGESETS_FILE = os.path.join(IMAGESETS_PATH,'trainval.txt')

if not os.path.exists(ANN): 
    os.makedirs(ANN) 
if not os.path.exists(JPEG_PATH): 
    os.makedirs(JPEG_PATH)
if not os.path.exists(IMAGESETS_FILE):
    os.makedirs(IMAGESETS_PATH)     

IMAGE_PATH_LIST = []
for f in sorted(jpg_list):
    f_path = os.path.join(INPUT_DIR, f)
    if os.path.isdir(f_path):
        # skip directories
        continue
    # check if it is an image
    test_img = cv2.imread(f_path)
    if test_img is not None:
        IMAGE_PATH_LIST.append(f_path)

# create empty annotation files for each image, if it doesn't exist already
for idx,img_path in enumerate(IMAGE_PATH_LIST):
    idx += 1000
    if(not os.path.exists(img_path)):
        continue
    np_annotation_file = os.path.basename(img_path).replace('jpg', 'npy', 1).replace('gen', 'gen_labels', 1)

    dest_path = os.path.join(JPEG_PATH,'{:07d}.jpg'.format(idx))
    shutil.move(img_path,dest_path)
    # image info for the .xml file
    test_img = cv2.imread(dest_path)
    abs_path = os.path.abspath(dest_path)
    folder_name = os.path.dirname(dest_path)
    image_name = os.path.basename(dest_path)
    img_height, img_width, depth = (str(number) for number in test_img.shape)

    for ann_path in get_annotation_paths(dest_path, annotation_formats):
        ann_path = ann_path.replace('JPEGImages','Annotations',1)
        xmlpath = create_PASCAL_VOC_xml(ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)
        for xmin, ymin, xmax, ymax in np.load(os.path.join(INPUT_DIR,np_annotation_file)):
            truncated = 0

            if xmin < 0:
                xmin = 0
                truncated = 1
            if xmax > int(img_width):
                xmax = int(img_width) - 1
                truncated = 1
            if ymin < 0:
                ymin = 0
                truncated = 1
            if ymax > int(img_height):
                ymax = int(img_height) - 1   
                truncated = 1
 
            append_bb(xmlpath,('particle', str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax)), str(truncated)))
    
    os.remove(os.path.join(INPUT_DIR,np_annotation_file))
    with open(IMAGESETS_FILE, 'a') as myfile:
        myfile.write('{:07d}'.format(idx) + '\n') # append line