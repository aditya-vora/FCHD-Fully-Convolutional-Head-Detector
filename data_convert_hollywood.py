"""
This script converts the data from the Hollywoods dataset in the format that is expected for the training by the train.py script.
"""
from __future__ import division

import xml.etree.ElementTree as ET
import os
import math
from PIL import Image
import numpy as np
import cv2
import src.utils as utils

PHASES = ['train', 'val', 'test']
IMGS_PTH = './data/HollywoodHeads/JPEGImages'


def convert_txt(src_path, dst_path, ann_path):
    wfp = open(dst_path, "wb")

    with open(src_path, 'rb') as rfp:
        for line in rfp.readlines():
            line, _ = line.split("\n")
            ann_file_name = line + ".xml"
            insert_line = ""
            insert_line = insert_line + line + ".jpeg"
            img_name = line + ".jpeg"
            img_path = os.path.join(IMGS_PTH, img_name)
            f = Image.open(img_path)

            W_o, H_o = f.size
            f = f.resize((640,480), Image.ANTIALIAS)
            
            W_n = 640
            H_n = 480
            # Convert to RGB
            f.convert('RGB')
            img = np.asarray(f, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #image_np = np.copy(np.asarray(img, dtype=np.uint8))
            # cv2.imshow('image',img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(os.path.join('./data/HollywoodHeads/Images',line+".jpeg"),img)

            insert_line = "\"" + insert_line + "\""
            ann = ET.parse(os.path.join(ann_path,ann_file_name))
            bboxs = []
            for obj in ann.findall('object'):
                bbox_ann = obj.find('bndbox')
                if bbox_ann is None:
                    continue
                bboxs.append([float(bbox_ann.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            for i in range(len(bboxs)): 
                bbox = bboxs[i]
                bbox[0] = int(bbox[0]*(W_n/W_o))
                bbox[1] = int(bbox[1]*(H_n/H_o))
                bbox[2] = int(bbox[2]*(W_n/W_o))
                bbox[3] = int(bbox[3]*(H_n/H_o))


            bboxs_string = ""
            nbboxs = len(bboxs)
            index = 1
            if nbboxs == 0:
                insert_line = insert_line + ';' + '\n'
                wfp.write(insert_line)
            else:
                for bbox in bboxs:
                    bbox_string = ""
                    if index <= nbboxs - 1:
                        bbox_string = bbox_string + ', '.join(str(math.floor(e)) for e in bbox) 
                        bbox_string = '(' + bbox_string + '), '
                        bboxs_string = bboxs_string + bbox_string
                        index += 1
                    else:
                        bbox_string = bbox_string + ', '.join(str(math.floor(e)) for e in bbox) 
                        bbox_string = '(' + bbox_string + '); '
                        bboxs_string = bboxs_string + bbox_string
                        index += 1
                insert_line = insert_line + ': '
                insert_line = insert_line + bboxs_string + '\n'
                wfp.write(insert_line)

def convert_hollywood(path):
    splits_folder = os.path.join(path, 'Splits')
    ann_folder = os.path.join(path, 'Annotations')
    for phase in PHASES:
        if phase == 'train':
            print "Phase train ongoing..."
            train_data_list_path = os.path.join(splits_folder, 'train.txt')
            train_data_path = os.path.join(path, 'hollywood_train.idl')
            convert_txt(train_data_list_path, train_data_path, ann_folder)
        elif phase == 'val':
            print "Phase val ongoing..."

            val_data_list_path = os.path.join(splits_folder, 'val.txt')
            val_data_path = os.path.join(path, 'hollywood_val.idl')
            convert_txt(val_data_list_path, val_data_path, ann_folder)
        else: 
            print "Phase test ongoing..."

            test_data_list_path = os.path.join(splits_folder, 'test.txt')
            test_data_path = os.path.join(path, 'hollywood_test.idl')
            convert_txt(test_data_list_path, test_data_path, ann_folder)


if __name__ == "__main__":
    data_path = './data/HollywoodHeads'
    convert_hollywood(data_path)