"""
This script converts the data from the Hollywoods dataset in the format that is expected for the training by the train.py script.
"""

import xml.etree.ElementTree as ET
import os
import math

PHASES = ['train', 'val', 'test']

def convert_txt(src_path, dst_path, ann_path):
    wfp = open(dst_path, "wb")

    with open(src_path, 'rb') as rfp:
        for line in rfp.readlines():
            line, _ = line.split("\n")
            ann_file_name = line + ".xml"
            insert_line = ""
            insert_line = insert_line + line + ".jpg"
            insert_line = "\"" + insert_line + "\""
            ann = ET.parse(os.path.join(ann_path,ann_file_name))
            bboxs = []
            for obj in ann.findall('object'):
                bbox_ann = obj.find('bndbox')
                if bbox_ann is None:
                    continue
                bboxs.append([float(bbox_ann.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
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
            train_data_list_path = os.path.join(splits_folder, 'train.txt')
            train_data_path = os.path.join(path, 'hollywood_train.idl')
            convert_txt(train_data_list_path, train_data_path, ann_folder)
        elif phase == 'val':
            val_data_list_path = os.path.join(splits_folder, 'val.txt')
            val_data_path = os.path.join(path, 'hollywood_val.idl')
            convert_txt(val_data_list_path, val_data_path, ann_folder)
        else: 
            test_data_list_path = os.path.join(splits_folder, 'test.txt')
            test_data_path = os.path.join(path, 'hollywood_test.idl')
            convert_txt(test_data_list_path, test_data_path, ann_folder)


if __name__ == "__main__":
    data_path = './data/HollywoodHeads'
    convert_hollywood(data_path)