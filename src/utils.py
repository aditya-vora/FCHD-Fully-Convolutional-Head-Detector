import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import os
import glob
import json
import pickle
from src.config import opt
import cv2
import matplotlib.pyplot as plt

K = 20 # For NN search


class data:
    def __init__(self, d):
        self.path = d['img_path']
        self.n_boxs = d['number']
        self.bboxs = d['coordinates']

def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def generate_gt_name(id):
    return 'gt_img_'+str(id).zfill(6)+'.jpg'

def generate_img_name(id):
    return 'img_'+str(id).zfill(6)+'.jpg'

def draw_bounding_box_on_image_array(image,ymin,xmin,ymax,xmax,color='red',thickness=4,
                                     use_normalized_coordinates=False):
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    #image_pil = image
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness,use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,ymin,xmin,ymax,xmax,color='red',thickness=4,
                               use_normalized_coordinates=False):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def adaptive_kernal(point_i,locations):
    dist = []
    for j in range(locations.shape[0]):
        point = locations[j,:]
        point = point.reshape(point.shape[0],1)
        diff = point_i - point
        dist.append(np.sqrt(np.dot(diff.T,diff)))
    dist.sort()
    nn_dists = dist[1:K+1]
    # print nn_dists
    dist_i = np.mean(np.asarray(nn_dists))
    fz = np.floor(0.3 * dist_i)
    return fz

def save_json(data,filepath):
    with open(filepath, 'w') as fp:
        json.dump(data, fp)

def save_pickle(data,filepath):
    with open(filepath,'wb') as fp:
        pickle.dump(data,fp)
    fp.close()

def load_annotations(path):
    with open(path,'rb') as fp:
        dict = pickle.load(fp)
    return dict

def get_ground_truth_name(file_path):
    file_id = get_file_id(file_path)
    return 'ground_truth_'+file_id.split('_')[1]


def data_line_parser(line, dataset):
    if ":" not in line:
        img_path, _ = line.split(";")
        if dataset == 'hollywood':            
            src_path = os.path.join(opt.hollywood_dataset_root_path,'Images',img_path.replace('"',''))
        elif dataset == 'brainwash':
            src_path = os.path.join(opt.brainwash_dataset_root_path, img_path.replace('"',''))
        num_coordinates = 0
        coordinates = []
        return {'img_path':src_path, 'number': num_coordinates,'coordinates':coordinates}
    else:
        img_path, bbox_coordinates_raw = line.split(":")
        if dataset == 'hollywood':
            src_path = os.path.join(opt.hollywood_dataset_root_path,'Images',img_path.replace('"',''))
        elif dataset == 'brainwash':
            src_path = os.path.join(opt.brainwash_dataset_root_path, img_path.replace('"',''))             
        bbox_coordinates_raw = bbox_coordinates_raw.replace("(","")
        bbox_coordinates_raw = bbox_coordinates_raw.replace("),",",")
        bbox_coordinates_raw = bbox_coordinates_raw.replace(").","")
        bbox_coordinates_raw = bbox_coordinates_raw.replace(");","")
        bbox_coordinates_str = bbox_coordinates_raw.split(", ")
        coordinates_list = [float(i) for i in bbox_coordinates_str]
        num_coordinates = len(coordinates_list)/4
        coordinates = np.zeros(shape=(int(num_coordinates),4),dtype=np.float)
        entry_idx = 0
        for i in range(0,len(coordinates_list),4):
            coord = coordinates_list[i:i+4]
            coord = [coord[1], coord[0], coord[3], coord[2]]
            coordinates[entry_idx,:] = coord
            entry_idx += 1
    return {'img_path':src_path, 'number': num_coordinates,'coordinates':coordinates}

def get_phase_data_list(data_list_path, dataset):
    """Return a list of data object.
    data object: path, n_boxs, bboxs
    
    Args: data_list_path: list of filenames and groundtruth information available
            in the brainwash dataset. 

    Returns: A list of data objects. Where the length of the list is equal 
            to the number of images contained in the split of the dataset.
    """
    data_list = []
    with open(data_list_path, 'rb') as fp:
        for line in fp.readlines():
            d = data_line_parser(line, dataset)
            if d['number'] != 0:
                d_object = data(d)
                data_list.append(d_object)
    return data_list

# def shanghai_tech_phase_data(data_list_path, phase):
#     data_list = []
#     d = {}
#     for img_path in data_list_path:
#         gt_name = get_ground_truth_name(img_path)
#         gt_path = os.path.join(opt.shanghai_data_root_path, phase ,'annotations', gt_name)
#         gt = load_annotations(gt_path)
#         d['img_path'] = img_path
#         d['number'] = gt['number']
#         bboxs_xyxy = gt['coordinates']
#         bboxs_yxyx = np.zeros(shape=(int(gt['number']),4),dtype=np.float)
#         entry_idx = 0
#         for i in range(gt['coordinates'].shape[0]):
#             xmin, ymin, xmax, ymax = bboxs_xyxy[i,:]
#             coord = [ymin, xmin, ymax, xmax]
#             bboxs_yxyx[entry_idx,:] = coord
#             entry_idx += 1

#         d['coordinates'] = bboxs_yxyx
#         if d['number'] != 0:
#             d_object = data(d)
#             data_list.append(d_object)
    
#     return data_list


def check_loaded_data(d):
    path = d.path
    n_boxs = d.n_boxs
    bboxs = d.bboxs

    # print bboxs.shape
    image = cv2.imread(path)
    image_np = np.copy(np.asarray(image, dtype=np.uint8))
    for i in range(n_boxs):
        ymin,xmin,ymax,xmax = bboxs[i,:]
        #xmin, ymin, xmax, ymax = bboxs[i,:]
        draw_bounding_box_on_image_array(image_np, ymin,xmin,ymax,xmax)
        #plt.imshow(image_np)
    cv2.imshow('image',image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.show()
    
def vis_anchors(img_tensor, anchor):
    img_tensor = img_tensor.cpu()
    img = img_tensor.data.numpy()
    img_cp = img.copy()
    img_cp = np.squeeze(img_cp)
    img_cp = draw_bboxs(img_cp, anchor, anchor.shape[0]) 
    plt.imshow(img_cp)
    plt.show()

def draw_bboxs(image, bboxs, n_boxs, is_transpose=True):
    if is_transpose:
        image = image.transpose((1,2,0))
    image_np = np.copy(np.asarray(image, dtype=np.uint8))
    for i in range(n_boxs):
        ymin,xmin,ymax,xmax = bboxs[i,:]
        draw_bounding_box_on_image_array(image_np, ymin,xmin,ymax,xmax)
    return image_np

# def _test():
#     img_path_list = [filename for filename in glob.glob((os.path.join(opt.shanghai_data_root_path,'images','*.jpg')))]
#     data_list = shanghai_tech_phase_data(img_path_list)    
#     pass

# if __name__ == "__main__":
#     _test()
