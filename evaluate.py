import os
import sys
import time
from xml.dom import minidom
import argparse
import numpy as np
import skimage.io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import tensorflow as tf

import keras.backend
config = tf.ConfigProto()
config.inter_op_parallelism_threads = 1
keras.backend.set_session(tf.Session(config=config))
config = None
mask_rcnn_model = None

def create_config():
    import coco

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BATCH_SIZE = 1

    print("Start creating config")
    c = InferenceConfig()
    print("Created config")
    c.display()
    return c


def load_mrcnn_model(root_dir, gpuId):
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(root_dir, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(root_dir, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    DEVICE = "/gpu:"+gpuId  # /cpu:0 or /gpu:0
    # Create model object in inference mode.
    # with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


def parse_annotation(image_name, file_dir):
    '''
    return lists of [name, y1, x1, y2, x2]
    file_dir is the annotation dir
    '''
    name = image_name.split('.')[0]
    file_name = name + '.xml'
    xml_file = os.path.join(file_dir, file_name)
    parse_doc = minidom.parse(xml_file)
    img_size = parse_doc.getElementsByTagName("size")[0]
    width = int(img_size.getElementsByTagName("width")[0].firstChild.data)
    height = int(img_size.getElementsByTagName("height")[0].firstChild.data)
    objects = parse_doc.getElementsByTagName("object")
    parse_metas = []
    for file_object in objects:
        parse_meta = []
        name = file_object.getElementsByTagName("name")[0]
        name_id = get_name_id(str(name.firstChild.data))
        parse_meta.append(name_id)
        xmax = file_object.getElementsByTagName("xmax")[0]
        ymax = file_object.getElementsByTagName("ymax")[0]
        xmin = file_object.getElementsByTagName("xmin")[0]
        ymin = file_object.getElementsByTagName("ymin")[0]
        parse_meta.append(int(ymin.firstChild.data))
        parse_meta.append(int(xmin.firstChild.data))
        parse_meta.append(int(ymax.firstChild.data))
        parse_meta.append(int(xmax.firstChild.data))
        parse_metas.append(parse_meta)
    gt_class_id = []
    gt_bbox = []
    for i in parse_metas:
        gt_class_id.append(i[0])
        gt_bbox.append(i[1:])
    gt_class_id = np.array(gt_class_id)
    gt_bbox = np.array(gt_bbox)
    return width, height, gt_class_id, gt_bbox


def resize_bbox(bbox, scale, padding):
    new_bbox = []
    for box in bbox:
        box = scale * box
        # y1
        box[0] = box[0] + padding[0][1]
        # x1
        box[1] = box[1] + padding[1][1]
        # y2
        box[2] = box[2] + padding[0][1]
        # y2
        box[3] = box[3] + padding[1][1]
        new_bbox.append(box)
    new_bbox = np.array(new_bbox)
    return new_bbox


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

# class_names = ['person', 'bicycle', 'car','motorcycle', 'bus']
class_names = ['BG', 'person', 'bicycle', 'car', 'motorbike', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def get_name_id(name):
    return class_names.index(name)


def mAP_test(image_dir, file_dir):

    APs = []
    t1 = time.time()
    image_names = next(os.walk(image_dir))[2]
    i = 1
    for image_name in image_names:
        image = skimage.io.imread(os.path.join(image_dir, image_name))
        image_dtype = image.dtype
        orig_width, orig_height, gt_class_id, ori_gt_bbox = parse_annotation(image_name, file_dir)
        image = (utils.resize(image, (orig_height, orig_width), preserve_range=True)).astype(image_dtype)
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)

        # If image name has prefix real_A, skip it, it's an original image
        if image_name.find('_real_A') != -1:
            continue
        if image_name.find('_fake_B') != -1:
            image_name = image_name.replace('_fake_B', '')
        if image_name.find('AOD') != -1:
            image_name = image_name.replace('_AOD-Net', '')
        if image_name.find('dehaze') != -1:
            image_name = image_name.replace('_dehazed', '')

        gt_bbox = resize_bbox(ori_gt_bbox, scale, padding)

        results = mask_rcnn_model.detect([image], verbose=0)
        r = results[0]
        save_path = os.path.join("Mask_RCNN/", image_name)

        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'], figsize=(8, 8), show_mask=False, save_path=None)
        AP, precisions, recalls, overlap = \
            utils.compute_box_ap(gt_bbox, gt_class_id,
                                 r["rois"], r["class_ids"], r["scores"])
        APs.append(AP)
        print('Image Processed #{0}, name: {1}, mAP: {2}: '.format(i, image_name, AP))
        i += 1

    t2 = time.time() - t1
    print("using time: ", t2)
    print("MaskRCNN mAP: ", np.mean(APs))
    return np.mean(APs)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, default="RTTS/Annotations",
                        help='Directory where RTTS annotations are present')
    parser.add_argument('--target', type=str, default="RTTS/DemoImages",
                        help='Image directory for which to compute mAP')
    parser.add_argument('--gpuId', type=str, default='0', help="Specify the GPU Id")
    opt = parser.parse_args()
    # Root directory of the project
    ROOT_DIR = os.path.abspath("")
    sys.path.append(ROOT_DIR)  # To find local version of the library

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

    dehazed_dir = os.path.join(ROOT_DIR,opt.target)
    annotations_dir = os.path.join(ROOT_DIR, opt.annotations)
    config = create_config()
    mask_rcnn_model = load_mrcnn_model(ROOT_DIR, opt.gpuId)
    mAP_test(dehazed_dir, annotations_dir)
