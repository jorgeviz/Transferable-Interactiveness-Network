from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import ipdb
import pickle

import _init_paths
from ult.visualization import draw_bounding_boxes
from ult.config import cfg
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ipdb

IMAGES_PATH = "Data/hico_20160224_det/images/"

def parse_args():
    """ Arguments Parser
    """
    parser = argparse.ArgumentParser(description='BBoxes and Image visualizer')
    parser.add_argument('--bboxes_pkl', dest='bboxes',
            help='Specify Pickle file with BBoxes file',
            default="-Results/1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2.pkl", type=str)
    parser.add_argument('--img_id', dest='img_id',
            help='Specify Image ID from HICO-DET Test Dataset',
            default=1830, type=int)
    parser.add_argument('--destination', dest='destination',
            help='Output JPG destination',
            default="-Results/bboxes_render.jpg", type=str) 
    parser.add_argument('--anno_bbox_mat', dest='anno_bbox_mat',
            help='Annotation Matrix Path',
            default="Data/hico_20160224_det/anno_bbox.mat", type=str) 
    parser.add_argument('--hico_obj_names', dest='hico_obj_names',
            help='HICO Objects name',
            default="Data/hico_20160224_det/hico_obj_names.npy", type=str) 
    args = parser.parse_args()
    return args


def display_img(img, sentence=""):
    """ Display image in screen

        Args
        -----
        img: np.ndarray
            (M,N,3) Array of floats between 0 and 1
        sentence: str
            Predicted sentence
    """
    implt =  plt.imshow(
        img
    )
    plt.show()
    print("Prediction sentence: " + sentence)
    input("[ENTER]")


def draw_bboxes(img, bboxes, color=(0, 0, 1.0), thickness=1):
    """ Draw Bounding boxes in image

        Args
        -----
        img: np.ndarray
            Image array
        bboxes: list
            List of B-boxes arrays
        color: tuple
            RGB tuple
        thickness: int
            Border Thickness
    """
    for bbox in bboxes:
        # if [x1, y1, x2, y2]
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[-2:]), color, thickness)


def decode_bboxes(_bboxes):
    """ Decode detected boxes
    """
    decd = []
    for _hum in _bboxes:
        decd.append(
            {
                "human_box": _hum[0],
                "obj_box": _hum[1],
                "obj_class": int(_hum[2]),
                "pred_score_ho": _hum[3],
                "pred_score_bin": _hum[6],
            }
        )
    return decd
    

def predict_sent(dbxs, anno_bbox, hico):
    """ Decode the Predicted sentence  of the HOI

        Args
        -----
        dbxs: list
            List of Dicts with evaluation data
        anno_bbox: str
            Annotation Matrix path
        hico: str
            HICO Object names path
        
        Return
        -----
        str
            Predicted sentences
    """
    _snt = []
    # Load data
    anno_list = loadmat(anno_bbox)["list_action"]
    hico = {int(_[0]): _[1] for _ in np.load(hico)} 
    for _h in dbxs:
        # Find verb
        try:
            _verb = anno_list\
                [_h['pred_score_ho'].argmax()]\
                ['vname'][0][0]
        except Exception as e:
            print("Verb Issue", e)
            _verb = ""
        # Find Object
        try:
            # hico[_h['obj_class']]
            # _objn = anno_list\
            #     [_h['obj_class']]\
            #     ['vname'][0][0]
            _objn = anno_list\
                [_h['pred_score_ho'].argmax()]\
                ['nname'][0][0]
        except Exception as e:
            print("Object Issue", e)
            _objn = ""
        _snt.append(
            "Human " + _verb + " " + _objn
        )
    return "\n".join(_snt)


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Generate image file
    _img = IMAGES_PATH + "HICO_test2015_{}.jpg".format(
        str(args.img_id).zfill(8)
    )
    print("Reading image file at: {}".format(_img))
    # Read image
    im = cv2.imread(_img)
    im_orig = im.astype(np.float32, copy=True)
    im_orig /= 255.0
    # Read BBoxes
    _bboxs = pickle.load(
        open(args.bboxes, 'rb')
        )[args.img_id]
    # Decode bboxes 
    d_bboxes = decode_bboxes(_bboxs)
    # Draw humans
    draw_bboxes(
        im_orig,
        [_['human_box'] for _ in d_bboxes]
    )
    # Draw Objects
    draw_bboxes(
        im_orig,
        [_['obj_box'] for _ in d_bboxes],
        (1.0,0,0)
    )
    # Predict Sentences
    _sents = predict_sent(d_bboxes, args.anno_bbox_mat, args.hico_obj_names)
    cv2.putText(im_orig, _sents, (20,20) , cv2.FONT_HERSHEY_PLAIN, 
        1.0, (1.0,1.0,1.0), 1)
    # Display Boxed Image
    display_img(im_orig, _sents)