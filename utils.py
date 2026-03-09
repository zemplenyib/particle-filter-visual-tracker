import os
import cv2
import numpy as np
from display_images import get_rectangle

RANGES = [0, 256, 0, 256, 0, 256]
EPSILON = 0.000001

def load_data(path):
        # Read images
        images = []
        filenames = []
        
        img_path = os.path.join(path, 'img')
        for filename in sorted(os.listdir(img_path)):
            img = cv2.imread(os.path.join(img_path, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)

        # Read ground truth
        gt_path = os.path.join(path, 'groundtruth_rect.txt')
        with open(gt_path, 'r') as f:
            # Handles both comma and space separated files
            gt = [[int(float(x)) for x in line.replace(',', ' ').split()] for line in f]
        gt = np.array(gt)
        
        return images, filenames, gt

def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # print('groundTruth' + '\n' + 'x1:' + str(bb1['x1']) + '\n' + 'x2:' + str(bb1['x2']) + '\n' + 'y1:' + str(bb1['y1']) + '\n' + 'y2:' + str(bb1['y2']))
    # print('Particle' + '\n' + 'x1:' + str(bb2['x1']) + '\n' + 'x2:' + str(bb2['x2']) + '\n' + 'y1:' + str(bb2['y1']) + '\n' + 'y2:' + str(bb2['y2']))
    # determine the coordinates of the intersection rectangle
    x_left   = max(bb1['x1'], bb2['x1'])
    y_top    = max(bb1['y1'], bb2['y1'])
    x_right  = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    # print('x_right:' + str(x_right) + '\n' + 'x_left:' + str(x_left) + '\n' + 'y_bottom:' + str(y_bottom) + '\n' + 'y_top:' + str(y_top))
    # print('intersection_area:' + str(intersection_area))

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']+1) * (bb1['y2'] - bb1['y1']+1)
    bb2_area = (bb2['x2'] - bb2['x1']+1) * (bb2['y2'] - bb2['y1']+1)
    # print('bb1_area:' + str(bb1_area) + '\n' + 'bb2_area:' + str(bb2_area))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class LikelihoodEvaluator:
    def __init__(self, w_init, h_init, initial_frame, state):
        self.w_init = w_init
        self.h_init = h_init
        self.target_hist = self.get_histogram(initial_frame, state)


    def get_histogram(self, frame, particle):
        if particle is not None:
            # Crop roi from image
            x1,x2,y1,y2 = get_rectangle(particle, self.w_init, self.h_init)
            roi = frame[y1:y2,x1:x2]
        else:
            roi = frame

        # Calculate histogram
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8,8,8], RANGES)
        hist = hist.flatten() / (np.sum(hist.flatten()) + EPSILON)
        return hist


    def update(self, frame, particles, weights, normalize = 'True'):
        for i,particle in enumerate(particles):
            particle_hist = self.get_histogram(frame,particle)
            weights[i] = 1-cv2.compareHist(self.target_hist, particle_hist, cv2.HISTCMP_BHATTACHARYYA) #cv2.HISTCMP_INTERSECT) #cv2.HISTCMP_CHISQR)
            
        if normalize:
            weights /= (np.sum(weights)+EPSILON)