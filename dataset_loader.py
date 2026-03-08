import os
import cv2
import numpy as np

class DatasetLoader:

    def load_data(self, path):
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
