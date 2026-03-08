from display_images import get_rectangle
import numpy as np
import cv2

RANGES = [0, 256, 0, 256, 0, 256]
EPSILON = 0.000001

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
