import numpy as np
import cv2
import os
from iou import iou
from display_images import get_rectangle
from numpy.random import randn
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt

RANGES = [0, 256, 0, 256, 0, 256]
EPSILON = 0.000001

class ParticleFilterTracker:
    def __init__(self, N, sigma, velocity, T, use_mgwo, mgwo_max_iter):
        self.N = N
        self.dim = 0
        self.sigma = sigma
        self.velocity = velocity
        self.T = T
        self.use_mgwo = use_mgwo
        self.mgwo_max_iter = mgwo_max_iter
        self.particles = None
        self.weights = None
        self.target_hist = None
        self.w_init = 0
        self.h_init = 0
        self.G = None
        self.Q = None

    def initialize(self, initial_state, w_init, h_init, frame_rgb):
        self.w_init = w_init
        self.h_init = h_init
        self.dim = initial_state.shape[0]

        # Create motion model matrices
        self.G = np.array([[1, self.T, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, self.T, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        Qx = np.array([[1/4*self.T**4, 1/2*self.T**3], [1/2*self.T**3, self.T**2]]) * self.sigma[0]**2
        Qy = np.array([[1/4*self.T**4, 1/2*self.T**3], [1/2*self.T**3, self.T**2]]) * self.sigma[1]**2
        self.Q = np.zeros((6, 6))
        self.Q[0:2, 0:2] = Qx
        self.Q[2:4, 2:4] = Qy
        self.Q[4, 4] = self.sigma[2]**2
        self.Q[5, 5] = self.sigma[3]**2

        # Create particles and weights
        self.particles = self.create_clone_particles(initial_state)
        self.weights = np.ones(self.N) / self.N
        
        # Create reference histogram
        self.target_hist = self.get_histogram(frame_rgb, initial_state)

    def predict(self):
        if self.G is not None and self.Q is not None:
            rng = np.random.default_rng()
            for i, particle in enumerate(self.particles):
                mean = self.G.dot(particle)
                self.particles[i] = rng.multivariate_normal(mean, self.Q)
        else:
            rng = np.random.default_rng()
            self.particles = rng.normal(self.particles, self.sigma)
    
    def update(self, frame, normalize = 'True'):
        for i,particle in enumerate(self.particles):
            particle_hist = self.get_histogram(frame,particle, visualize = False)
            self.weights[i] = 1-cv2.compareHist(self.target_hist, particle_hist, cv2.HISTCMP_BHATTACHARYYA) #cv2.HISTCMP_INTERSECT) #cv2.HISTCMP_CHISQR)
            
        if normalize:
            self.weights = self.weights / (np.sum(self.weights)+EPSILON)

    def get_histogram(self, frame, particle, visualize = False):
        if particle is not None:
            # Crop roi from image
            x1,x2,y1,y2 = get_rectangle(particle, self.w_init, self.h_init)
            roi = frame[y1:y2,x1:x2]
        else:
            roi = frame

        # Calculate histogram
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8,8,8], RANGES)
        hist = hist.flatten() / (np.sum(hist.flatten()) + EPSILON)
        if visualize:
            fig, ax = plt.subplots()
            plt.imshow(roi)
            plt.show()
        return hist
    
    def apply_mgwo_optimizer(self, frame):
  
        # In case of losing object
        #TODO: can I just call initializer here?
        if not sum(self.weights)>0:
            height,width = frame.shape
            particles = self.create_gaussian_particles([int(width/2), 1, int(height/2), 1, 1, 1], [100,5,100,5,5,1], self.N, self.dim)
            self.weights = self.update(frame, particles, self.weights, self.target_hist, self.w_init, self.h_init)
            print('Object lost')

            a = 2
            particles_new = np.empty((self.N, self.dim))
            weights_new = np.empty((self.N, 1))

            for t in range (self.mgwo_max_iter):
                #TODO: create enum for sigma
                r1 = 0.5 + randn(self.N, 3, self.dim)*self.sigma[4]
                r2 = 0.5 + randn(self.N, 3, self.dim)*self.sigma[4]
                A = 2*a*r1-a
                C = 2*r2
                
                ind = np.argpartition(self.weights, -3)[-3:]
                ind = ind[np.argsort(self.weights[ind])]

                alpha = ind[2]
                beta = ind[1]
                delta = ind[0]

                X_alpha = particles[alpha]
                X_beta = particles[beta]
                X_delta = particles[delta]

                for i, particle in enumerate(particles):
                    D_alpha = np.absolute(C[i,0,:]*X_alpha - particle)
                    D_beta = np.absolute(C[i,1,:]*X_beta - particle)
                    D_delta = np.absolute(C[i,2,:]*X_delta - particle)

                    X_1 = X_alpha - A[i,0,:]*D_alpha
                    X_2 = X_beta - A[i,1,:]*D_beta
                    X_3 = X_delta - A[i,2,:]*D_delta

                    particles_new[i] = (X_1 + X_2 + X_3)/3
                
                # Update particle if new solution is better
                #TODO: make this more efficient
                self.weights     = self.update(frame, particles, self.weights, self.target_hist, self.w_init, self.h_init, normalize = False)
                weights_new = self.update(frame, particles_new, weights_new, self.target_hist, self.w_init, self.h_init, normalize = False)
                for i,(weight, weight_new) in enumerate(zip(self.weights, weights_new)):
                    if weight_new > weight:
                        self.particles[i] = particles_new[i]
                        self.weights[i] = weights_new[i]
                # Update 'a' parameter
                a = 2 - 2*(np.sin(np.pi*t/self.mgwo_max_iter/2))**2

                # Display particles
                # display_image(frame, w_init, h_init, 'MGWO_'+str(t+1), size=1.0, particles = particles, weights = weights)

        self.weights = self.weights / (np.sum(self.weights)+EPSILON)

    def create_clone_particles(self, initial):
        particles = np.tile(initial,(self.N,1))
        return particles
    
    def create_gaussian_particles(self,mean, std, N, dim):
        particles = np.empty((N, dim))
        for i in range(dim):
            particles[:, i] = mean[i] + (randn(N) * std[i])
        return particles
    
    def estimate(self, groundTruth):
        # State estimation by average of particles
        state_avg = np.sum(self.particles * self.weights[:,None], axis=0)

        # State estimation by largest weight
        state_lw = self.particles[np.argmax(self.weights)]

        # print ('Average:        ', state_avg)
        # print ('Highest weight: ', state_lw)
        # print ('True state:     ', tst[1,:])

        # Bounding box of the ground truth
        bb_gt = {'x1':groundTruth[0], 'y1':groundTruth[1],'x2':groundTruth[0]+groundTruth[2], 'y2':groundTruth[1]+groundTruth[3]}
        # Bounding box of the solution using the average
        # print('state_avg:' + str(state_avg))
        x1,x2,y1,y2 = get_rectangle(state_avg, self.w_init, self.h_init)

        if x1 < x2:
            bb_avg = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
        elif x1 > x2:
            bb_avg = {'x1':x2, 'y1':y1, 'x2':x1, 'y2':y2}
        else:
            bb_avg = {'x1':x1, 'y1':y1, 'x2':x2+1, 'y2':y2}

        if not y1 < y2:
            print(state_avg)
            print(x1, y1, x2-x1, y2-y1)
            print(groundTruth)
        # Bounding box of the solution using the largest weight
        # bb_lw  = {'x1':state_lw[0]-state_lw[2], 'y1':state_lw[1]-state_lw[3], 'x2':state_lw[0]+state_lw[2], 'y2':state_lw[1]+state_lw[3]}

        # Compute Intersection over Union to evaluate the solution. Higher values are 
        # better. IoU is bounded in [0,1]. In object detection an IoU >= 0.5 is usually 
        # considered a correct detection.
        IOU_avg = iou(bb_gt, bb_avg)
        # print ('IOU avg  = {:.3f}'.format(IOU_avg))
        # print ('IOU best = {:.3f}'.format(iou(bb_tst, bb_lw)))
        return state_avg, IOU_avg

    def get_histogram(self, frame, particle, visualize = False):
        if particle is not None:
            # Crop roi from image
            x1,x2,y1,y2 = get_rectangle(particle, self.w_init, self.h_init)
            roi = frame[y1:y2,x1:x2]
        else:
            roi = frame

        # Calculate histogram
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8,8,8], RANGES)
        hist = hist.flatten() / (np.sum(hist.flatten()) + EPSILON)
        if visualize:
            fig, ax = plt.subplots()
            plt.imshow(roi)
            plt.show()
        return hist

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill (1.0 / len(self.weights))

    def resample(self):
        indexes = systematic_resample(self.weights)
        self.resample_from_index(indexes)
        assert np.allclose(self.weights, 1/self.N)

    def import_data(self, dataset):
        folder = 'Datasets/' + dataset + '/' + dataset

        # Read images  
        images,filenames = self.load_images_from_folder(folder + '/img')

        # Read ground truth
        with open(folder + '/groundtruth_rect.txt', 'r') as f:
            gt = [[int(x) for x in line.split()] for line in f]
        gt = np.array(gt)
        return images, filenames, gt

    def load_images_from_folder(self, folder):
        images = []
        filenames = []
        for filename in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
        return images, filename

    def run_pf(self, dataset):
        # Import images, ground truth
        images, filenames, gt = self.import_data(dataset)
        initial_state = np.array([gt[0,0]+gt[0,2]/2,self.velocity[0],gt[0,1]+gt[0,3]/2,self.velocity[1], 0, 1])
        w_init = gt[0,2]
        h_init = gt[0,3]

        self.initialize(initial_state=initial_state, w_init=w_init, h_init=h_init, frame_rgb=images[0])

        estimation = np.empty((len(images),self.dim))
        frames_rgb = []
        IOU_avg = []

        for index,frame_bgr in enumerate(images):
            if frame_bgr is not None:
                # Convert to rgb for display
                frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
                # Convert to lab for histogram
                frame_lab = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2Lab)

                # Create reference
                if index == 0:
                    target_hist = self.get_histogram(frame_rgb, initial_state)
                    # display_image(frame_rgb, w_init, h_init, '', size=1.0, particles = particles, weights = weights)

                # Move particles
                particles = self.predict()
                # display_image(frame_rgb, w_init, h_init, 'predict', size=1.0, particles = particles, weights = weights)

                # Evaluate particles and calculate weights
                frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                weights = self.update(frame_rgb)

                # Apply Modified Gray Wolf Optimizer
                if self.use_mgwo:
                    frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                    self.apply_mgwo_optimizer(frame = frame_rgb)
                    # display_image(frame_rgb, w_init, h_init, 'MGWO', size=1.0, particles = particles, weights = weights)

                # Estimate current state
                state_estimate, IOU_avg_act = self.estimate(gt[index,:])
                IOU_avg.append(IOU_avg_act)
                estimation[index,:] = state_estimate      

                self.resample()

                if index % 1 == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # display_image(frame_rgb, w_init, h_init, 'resample', size=1.0, particles = particles, weights = weights)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # display_image(frame_rgb, w_init, h_init, 'estimate', size=1.0, particles = state_estimate, weights = weights)
                index += 1
    
