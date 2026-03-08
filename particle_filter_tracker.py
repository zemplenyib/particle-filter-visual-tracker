import numpy as np
import cv2
import os
from iou import iou
from display_images import get_rectangle
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
from display_images import draw_particles_cv2, draw_box_cv2
from dataset_loader import DatasetLoader
from likelihood_evaluator import LikelihoodEvaluator
from mgwo import MGWOptimizer

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
        self.w_init = 0
        self.h_init = 0
        self.G = None
        self.Q = None
        self.rng = np.random.default_rng(seed=42)
        self.lh_eval = None
        self.mgwo = None

    def initialize(self, initial_state, w_init, h_init, frame_rgb):
        np.random.seed(42)
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
        #self.target_hist = self.get_histogram(frame_rgb, initial_state)
        self.lh_eval = LikelihoodEvaluator(self.w_init, self.h_init, frame_rgb, initial_state)
        self.mgwo = MGWOptimizer(self.mgwo_max_iter, 2, self.rng, self.lh_eval)
        

    def predict(self):
        if self.G is not None and self.Q is not None:
            for i, particle in enumerate(self.particles):
                mean = self.G.dot(particle)
                self.particles[i] = self.rng.multivariate_normal(mean, self.Q)
        else:
            self.particles = self.rng.normal(self.particles, self.sigma)

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

    def create_clone_particles(self, initial):
        particles = np.tile(initial,(self.N,1))
        return particles
    
    def create_gaussian_particles(self,mean, std, N, dim):
        particles = np.empty((N, dim))
        for i in range(dim):
            particles[:, i] = mean[i] + (self.rng.standard_normal(N) * std[i])
        return particles

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill (1.0 / len(self.weights))

    def resample(self):
        indexes = systematic_resample(self.weights)
        self.resample_from_index(indexes)
        assert np.allclose(self.weights, 1/self.N)
    
    def apply_mgwo_optimizer(self, frame):
  
        # In case of losing object
        #TODO: can I just call initializer here?
        if not sum(self.weights)>0:
            height,width,_ = frame.shape
            self.particles = self.create_gaussian_particles([int(width/2), 1, int(height/2), 1, 1, 1], [100,5,100,5,5,1], self.N, self.dim)
            self.lh_eval.update(frame, self.particles, self.weights)
            print('Object lost')

        self.mgwo.optimize(frame, self.particles, self.weights, self.sigma[4])

    def run_pf(self, dataset):
        # Import images, ground truth

        ds_loader = DatasetLoader()
        images, filenames, gt = ds_loader.load_data(dataset)
        initial_state = np.array([gt[0,0]+gt[0,2]/2,self.velocity[0],gt[0,1]+gt[0,3]/2,self.velocity[1], 0, 1])
        w_init = gt[0,2]
        h_init = gt[0,3]

        self.initialize(initial_state=initial_state, w_init=w_init, h_init=h_init, frame_rgb=cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB))

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
                # if index == 0:
                #     self.target_hist = self.get_histogram(frame_rgb, initial_state)
                    # display_image(frame_rgb, w_init, h_init, '', size=1.0, particles = particles, weights = weights)

                # Move particles
                self.predict()
                # display_image(frame_rgb, w_init, h_init, 'predict', size=1.0, particles = particles, weights = weights)

                # Evaluate particles and calculate weights
                frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                self.lh_eval.update(frame_rgb, self.particles, self.weights)

                # Apply Modified Gray Wolf Optimizer
                if self.use_mgwo:
                    frame_rgb = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                    self.apply_mgwo_optimizer(frame = frame_rgb)
                    # display_image(frame_rgb, w_init, h_init, 'MGWO', size=1.0, particles = particles, weights = weights)

                # Estimate current state
                state_estimate, IOU_avg_act = self.estimate(gt[index,:])
                IOU_avg.append(IOU_avg_act)
                estimation[index,:] = state_estimate      

                # 5. Resample
                self.resample()

                # 6. Visualize
                print(f"Processing frame: {index} / {len(images)}", end=" ")
                print('IOU avg  = {:.3f}'.format(IOU_avg_act))
                #vis_frame = frame_bgr.copy()
                #vis_frame = draw_particles_cv2(vis_frame, self.particles, self.w_init, self.h_init)
                #vis_frame = draw_box_cv2(vis_frame, state_estimate, self.w_init, self.h_init)
                #
                ## Add text info
                #cv2.putText(vis_frame, f"Frame: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(vis_frame, f"IoU: {IOU_avg_act:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #
                #cv2.imshow('Particle Filter MGWO Tracker', vis_frame)
                #
                ## Handle user input
                #key = cv2.waitKey(30) & 0xFF
                #if key == ord('q'):
                #    break
        print(sum(IOU_avg)/len(IOU_avg))
