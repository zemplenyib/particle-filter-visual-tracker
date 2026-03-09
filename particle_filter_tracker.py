import numpy as np
import utils
from display_images import Visu
from display_images import get_rectangle
from display_images import extract_rotated_roi
from filterpy.monte_carlo import systematic_resample
from mgwo import MGWOptimizer
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d] [%(levelname).3s] %(message)s")
logger = logging.getLogger(__name__)


RANGES = [0, 256, 0, 256, 0, 256]
EPSILON = 0.000001
SEED = 42

class ParticleFilterTracker:
    def __init__(self, N, sigma, velocity, T, use_mgwo, mgwo_max_iter):
        self._N = N
        self._dim = 0
        self._sigma = sigma
        self._velocity = velocity
        self._T = T
        self._use_mgwo = use_mgwo
        self._mgwo_max_iter = mgwo_max_iter
        self._particles = None
        self._weights = None
        self._w_init = 0
        self._h_init = 0
        self._G = None
        self._Q = None
        self._rng = np.random.default_rng(seed=SEED)
        self._lh_eval = None
        self._mgwo = None
        self._visu = Visu()

    def initialize(self, initial_state, w_init, h_init, frame):
        np.random.seed(SEED)
        self._w_init = w_init
        self._h_init = h_init
        self._dim = initial_state.shape[0]

        # Create motion model matrices
        self._G = np.array([[1, self._T, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, self._T, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        Qx = np.array([[1/4*self._T**4, 1/2*self._T**3], [1/2*self._T**3, self._T**2]]) * self._sigma[0]**2
        Qy = np.array([[1/4*self._T**4, 1/2*self._T**3], [1/2*self._T**3, self._T**2]]) * self._sigma[1]**2
        self._Q = np.zeros((6, 6))
        self._Q[0:2, 0:2] = Qx
        self._Q[2:4, 2:4] = Qy
        self._Q[4, 4] = self._sigma[2]**2
        self._Q[5, 5] = self._sigma[3]**2

        # Create particles and weights
        self._particles = self.create_clone_particles(initial_state)
        self._weights = np.ones(self._N) / self._N
        
        # Create reference histogram
        self._lh_eval = utils.LikelihoodEvaluator(self._w_init, self._h_init, frame, initial_state)
        self._mgwo = MGWOptimizer(self._mgwo_max_iter, 2, self._rng, self._lh_eval)

    def predict(self):
        if self._G is not None and self._Q is not None:
            for i, particle in enumerate(self._particles):
                mean = self._G.dot(particle)
                self._particles[i] = self._rng.multivariate_normal(mean, self._Q)
        else:
            self._particles = self._rng.normal(self._particles, self._sigma)

    def estimate(self):
        # State estimation by average of particles
        state_estimation = np.sum(self._particles * self._weights[:,None], axis=0)
        return state_estimation

        # State estimation by largest weight
        # state_estimation = self._particles[np.argmax(self._weights)]

    def verify(self, state_estimation, groundTruth):
        if groundTruth is None:
            return None

        # Bounding box of the estimation
        x1,x2,y1,y2 = get_rectangle(state_estimation, self._w_init, self._h_init)

        # Bounding box of the solution using the largest weight
        # bb_estimation = {'x1':state_estimation[0]-state_estimation[2],
        #                  'y1':state_estimation[1]-state_estimation[3],
        #                  'x2':state_estimation[0]+state_estimation[2],
        #                  'y2':state_estimation[1]+state_estimation[3]}
        if x1 < x2:
            bb_estimation = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
        elif x1 > x2:
            bb_estimation = {'x1':x2, 'y1':y1, 'x2':x1, 'y2':y2}
        else:
            bb_estimation = {'x1':x1, 'y1':y1, 'x2':x2+1, 'y2':y2}

        # Compute Intersection over Union to evaluate the solution. Higher values are 
        # better. IoU is bounded in [0,1]. In object detection an IoU >= 0.5 is usually 
        # considered a correct detection. 
        # Bounding box of the ground truth
        bb_gt = {'x1':groundTruth[0], 'y1':groundTruth[1],'x2':groundTruth[0]+groundTruth[2], 'y2':groundTruth[1]+groundTruth[3]}

        return utils.iou(bb_gt, bb_estimation)

    def create_clone_particles(self, initial):
        particles = np.tile(initial,(self._N,1))
        return particles

    def create_gaussian_particles(self, mean, std, N, dim):
        particles = np.empty((N, dim))
        for i in range(dim):
            particles[:, i] = mean[i] + (self._rng.standard_normal(N) * std[i])
        return particles

    def resample_from_index(self, indexes):
        self._particles[:] = self._particles[indexes]
        self._weights.resize(len(self._particles))
        self._weights.fill (1.0 / len(self._weights))

    def resample(self):
        indexes = systematic_resample(self._weights)
        self.resample_from_index(indexes)
        assert np.allclose(self._weights, 1/self._N)

    def apply_mgwo_optimizer(self, frame):
  
        # In case of losing object
        #TODO: can I just call initializer here?
        if not sum(self._weights)>0:
            height,width,_ = frame.shape
            self._particles = self._create_gaussian_particles([int(width/2), 1, int(height/2), 1, 1, 1], [100,5,100,5,5,1], self._N, self._dim)
            self._lh_eval.update(frame, self._particles, self._weights)
            logger.info('Object lost')

        self._mgwo.optimize(frame, self._particles, self._weights, self._sigma[4])

    def process_dataset(self, dataset, visualize = False):
        # Import images, ground truth
        images, _, gt = utils.load_data(dataset)
        initial_state = np.array([gt[0,0]+gt[0,2]/2,self._velocity[0],gt[0,1]+gt[0,3]/2,self._velocity[1], 0, 1])

        self.initialize(initial_state=initial_state, w_init=gt[0,2], h_init=gt[0,3], frame=images[0])

        state_estimate_arr = []
        iou_arr = []

        for index,frame_bgr in enumerate(images):
            if frame_bgr is not None:
                logging.info(f"Processing frame: {index} / {len(images)}")
                state_estimation, iou = self.step(frame_bgr, gt[index,:])

                iou_arr.append(iou)
                state_estimate_arr.append(state_estimation)

                #logging.info('IOU = {:.3f}'.format(iou))

                if visualize:
                    self._visu.draw_frame_cv2(frame_bgr, self._particles, state_estimation, index, self._w_init, self._h_init, iou)
                    best_particle = self._particles[np.argmax(self._weights)]
                    self._visu.draw_box(best_particle, self._w_init, self._h_init, color=(255,0,0))
                    self._visu.show()

        iou_avg = sum(iou_arr)/(len(iou_arr))
        logging.info('IOU avg  = {:.3f}'.format(iou_avg))

    def step(self, frame, gt = None):
        # 1. Move particles
        self.predict()

        # 2. Evaluate particles and calculate weights
        self._lh_eval.update(frame, self._particles, self._weights)

        # 3. Apply Modified Gray Wolf Optimizer       
        if self._use_mgwo:
            self.apply_mgwo_optimizer(frame)
        
        # 4. Estimate current state
        state_estimation = self.estimate()

        # 5. Verify
        iou = self.verify(state_estimation, gt)

        # 5. Resample
        self.resample()

        return state_estimation, iou
        