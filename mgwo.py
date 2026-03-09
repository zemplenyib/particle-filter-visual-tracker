import numpy as np
from display_images import Visu



EPSILON = 0.000001

class MGWOptimizer:
    def __init__(self, mgwo_max_iter, a, rng, lh_eval):
        self.lh_eval = lh_eval
        self.mgwo_max_iter = mgwo_max_iter
        self.a = a
        self.rng = rng


    def optimize(self, frame, particles, weights, std_mgwo):
        N = particles.shape[0]
        dim = particles.shape[1]

        self.a = 2
        particles_new = np.empty((N, dim))
        weights_new = np.empty((N, 1))

        for t in range (self.mgwo_max_iter):
            r1 = 0.5 + self.rng.standard_normal((N, 3, dim)) * std_mgwo
            r2 = 0.5 + self.rng.standard_normal((N, 3, dim)) * std_mgwo
            A = 2*self.a*r1-self.a
            C = 2*r2
            
            ind = np.argpartition(weights, -3)[-3:]
            ind = ind[np.argsort(weights[ind])]

            alpha = ind[2]
            beta = ind[1]
            delta = ind[0]

            X_alpha = particles[alpha]
            X_beta = particles[beta]
            X_delta = particles[delta]

            # visu = Visu()
            # visu.start_frame(frame)
            # visu.draw_particles(np.array([X_alpha, X_beta, X_delta]))
            # visu.draw_all_box(particles, 80, 111)
            # visu.show()

            # Modify particles
            D_alpha = np.abs(C[:,0,:] * X_alpha - particles)
            D_beta  = np.abs(C[:,1,:] * X_beta  - particles)
            D_delta = np.abs(C[:,2,:] * X_delta - particles)

            X_1 = X_alpha - A[:,0,:] * D_alpha
            X_2 = X_beta  - A[:,1,:] * D_beta
            X_3 = X_delta - A[:,2,:] * D_delta

            particles_new = (X_1 + X_2 + X_3) / 3
            
            # Update particle if new solution is better
            #TODO: make this more efficient
            self.lh_eval.update(frame, particles, weights, normalize = False)
            self.lh_eval.update(frame, particles_new, weights_new, normalize = False)
            for i,(weight, weight_new) in enumerate(zip(weights, weights_new)):
                if weight_new > weight:
                    particles[i] = particles_new[i]
                    weights[i] = weights_new[i]
            # Update control parameter
            phi = np.pi*t/self.mgwo_max_iter
            self.a = 2 - 2*(np.sin(phi/2))**2

        weights /= (np.sum(weights)+EPSILON)