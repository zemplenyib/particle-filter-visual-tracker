from particle_filter_tracker import ParticleFilterTracker
import os

sigma_x = 1.73 #1.4 #6 #1.4
sigma_y = 0.64 #1.4 #6 #1.4
sigma_theta = 1.96 #2.5
sigma_s = 0.023 #0.025
sigma_mgwo = 0.0458 #0.1 #0.05 #0.1

pfTracker = ParticleFilterTracker(100,
                                  [sigma_x,sigma_y,sigma_theta,sigma_s, sigma_mgwo],
                                  [1,1],
                                  1,
                                  True,
                                  10)

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Datasets', 'BlurBody')
pfTracker.run_pf(data_path)