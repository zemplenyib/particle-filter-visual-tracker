from particle_filter_tracker import ParticleFilterTracker
import os

std_x = 1.73 #1.4 #6 #1.4
std_y = 1.5 #0.64 #1.4 #6 #1.4
std_theta = 0.03 #1.96 #2.5
std_s = 0.023 #0.025
std_mgwo = 0.0458 #0.1 #0.05 #0.1

pfTracker = ParticleFilterTracker(100,
                                  [std_x,std_y,std_theta,std_s, std_mgwo],
                                  [1,1],
                                  1,
                                  True,
                                  10)

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Datasets', 'BlurBody')
pfTracker.process_dataset(data_path, visualize=True)