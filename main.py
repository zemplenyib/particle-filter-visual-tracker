from particle_filter_tracker import ParticleFilterTracker
import os

std_x = 1.73     # [pix]
std_y = 1.8      # [pix]
std_theta = 0.03 # [rad]
std_s = 0.023    # [unit]

pfTracker = ParticleFilterTracker(100,
                                  [std_x,std_y,std_theta,std_s],
                                  [1,1],
                                  1,
                                  True,
                                  10)

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Datasets', 'BlurBody')
pfTracker.process_dataset(data_path, visualize=True)