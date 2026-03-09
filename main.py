from particle_filter_tracker import ParticleFilterTracker
import os

# BlurBody
#std_x = 1.73      # [pix]
#std_y = 1.5       # [pix]
#std_theta = 0.2   # [rad]
#std_s = 0.04     # [unit]
#std_mgwo = 0.0458 # [unit]

# Box
std_x = 2      # [pix]
std_y = 3       # [pix]
std_theta = 0.2   # [rad]
std_s = 0.04     # [unit]
std_mgwo = 0.0458 # [unit]

pfTracker = ParticleFilterTracker(100,
                                  [std_x,std_y,std_theta,std_s, std_mgwo],
                                  [1,1],
                                  1,
                                  True,
                                  10)

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Datasets', 'Skating1')
pfTracker.process_dataset(data_path, visualize=True)