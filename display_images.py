import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visu:
    def __init__(self):
        pass

    @staticmethod
    def display_frame_cv2(frame, particles, estimation, index, w_init, h_init, iou = None):
        if iou is None:
            color = (0, 0, 255)
        elif iou <= 0.5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        vis_frame = frame.copy()
        vis_frame = draw_particles_cv2(vis_frame, particles, w_init, h_init)
        vis_frame = draw_box_cv2(vis_frame, estimation, w_init, h_init, color)

        # Add text info
        cv2.putText(vis_frame, f"Frame: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"IoU: {iou:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Particle Filter MGWO Tracker', vis_frame)
        # Handle user input
        key = cv2.waitKey(30) & 0xFF
        # 27: ASCII code for ESC
        if key == 27:
            exit()

    @staticmethod
    def draw_particles_cv2(img, particles, w_init, h_init, color=(0, 0, 255), thickness=1):
        return draw_particles_cv2(img, particles, w_init, h_init, color, thickness)

    @staticmethod
    def draw_box_cv2(img, particle, w_init, h_init, color=(255, 0, 0), thickness=2):
        return draw_box_cv2(img, particle, w_init, h_init, color, thickness)

    @staticmethod
    def display_image(img, w_init, h_init, title='', size=None, show_axis=False, particles=None, weights=None, t=None):
        return display_image(img, w_init, h_init, title, size, show_axis, particles, weights, t)

    @staticmethod
    def display_images(ima1, ima2, title1='', title2='', size=None, show_axis=False, hsep=0.1):
        return display_images(ima1, ima2, title1, title2, size, show_axis, hsep)

    @staticmethod
    def display_scatter(img, title='', size=None, show_axis=False, particles=None, weights=None):
        return display_scatter(img, title, size, show_axis, particles, weights)


def close_figure(event):
    if event.key == 'escape':
        exit()

def get_rectangle(particle, w_init, h_init):
  theta = np.deg2rad(particle[4]).item()
  s = particle[5]
  w = (np.cos(theta)*w_init*s).item()
  h = h_init*s
  x1 = particle[0] - w/2
  x2 = particle[0] + w/2
  y1 = particle[2] - h/2
  y2 = particle[2] + h/2

  return x1.astype(int),x2.astype(int),y1.astype(int),y2.astype(int)

def draw_particles_cv2(img, particles, w_init, h_init, color=(0, 0, 255), thickness=1):
    """Draws particles on an OpenCV image."""
    img_disp = img.copy()
    if particles is not None:
        if particles.ndim == 1:
             particles = particles[np.newaxis, :]
        
        for particle in particles:
            x1, x2, y1, y2 = get_rectangle(particle, w_init, h_init)
            # Draw center point
            cv2.circle(img_disp, (int(particle[0]), int(particle[2])), 1, color, -1)
    return img_disp

def draw_box_cv2(img, particle, w_init, h_init, color=(255, 0, 0), thickness=2):
    """Draws the bounding box of a single particle/state."""
    x1, x2, y1, y2 = get_rectangle(particle, w_init, h_init)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# Function to display one image
def display_image(img, w_init, h_init, title='', size=None, show_axis=False, particles = None, weights = None, t = None):
    img_disp = img.copy()
    if not show_axis:
      plt.axis('off')

    plt.figure(1)
    plt.clf()
    if particles is not None:
        if particles.shape == (6,):
            particle = particles
            x1,x2,y1,y2 = get_rectangle(particle,w_init,h_init)
            cv2.rectangle(img_disp,(x1, y1),(x2, y2),(255,0,0),2)
        else:
            for i,particle in enumerate(particles):
                x1,x2,y1,y2 = get_rectangle(particle,w_init,h_init)
                cv2.rectangle(img_disp,(x1, y1),(x2, y2),(255,0,0),2)
    h = plt.imshow(img_disp, interpolation='none')
    if size:
        factor = 2
        dpi = h.figure.get_dpi()/size*factor
        h.figure.set_figwidth(img.shape[1] / dpi)
        h.figure.set_figheight(img.shape[0] / dpi)
        h.figure.canvas.resize(img.shape[1] + 1, img.shape[0] + 1)
        h.axes.set_position([0, 0, 1, 1])

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50,100,640,600)

        if show_axis:
            h.axes.set_xlim(-1, img.shape[1])
            h.axes.set_ylim(img.shape[0], -1)
    plt.grid(False)
    plt.title(title)
    if t is None:
        plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
        plt.show(block = False)
        plt.waitforbuttonpress()
    else:
        plt.show(block = False)
        plt.pause(t)

# Function to display 2 images side by side
def display_images(ima1, ima2, title1='', title2='', size=None, show_axis=False, hsep=0.1):
    fig, ax = plt.subplots(1,2)
    plt.grid(False)
    h = ax[0].imshow(ima1.astype(np.uint8), cmap=plt.cm.gray)
    ax[0].set_title(title1)

    if size:
        dpi = h.figure.get_dpi()/size
        h.figure.set_figwidth(ima1.shape[1] / dpi)
        h.figure.set_figheight(ima1.shape[0] / dpi)
        h.figure.canvas.resize(ima1.shape[1] + 1, ima1.shape[0] + 1)
        h.axes.set_position([0, 0, 1, 1])

    if not show_axis:
        ax[0].axis('off')
    else: 
        ax[0].axes.set_xlim(-1, ima1.shape[1])
        ax[0].axes.set_ylim(ima1.shape[0], -1)

    h = ax[1].imshow(ima2.astype(np.uint8), cmap=plt.cm.gray)
    ax[1].set_title(title2)

    if size:
        dpi = h.figure.get_dpi()/size
        h.figure.set_figwidth(ima2.shape[1] / dpi)
        h.figure.set_figheight(ima2.shape[0] / dpi)
        h.figure.canvas.resize(ima2.shape[1] + 1, ima2.shape[0] + 1)
        h.axes.set_position([1+hsep, 0, 1, 1])

    if not show_axis:
        ax[1].axis('off')
    else: 
        ax[1].axes.set_xlim(-1, ima2.shape[1])
        ax[1].axes.set_ylim(ima2.shape[0], -1)

    plt.tight_layout()
    plt.show()

# Function to display image with scatter
def display_scatter(img, title='', size=None, show_axis=False, particles = None, weights = None):
    if not show_axis:
      plt.axis('off')

    plt.figure(1)
    h = plt.imshow(img, interpolation='none')
    if particles is not None:
        particles = particles.astype(int)
        for i,particle in enumerate(particles):
            plt.scatter(x = particle[0]-particle[2], y = particle[1]-particle[3], s = weights[i]*10, c = 'Red')
    if size:
        factor = 2
        dpi = h.figure.get_dpi()/size*factor
        h.figure.set_figwidth(img.shape[1] / dpi)
        h.figure.set_figheight(img.shape[0] / dpi)
        h.figure.canvas.resize(img.shape[1] + 1, img.shape[0] + 1)
        h.axes.set_position([0, 0, 1, 1])

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50,100,640, 545)

        if show_axis:
            h.axes.set_xlim(-1, img.shape[1])
            h.axes.set_ylim(img.shape[0], -1)
    plt.grid(False)
    plt.title(title)
    plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
    plt.show(block = False)
    plt.waitforbuttonpress()
