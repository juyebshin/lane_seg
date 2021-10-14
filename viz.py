import os
import sys
import time
import glob
import argparse
import cv2
import numpy as np
from PIL import Image

from collections import namedtuple

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label.
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #           name     id trainId      category  catId hasInstances ignoreInEval            color
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
    Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
    Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),
    Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
    Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
    Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
    Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
    Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
# color to label
color2label     = { label.color   : label for label in labels}

print(cv2.__spec__)
print(np.__version__)
print(Image.__spec__)

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    # self.sizer = [height, width]
    self.i = 0
    self.skip = 1
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(0)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        folders = os.listdir(basedir)
        if len(folders) == 0:
            print('There are no folder at directory %s. Check the data path.' % (basedir))
        else:
            print('There are %d folders to be processed.' % (len(folders)))
        folders.sort()

        self.listing = []
        for folder in folders:
          current_dir = os.path.join(basedir, folder)
          current_dir = os.path.join(current_dir, 'Camera 5/')
          search = os.path.join(current_dir, img_glob)
          # print('Folder: {folder}'.format(folder=search))
          cur_listing = glob.glob(search)
          cur_listing.sort()
          print('Files: {file}'.format(file=cur_listing))
          if cur_listing == 0:
            raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')
          self.listing.extend(cur_listing)
          # self.listing = self.listing[::self.skip]
          self.maxlen = len(self.listing)

  def read_image(self, impath):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      im: float32 numpy array sized H x W with values in range [0, 1].
    """
    print('Image path: {path}'.format(path=impath))
    im = cv2.imread(impath, 1)
    if im is None:
      raise Exception('Error reading image %s' % impath)
    # im = (im.astype('float32') / 255.)
    return im

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      # input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
      #                          interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file)
    # Increment internal counter.
    self.i = self.i + 1
    # input_image = input_image.astype('float32')
    return (input_image, True)

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='ApolloScape Dataset Visualizer.')
    parser.add_argument('input', type=str, default='',
        help='Image directory or movie file or "camera" (for webcam).')
    # parser.add_argument('--H', type=int, default=271,
    #     help='Input image height (default: 120).')
    # parser.add_argument('--W', type=int, default=338,
    #     help='Input image width (default:160).')
    parser.add_argument('--scale', type=float, default=0.3,
        help='Scale of image to display')
    parser.add_argument('--img_glob', type=str, default='*.png',
        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--show_extra', action='store_true',
        help='Show extra debug outputs (default: False).')
    parser.add_argument('--display_scale', type=float, default=0.5, # default=2
        help='Factor to scale output visualization (default: 2).')
    parser.add_argument('--waitkey', type=int, default=33,
        help='OpenCV waitkey time in ms (default: 1).')
    # parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
    #     help='Path to pretrained weights file (default: superpoint_v1.pth).')
    # parser.add_argument('--skip', type=int, default=1,
    #     help='Images to skip if input is movie or directory (default: 1).')
    # parser.add_argument('--min_length', type=int, default=2,
    #     help='Minimum length of point tracks (default: 2).')
    # parser.add_argument('--max_length', type=int, default=5,
    #     help='Maximum length of point tracks (default: 5).')
    # parser.add_argument('--nms_dist', type=int, default=4,
    #     help='Non Maximum Suppression (NMS) distance (default: 4).')
    # parser.add_argument('--conf_thresh', type=float, default=0.015,
    #     help='Detector confidence threshold (default: 0.015).')
    # parser.add_argument('--nn_thresh', type=float, default=0.7,
    #     help='Descriptor matching threshold (default: 0.7).')
    # parser.add_argument('--camid', type=int, default=0,
    #     help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    # parser.add_argument('--cuda', action='store_true',
    #     help='Use cuda GPU to speed up network processing speed (default: False)')
    # parser.add_argument('--no_display', action='store_true',
    #     help='Do not display images to screen. Useful if running remotely (default: False).')
    # parser.add_argument('--write', action='store_true',
    #     help='Save output frames to a directory (default: False)')
    # parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
    #     help='Directory where to write output frames (default: tracker_outputs/).')
    opt = parser.parse_args()
    print(opt)

    # This class helps load input images from different sources.
    vs = VideoStreamer(opt.input, opt.img_glob)

    win = 'ApolloScape lane_segmentation Labels'
    cv2.namedWindow(win)

    while True:
        img, status = vs.next_frame()
        if status is False:
            break
        # Image is resized via opencv just for display.
        img = cv2.resize(img, (0, 0), fx=opt.scale, fy=opt.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(win, img)
        key = cv2.waitKey(opt.waitkey) & 0xFF
        if key == ord('q'):
            print('Quitting, \'q\' pressed.')
            break

        # end while