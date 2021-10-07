# lane_seg
## Requirements:
Download datasets: http://apolloscape.auto/lane_segmentation.html
```
opencv==3.4
Pillow==5.1.0
```
```
$ conda create -n lane_seg python=3.6
$ conda install Pillow==5.1.0
```
## Usage:
For color images:
```
$ python viz.py /<path_to_Apolloscape_dataset>/ColorImage_road02/ColorImage/ --img_glob *.jpg
```
For label images:
```
$ python viz.py /<path_to_Apolloscape_dataset>/Labels_road02/Label/
```
