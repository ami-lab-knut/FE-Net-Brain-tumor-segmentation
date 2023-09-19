# FE-Net-Brain-tumor-segmentation

FE-Net is a feature enhancement module designed for brain tumor segmentation. It works by enhancing each individual slice in a 3D brain MRI
by using the information from enhanced adjacent slices. It uses a correlation module as a trainable feature selector that can select
the relevant features from the adjacent slices to combine it with the main slice.

## Getting Started
### Prerequisits
* Python 3.7 or higher
* Torch
* Torchvision

