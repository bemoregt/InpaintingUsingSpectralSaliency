# InpaintingUsingSpectralSaliency

An automatic image inpainting application that combines spectral residual saliency detection with Segment Anything Model (SAM) to seamlessly remove objects from images.

## Overview

This application automates the inpainting process by:
1. Detecting salient regions using spectral residual analysis (OpenFV)
2. Segmenting those regions using Meta AI's Segment Anything Model (SAM)
3. Applying advanced inpainting algorithms to remove the segmented objects

The process iteratively removes the most visually prominent objects from the image, allowing for controlled object removal without manual masking.

## Features

- **Automatic object detection** using spectral residual saliency maps
- **Precise segmentation** with SAM (Segment Anything Model)
- **Advanced inpainting** combining TELEA and NS algorithms for better results
- **Iterative processing** that continuously removes objects in order of visual prominence
- **Simple user interface** for loading images and controlling the inpainting process

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- tkinter
- SAM (Segment Anything Model) - [GitHub](https://github.com/facebookresearch/segment-anything)
- OpenFV - Frequency Domain Computer Vision package

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bemoregt/InpaintingUsingSpectralSaliency.git
   cd InpaintingUsingSpectralSaliency
   ```

2. Install dependencies:
   ```bash
   pip install numpy opencv-python pillow torch torchvision
   pip install git+https://github.com/facebookresearch/segment-anything.git
   pip install openfv
   ```

3. Download the SAM checkpoint:
   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

## Usage

1. Run the application:
   ```bash
   python inpainting_app.py
   ```

2. Use the interface:
   - Click "Load Image" to select an image file
   - Click "Start Auto Inpaint" to begin the automatic removal process
   - The process will iteratively remove salient objects
   - Click "Stop Auto Inpaint" to halt the process at any time

## How It Works

1. **Spectral Residual Analysis**:
   - Converts image to grayscale
   - Computes the spectral residual saliency map
   - Identifies the most visually salient point

2. **SAM Segmentation**:
   - Uses the salient point as input to SAM
   - Generates a precise mask for the object at that location

3. **Advanced Inpainting**:
   - Applies both TELEA and NS inpainting algorithms
   - Blends the results for superior quality
   - Updates the image for the next iteration

## Technical Notes

- The size of the spectral residual kernel is dynamically calculated based on the image dimensions
- The application uses a blended approach of two inpainting algorithms for better results
- Processing runs in a separate thread to keep the UI responsive

## License

MIT

## Author

@bemoregt
