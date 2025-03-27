# Usage Examples

## Basic Setup

Before running the application, make sure you have:
1. Installed all required dependencies (`pip install -r requirements.txt`)
2. Downloaded the SAM model weights (`sam_vit_b_01ec64.pth`)
3. Updated the model path in `inpainting_app.py` to point to your downloaded model

## Typical Workflow

1. Start the application by running:
   ```
   python inpainting_app.py
   ```

2. Click "Load Image" and select an image containing objects you want to remove.

3. Click "Start Auto Inpaint" to begin the automatic inpainting process:
   - The system will find the most visually salient (attention-grabbing) object
   - It will segment that object using the Segment Anything Model (SAM)
   - It will remove the object using advanced inpainting
   - The process repeats, progressively removing objects in order of their visual saliency

4. Click "Stop Auto Inpaint" at any time to pause the process.

## Tips for Best Results

- **Image Selection**: Works best with images that have distinct foreground objects
- **Processing Time**: Each iteration takes a few seconds depending on your hardware
- **Model Path**: Make sure to update the SAM model path in the code to match your system
- **Iterative Approach**: Let the process run for multiple iterations to progressively clean up the image

## Understanding the Process

The system works by finding regions in the image that attract visual attention using spectral residual saliency detection, a technique that analyzes the frequency domain of the image to identify visually important regions.

For each iteration:
1. A spectral residual saliency map is computed
2. The most salient point is identified 
3. This point is fed to SAM, which precisely segments the object at that location
4. The segmented object is removed using inpainting algorithms
5. The process repeats on the modified image

## Example Use Cases

- Removing distracting elements from photos
- Cleaning up product images
- Creating clean backgrounds for compositing
- Removing unwanted objects from landscapes
- Educational tool for understanding computer vision techniques
