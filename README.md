# OpenCV Face Scan

## About

This project utilizes OpenCV and Python to perform face detection on images and real-time video streams. It uses Haar Cascade classifiers, a machine learning object detection method used to identify objects in an image or video stream. The application can process both static images and live video feeds from a webcam, detecting and highlighting faces with bounding boxes.

## Getting Started

### Prerequisites

To run this application, you'll need the following installed on your system:
- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

This project depends on OpenCV and Matplotlib, which can be installed using pip. Below are the necessary libraries:
- `opencv-python` for OpenCV functionalities.
- `matplotlib` for displaying images.

To install these dependencies, run the following command:

```bash
pip install opencv-python matplotlib
```
#### Running the Application

#### Static Image Face Detection
- Place the image you want to process in the project directory and rename it to input_image.jpg or update the imagePath variable in the script.
- Run the script to detect faces in the image. The image will be displayed with faces highlighted by green rectangles.

#### Real-time Video Face Detection
- Ensure a webcam is connected and properly set up on your system.
- Run the script. The webcam will activate, and the video stream will display in a window with faces highlighted by green rectangles.
- Press 'q' to quit the webcam and close all windows.

##### Code Explanation

##### Image Processing
The code reads an image in BGR format, converts it to grayscale to reduce complexity, and then applies a Haar Cascade classifier to detect faces.

##### Video Processing
The webcam is accessed using OpenCV's VideoCapture. Frames are read from the video stream, processed to detect faces, and displayed in real-time. The loop continues until 'q' is pressed.

###### Limitations
The accuracy of face detection might vary depending on the lighting conditions and the quality of the webcam.
Haar Cascades are efficient but might not work as well on non-frontal faces or faces in profile.
