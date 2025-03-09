import cv2
import matplotlib.pyplot as plt

# Load the image
imagePath = 'input_image.jpg'

# Read the image in BGR format
img = cv2.imread(imagePath)
img.shape
(4000, 2667, 3)

# Convert the image to Grayscale then RGB format
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image.shape
(4000, 2667)

# Load the Haar Cascade Classifier for Face Detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Detect faces in the image
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors = 5, minSize = (40, 40)
)

# Draw bounding box around the detected faces
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

# Convert and display the image in RGB format
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# Access the Webcam
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_image, 1.1, 5, minSize = (40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 4)
    return faces    

# Creating loop for Real-time Face Detection
while True:
    result, video_frame = video_capture.read() # read frames from the video
    if result is False:
        break  # Break the loop if there is no frame to read

    faces = detect_bounding_box(
        video_frame # Apply function to detect faces in the frame
    )

    cv2.imshow(
        "Face Detection Project", video_frame 
    ) # Display the frame with bounding box

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Break the loop if 'q' is pressed

video_capture.release() # Release the video capture object
cv2.destroyAllWindows() # Close all windows