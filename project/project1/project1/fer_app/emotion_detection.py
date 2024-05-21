import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained RNN model
model = load_model('rnn2.h5')

# Define the emotions
emotions = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Scared', 5: 'Surprised'}

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion
def predict_emotion(face_img):
    # Preprocess the image
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face_img = cv2.resize(face_img, (177, 177))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    
    # Predict the emotion
    emotion_prob = model.predict(face_img)[0]
    emotion_label = emotions[np.argmax(emotion_prob)]
    
    return emotion_label


# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.65, 6)
    
    # Process each face detected
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Predict emotion
        emotion = predict_emotion(face_roi)
        
        # Display emotion text next to the face
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()