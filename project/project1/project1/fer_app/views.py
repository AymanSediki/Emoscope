from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
import cv2
import numpy as np
from keras.models import load_model
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings


# Load the pre-trained RNN model
model = load_model('latof.h5')

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


def detect_emotion(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        image_name = fs.save(uploaded_image.name, uploaded_image)
        image_path = os.path.join(fs.location, image_name)
        
        # Read the uploaded image
        img = cv2.imread(image_path)
        
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.65, 6)
        
        # Process each face detected
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = img[y:y+h, x:x+w]
            
            # Predict emotion
            emotion = predict_emotion(face_roi)
            
            # Display emotion text next to the face
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the annotated image with detected emotions
        annotated_image_name = 'annotated_' + uploaded_image.name
        annotated_image_relative_path = os.path.join('media', annotated_image_name)
        annotated_image_absolute_path = os.path.join(settings.MEDIA_ROOT, annotated_image_name)
        cv2.imwrite(annotated_image_absolute_path, img)
        
        # Return the relative path of the annotated image
        return JsonResponse({'image_path': annotated_image_relative_path})
    else:
        return HttpResponse('Error: No image uploaded.')

def generate_frames():
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
        
        # Convert frame to bytes for rendering in template
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'fer_app/index.html')

