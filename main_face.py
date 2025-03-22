import cv2
import numpy as np
import face_recognition
import pyttsx3
import csv
import matplotlib.pyplot as plt
from collections import Counter
from deepface import DeepFace
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Dynamically select a valid voice
voices = engine.getProperty('voices')
selected_voice = voices[1].id if len(voices) > 1 else voices[0].id  # Avoid IndexError
engine.setProperty('voice', selected_voice)

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

# Open webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 15)

# Define color ranges in HSV
color_ranges = {
    "Red": [(0, 120, 70), (10, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Green": [(36, 100, 100), (86, 255, 255)],
    "Blue": [(90, 100, 100), (128, 255, 255)],
    "Purple": [(129, 50, 70), (158, 255, 255)],
    "Orange": [(10, 100, 20), (25, 255, 255)],
    "Black": [(0, 0, 0), (180, 255, 30)],
    "White": [(0, 0, 231), (180, 20, 255)]
}

# Function to detect color
def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if np.sum(mask) > 1000:  # Ensure the detected color is significant
            return color_name
    return "Unknown"

# Function to save detection data to CSV
def save_detection(object_name, color, emotion):
    with open("detection_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([object_name, color, emotion])

# Function to analyze detection data
def analyze_data():
    objects = []
    colors = []
    emotions = []

    try:
        with open("detection_log.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 3:
                    objects.append(row[0])
                    colors.append(row[1])
                    emotions.append(row[2])

        # Count occurrences
        obj_count = Counter(objects)
        color_count = Counter(colors)
        emotion_count = Counter(emotions)

        # Plot data
        plt.figure(figsize=(10, 5))

        # Object Detection Stats
        plt.subplot(1, 3, 1)
        plt.bar(obj_count.keys(), obj_count.values(), color="skyblue")
        plt.xticks(rotation=90)
        plt.title("Detected Objects")

        # Color Detection Stats
        plt.subplot(1, 3, 2)
        plt.bar(color_count.keys(), color_count.values(), color="lightcoral")
        plt.xticks(rotation=90)
        plt.title("Detected Colors")

        # Emotion Stats
        plt.subplot(1, 3, 3)
        plt.bar(emotion_count.keys(), emotion_count.values(), color="lightgreen")
        plt.xticks(rotation=90)
        plt.title("Detected Emotions")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No data available for analytics.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Couldn't capture frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    for face in face_locations:
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Extract face for emotion detection
        face_image = frame[top:bottom, left:right]

        try:
            analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']

            # Define text responses
            emotion_texts = {
                "happy": "You are smiling!",
                "sad": "You look sad!",
                "angry": "You seem angry!",
                "surprise": "You look surprised!",
                "neutral": "You look neutral!"
            }

            text = emotion_texts.get(emotion, None)
            if text:
                engine.say(text)
                engine.runAndWait()

            # Display detected emotion
            cv2.putText(frame, emotion.capitalize(), (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print("Error in emotion detection:", e)
            emotion = "Unknown"

    # Object Detection
    results = model(frame)
    for result in results:
        for obj in result.boxes:
            class_id = int(obj.cls[0])
            confidence = float(obj.conf[0])

            class_names = model.names
            
            if confidence > 0.5 and class_id in class_names:
                object_name = class_names[class_id]
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                object_image = frame[y1:y2, x1:x2]

                detected_color = detect_color(object_image)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                engine.say(f"I see a {detected_color} {object_name}")
                engine.runAndWait()

                cv2.putText(frame, f"{detected_color} {object_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Save detection data
                save_detection(object_name, detected_color, emotion)

    cv2.imshow('Emotion & Object Detector', frame)

    # Press 'q' to exit and analyze data
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Analyze Data
analyze_data()
