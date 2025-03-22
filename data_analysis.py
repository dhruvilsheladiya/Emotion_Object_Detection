import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("detection_log.csv")

# Count occurrences of each object
object_counts = df.iloc[:, 0].value_counts()
print("\n--- Object Counts ---\n", object_counts)

# Count occurrences of each color
color_counts = df.iloc[:, 1].value_counts()
print("\n--- Color Counts ---\n", color_counts)

# Count occurrences of each emotion
emotion_counts = df.iloc[:, 2].value_counts()
print("\n--- Emotion Counts ---\n", emotion_counts)

# Plot bar charts for better visualization
plt.figure(figsize=(10, 5))

# Plot Object Count
plt.subplot(1, 3, 1)
object_counts.plot(kind='bar', color='skyblue')
plt.title("Detected Objects")
plt.xticks(rotation=45)

# Plot Color Count
plt.subplot(1, 3, 2)
color_counts.plot(kind='bar', color='orange')
plt.title("Detected Colors")
plt.xticks(rotation=45)

# Plot Emotion Count
plt.subplot(1, 3, 3)
emotion_counts.plot(kind='bar', color='green')
plt.title("Detected Emotions")
plt.xticks(rotation=45)

# Show the graphs
plt.tight_layout()
plt.show()
