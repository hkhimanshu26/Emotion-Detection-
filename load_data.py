def load_data(dataset_path):
images = []
labels = []
for emotion in EMOTIONS:
emotion_folder = os.path.join(dataset_path, emotion)
for img_name in os.listdir(emotion_folder):
img_path = os.path.join(emotion_folder, img_name)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48)) # Resize image to 48x48
images.append(img)
labels.append(emotion)
images = np.array(images)
labels = np.array(labels)
images = images / 255.0 # Normalize pixel values (0 to 1)
images = images.reshape(-1, 48, 48, 1) # Add channel dimension (grayscale)
le = LabelEncoder()
labels = le.fit_transform(labels) # Convert labels to numeric
labels = tf.keras.utils.to_categorical(labels, num_classes=7) # One-hot encode
return images, labels
