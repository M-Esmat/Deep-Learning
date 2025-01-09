import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Flatten, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# Define paths
data_path = "./cifar-10"
labels_path = os.path.join(data_path, "trainLabels.csv")
train_dir = os.path.join(data_path, "train")
reduced_dataset_dir = os.path.join(data_path, "reduced_train")
for_test_dir = os.path.join(data_path, "for_test_dir")

# Parameters
image_size = (32, 32)
num_classes = 10
batch_size = 32
epochs = 10

# Step 1: Prepare Reduced Dataset
print("Preparing reduced dataset...")
labels_df = pd.read_csv(labels_path)
reduced_df = labels_df.groupby("label").apply(lambda x: x.sample(n=4000, random_state=42)).reset_index(drop=True)
os.makedirs(reduced_dataset_dir, exist_ok=True)

for _, row in reduced_df.iterrows():
    img_id = row["id"]
    img_label = row["label"]
    src_path = os.path.join(train_dir, f"{img_id}.png")
    dst_dir = os.path.join(reduced_dataset_dir, img_label)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{img_id}.png")
    shutil.copy(src_path, dst_path)

print("Reduced dataset created successfully.")

# Step 2: Prepare Unused Test Dataset
print("Preparing test dataset...")
all_ids = set(labels_df["id"])
used_ids = set(reduced_df["id"])
unused_ids = all_ids - used_ids
os.makedirs(for_test_dir, exist_ok=True)

for img_id in unused_ids:
    img_label = labels_df.loc[labels_df["id"] == img_id, "label"].values[0]
    src_path = os.path.join(train_dir, f"{img_id}.png")
    dst_dir = os.path.join(for_test_dir, img_label)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{img_id}.png")
    shutil.copy(src_path, dst_path)

print("Unused images moved to test dataset.")

# Step 3: Load Datasets
def load_dataset_from_directory(directory, image_size, batch_size):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

train_data = load_dataset_from_directory(reduced_dataset_dir, image_size, batch_size)
test_data = load_dataset_from_directory(for_test_dir, image_size, batch_size)

# Normalize images
train_data = train_data.map(lambda x, y: (x / 255.0, y))
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# Step 4: Define the Model
print("Building the model...")
base_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False

model = Sequential([
    UpSampling2D(size=(7, 7), input_shape=(32, 32, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax", name="classification")
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Step 5: Train the Model
print("Training the model...")
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss")
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

# Step 6: Evaluate the Model
print("Evaluating the model...")
eval_results = model.evaluate(test_data)
print(f"Loss: {eval_results[0]:.4f}, Accuracy: {eval_results[1]:.4f}")

# Step 7: Generate Predictions and Submission File
def generate_submission(test_data, model, output_file):
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predictions = []
    image_ids = []

    for images, labels in test_data:
        preds = model.predict(images)
        predicted_classes = tf.argmax(preds, axis=1).numpy()
        predictions.extend(predicted_classes)
        image_ids.extend(labels.numpy())

    predicted_labels = [label_names[i] for i in predictions]
    submission_df = pd.DataFrame({"id": image_ids, "label": predicted_labels})
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

print("Generating submission file...")
generate_submission(test_data, model, "submission.csv")
