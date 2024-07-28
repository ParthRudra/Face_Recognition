# Face_Recognition
Face_Recognition and Detection on LFW Dataset

 Library Installation

python

!pip install scikit-learn matplotlib facenet-pytorch albumentations torch torchvision pillow

    This line installs the necessary libraries if they are not already installed.

2. Import Libraries

python

import ...

    Various libraries are imported, including those for data handling (scikit-learn), image processing (Pillow, matplotlib), deep learning (torch, torchvision), face recognition (facenet_pytorch), and data augmentation (albumentations).

3. Load the LFW Dataset

python

lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.5)

    The LFW dataset is loaded with images resized to 50% of their original size, and only including people with at least 20 images.

4. Get Data Attributes

python

images = lfw_people.images
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

    The images, data, target labels, and target names are extracted from the dataset.

5. Normalize the Images

python

images = images / 255.0

    The images are normalized to a range of [0, 1].

6. Split the Data

python

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

    The dataset is split into training and testing sets.

7. Initialize MTCNN and InceptionResnetV1

python

mtcnn = MTCNN(keep_all=True, device='cpu')
model = InceptionResnetV1(pretrained='vggface2').eval()

    MTCNN (Multi-task Cascaded Convolutional Networks) is initialized for face detection.
    InceptionResnetV1, pretrained on the VGGFace2 dataset, is initialized for feature extraction.

8. Data Augmentation

python

augmentations = A.Compose([ ... ])

    Augmentation techniques such as horizontal flip, brightness/contrast adjustment, and rotation are defined.

9. Define Dataset Class

python

class LFWCustomDataset(Dataset): ...

    A custom dataset class is defined to handle the images and their corresponding labels, along with optional augmentation.

10. Convert to RGB

python

def convert_to_rgb(image): ...

    A helper function is defined to convert grayscale images to RGB.

11. Transformations

python

transform = transforms.Compose([ ... ])

    Image transformations are defined, including converting images to RGB, resizing, and converting to tensor format.

12. Create Data Loaders

python

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    Data loaders for the training and testing datasets are created.

13. Define Face Recognition Model

python

class FaceRecognitionModel(nn.Module): ...

    A neural network model is defined, with a feature extractor and a fully connected layer for classification.

14. Set Up Loss and Optimizer

python

class_weights = ...
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(face_recognition_model.parameters(), lr=0.0001)

    Class weights are computed to handle class imbalance.
    Cross-entropy loss with class weights is used.
    The Adam optimizer is used with a learning rate of 0.0001.

15. Training and Evaluation Functions

python

def train(model, train_loader, criterion, optimizer, device): ...
def evaluate(model, test_loader, device): ...

    Functions are defined for training and evaluating the model.

16. Training Loop with Early Stopping

python

num_epochs = 50
best_accuracy = 0.0
patience = 5
early_stopping_counter = 0
for epoch in range(num_epochs): ...

    A training loop is defined with early stopping to prevent overfitting.

17. Testing with a New Image

python

image_url = 'https://example.com/path_to_your_image.jpg'
response = requests.get(image_url)
new_image = Image.open(BytesIO(response.content)).convert('RGB')
new_image = np.array(new_image)
faces, boxes, _ = mtcnn.detect(new_image, landmarks=True)
if faces is not None: ...

    A new image is downloaded from a URL and converted to an array.
    Faces are detected in the new image using MTCNN.
    The detected faces are prepared and passed through the model to get predictions.
    Bounding boxes with predicted labels are drawn on the image.
    The image is displayed with bounding boxes and labels.

Summary

This code provides a complete pipeline for face recognition using the LFW dataset. It includes data preprocessing, augmentation, model training with early stopping, and testing on a new image. The use of Facenet PyTorch and Albumentations enhances the robustness and accuracy of the face recognition model.
