import torch
import os

from torch.utils.data import Dataset

from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os



import mimetypes
import argparse
import imutils
import pickle

### Config ###

# path to the input dataset from which we derive the images and annotation files
BASE_PATH = "/home/irikos/Work/datasets/coco/"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images/val2017"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "labels/val2017"])

BASE_OUTPUT = "output"

# define the path to the output model, label encoder, plots output directory and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


### HYPERPARAMETERS ###

# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 8

# specify the loss weights
LABELS = 1.0
BBOX = 1.0


### CUSTOM TENSOR DATASET ###
class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms
        
    def __getitem__(self, index):
        # grab the image, label and its bbox coordinates
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        
        # transpose the image such that its channel dimensions becomes the leading one
        # PyTorch requirements. Basically move from Height x Width x Channels to
        # Channels x Height x Width
        image = image.permute(2, 0, 1)
        
        # check to see if we have any image transformations to apply and, if so, apply them
        if self.transforms:
            image = self.transforms(image)
            
        # return a tuble of images, labels and bounding box coordinates
        return (image, label, bbox)
    
    def __len__(self):
        # return the size of the dataset
        return len(self.tensors[0])




### BBOX REGRESSOR ###
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses
        
        # build the regressor head for outputting the bounding box coordinates
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )
        
        # build the classifier head to predict the class labels
        self.classifier = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses)
        )
    
        # set the classifier of our base model to produce outputs from the last convolution block
        self.baseModel.fc = Identity()
        
    def forward(self, X):
        # pass the inputs through the base model and then obtain predictions
        # from two different branches of the network
        features = self.baseModel(X)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
        
        # return the outputs as a tuple
        return (bboxes, classLogits)
    
### TRAINING ###

# initialize the list of data (images), class labels, target bounding box coordinates and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

for csvPath in paths.list_files(ANNOTS_PATH, validExts=(".txt")):
    rows = open(csvPath).read().strip().split("\n")
    # loop over the rows
    for row in rows:
        # break fow into filename, bbox coords and class label
        # print(row)
        row = row.split(" ")
        if (len(row) > 1): 
            # print(row)
            (label, startX, startY, endX, endY) = row
            filename = csvPath.split("/")[-1].split('.')[-2]
            # derive the path to the input image, load the image (in openCV format) and grab its dimensions
            imagePath = os.path.sep.join([IMAGES_PATH, filename]) + ".jpg"

            # print(imagePath)
            image = cv2.imread(imagePath)
            (h, w) = image.shape[:2]

            # scale the bbox coords relative to the spatial dim for the input image

            startX = float(startX) / w
            startY = float(startY) / h
            endX = float(endX) / w
            endY = float(endY) / h

            # load the image and preprocess it
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            # update our lists of data, class labels, bboxes and image paths
            data.append(image)
            labels.append(label)
            bboxes.append((startX, startY, endX, endY))
            imagePaths.append(imagePath)

data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# perform label encoding on the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 80% - 20%
split = train_test_split(data, labels, bboxes, imagePaths, test_size=0.2, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# convert NumPy arrays to PyTorch tensors
(trainImages, testImages) = torch.tensor(trainImages), torch.tensor(testImages)
(trainLabels, testLabels) = torch.tensor(trainLabels), torch.tensor(testLabels)
(trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes), torch.tensor(testBBoxes)

# define normalization transforms
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# convert NumPy arrays to PyTorch datasets
trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes), transforms=transforms)
testDS = CustomTensorDataset((testImages, testLabels, testBBoxes), transforms=transforms)
print("[INFO] total training samples: {}...".format(len(trainDS)))
print("[INFO] total test samples: {}...".format(len(testDS)))

# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // BATCH_SIZE
valSteps = len(testDS) // BATCH_SIZE

# create data loaders
trainLoader = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), 
                         pin_memory=PIN_MEMORY)

testLoader = DataLoader(testDS, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), 
                         pin_memory=PIN_MEMORY)


# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()


# load the ResNet50 network as a base netowrk
resnet = resnet50(pretrained=True)

# freeze all ResNet50 layers so they will not be updated during the training process
for params in resnet.parameters():
    params.requires_grad = False

# create or custom object detector model and flash it to the current device
objectDetector = ObjectDetector(resnet, len(le.classes_))
objectDetector = objectDetector.to(DEVICE)

# define our loss functions
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

# initialize the optimizer, compile the model and show the model summary
opt = Adam(objectDetector.parameters(), lr=INIT_LR)
print(objectDetector)

# initialize a dictionary to store training history
H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [], "val_class_acc": []}


# loop over epochs
print("[INFO] training the network...")

startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)):
    objectDetector.train()
    
    # initialize the total trainign and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    
    # initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0
    
    for (images, labels, bboxes) in trainLoader:
        (images, labels, bboxes) = (images.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE))
        
        # perform a forward pass and calculate the training loss
        predictions = objectDetector(images)
        bboxLoss = bboxLossFunc(predictions[0], bboxes)
        classLoss = classLossFunc(predictions[1], labels)
        totalLoss = (BBOX * bboxLoss) + (LABELS * classLoss)
        
        # zero out the gradients, peform the backpropagation step and upgrade the weights
        opt.zero_grad()
        totalLoss.backward()
        opt.step()
        
        # add the loss to the total training loss so far and calcul;ate the number of correct predictions
        totalTrainLoss += totalLoss
        trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()
        
    # switch off autograd
    with torch.no_grad():
        objectDetector.eval()

        # loop over the validatoon set

        for (images, labels, bboxes) in testLoader:
            (images, labels, bboxes) = (images.to(DEVICE), labels.to(DEVICE), bboxes.to(DEVICE))
            
            # make predictions and calculate the validation loss
            predictions = objectDetector(images)
            bboxLoss = bboxLossFunc(predictions[0], bboxes)
            classLoss = classLossFunc(predictions[1], labels)
            
            totalLoss = (BBOX * bboxLoss) + (LABELS * classLoss)
            totalValLoss += totalLoss
            # calculate the number of correct predictions
            valCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

    # calculate tthe average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    
    # calculate the training and vaalidation accuracy
    trainCorrect = trainCorrect / len(trainDS)
    valCorrect = valCorrect / len(testDS)
    
    # update our training history
    H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_class_acc"].append(trainCorrect)
    H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_class_acc"].append(valCorrect)
    
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(avgValLoss, valCorrect))
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    


# serialize the model to disk
print("[INFO] saving object detector model...")
torch.save(objectDetector, MODEL_PATH)
# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="total_train_loss")
plt.plot(H["total_val_loss"], label="total_val_loss")
plt.plot(H["train_class_acc"], label="train_class_acc")
plt.plot(H["val_class_acc"], label="val_class_acc")
plt.title("Total Training Loss and Classification Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# save the training plot
plotPath = os.path.sep.join([PLOTS_PATH, "training.png"])
plt.savefig(plotPath)