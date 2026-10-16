import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import datasets
import numpy as np
import argparse
import locale
import os
import matplotlib.pyplot as plt
import cv2



# CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to model")
ap.add_argument("-s", "--size", type=int, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

size = args["size"]

# Load the trained model
model = keras.models.load_model(args["model"])

# Price labels from HousesInfo.txt
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# Build the 2x2 room mosaics (same size as used in training)
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"], size=size)

# Preview a few houses with their prices

cv2.imshow("test",images[0])
cv2.waitKey(0)

num_images = 10
for index in range(0, num_images):
	plt.subplot(2, (num_images//2), index + 1)
	plt.imshow(images[index], interpolation='nearest')
	plt.title(f'{df["price"][index]}')
	plt.axis('off')
plt.show()


# Normalize pixels to [0, 1]
images = images / 255.0

# Same split and seed as training, so the test set is identical
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# Normalize prices by the training max, as in training
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# Evaluation on the test split
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

# Absolute percentage error per house
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# Report results
locale.setlocale(locale.LC_ALL, "tr_TR.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))