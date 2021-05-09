import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import datasets
import numpy as np
import argparse
import locale
import os



# CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to model")
args = vars(ap.parse_args())

model = keras.models.load_model(args["model"])

# Price labels from HousesInfo.txt
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# Build the 2x2 room mosaics (same size as used in training)
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])
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