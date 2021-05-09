# Usage: python cnn_regression.py --dataset ./mydataset --size 64

from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from utils import datasets
from utils import models
import numpy as np
import argparse
import locale
import os
import matplotlib.pyplot as plt

# CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
ap.add_argument("-s", "--size", type=int, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())
size = args["size"]

# Price labels from HousesInfo.txt
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# Build the 2x2 room mosaics and normalize pixels to [0, 1]
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"], size=size)
images = images / 255.0

# 75% train / 25% test split (fixed seed for reproducible experiments)
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# Normalize prices by the training max; maxPrice is saved to map predictions back to TL
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# CNN regressor optimized for MAPE, so the error is relative to each house's price
model = models.create_cnn(size*2, size*2, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 10000)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# Training
print("[INFO] training model...")
history = model.fit(x=trainImagesX, y=trainY, 
    validation_data=(testImagesX, testY),
    epochs=2000, batch_size=16)

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

model.save(f"model/mymodel-{size}")

plt.figure(1)
plt.title("Loss")
plt.plot(range(len(history.history["loss"])), history.history["loss"])
plt.savefig(f"model/mymodel-{size}/loss.png")

plt.figure(2)
plt.plot(range(len(history.history["val_loss"])), history.history["val_loss"])
plt.title("Val Loss")
plt.savefig(f"model/mymodel-{size}/val_loss.png")