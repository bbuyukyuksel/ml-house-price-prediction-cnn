from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_house_attributes(inputPath):
	# HousesInfo.txt: space separated, one house per line
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()

	return df

def process_house_attributes(df, train, test):
	# Continuous features are min-max scaled, zipcode is one-hot encoded
	continuous = ["bedrooms", "bathrooms", "area"]

	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])

	# Final feature vector: [one-hot zipcode | scaled continuous features]
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])

	return (trainX, testX)

def load_house_images(df, inputPath):
	images = []

	# Each house has 4 photos named <house_id>_<room>.<ext>
	for i in df.index.values:
		# Sort so the rooms always land in the same mosaic cell
		basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
		housePaths = sorted(list(glob.glob(basePath)))

		inputImages = []
		outputImage = np.zeros((64, 64, 3), dtype="uint8")

		for housePath in housePaths:
			image = cv2.imread(housePath)
			image = cv2.resize(image, (32, 32))
			inputImages.append(image)

		# 2x2 mosaic: [0] top-left, [1] top-right, [2] bottom-right, [3] bottom-left
		outputImage[0:32, 0:32] = inputImages[0]
		outputImage[0:32, 32:64] = inputImages[1]
		outputImage[32:64, 32:64] = inputImages[2]
		outputImage[32:64, 0:32] = inputImages[3]

		images.append(outputImage)

	return np.array(images)