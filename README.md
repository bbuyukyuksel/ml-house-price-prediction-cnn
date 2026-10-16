# House Price Prediction from Images (CNN Regression)

Predicting house prices **from interior photos only**, using a convolutional neural network trained as a regressor.
Built as a deep learning course project (MasterEdu) on a custom dataset of real estate listings from Istanbul and Ankara.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.3-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.1-5C3EE8?logo=opencv&logoColor=white)

## How It Works

Each house is represented by four photos: **living room, kitchen, bedroom and bathroom**.

1. Every photo is resized to `size × size`.
2. The four photos are tiled into a single `2·size × 2·size` mosaic, so one image describes the whole house.
3. Prices are scaled to `[0, 1]` by dividing by the highest price in the training set.
4. A small CNN (`CONV → ReLU → BN → POOL` ×3 with 16/32/64 filters, then dense layers) outputs a single value.
5. The model is trained with **mean absolute percentage error (MAPE)** as the loss, using Adam.

```
┌──────────┬──────────┐
│  living  │ kitchen  │
│   room   │          │      ┌─────┐
├──────────┼──────────┤ ───▶ │ CNN │ ───▶ price
│ bathroom │ bedroom  │      └─────┘
│          │          │
└──────────┴──────────┘
```

## Dataset

- **891 houses, 3,564 images** collected from real estate listings (Istanbul and Ankara).
- Image order per house: `1` living room, `2` kitchen, `3` bedroom, `4` bathroom.
- The dataset itself is **not included** in this repository.

`SahibindenDatasetConverter/` contains the scripts that turn the raw scraped listings into the training format:

| Script | Purpose |
| --- | --- |
| `dataset_converter_image.py` | Copies listing photos as `<house_id>_<room>.<ext>` |
| `dataset_converter_annotation.py` | Writes prices into `HousesInfo.txt` |

Expected layout:

```
mydataset/
├── HousesInfo.txt        # one line per house: "bedrooms bathrooms area zipcode price"
├── 1_salon.jpg
├── 1_mutfak.jpg
├── 1_yatak.jpg
├── 1_banyo.jpg
└── ...
```

## Project Structure

```
.
├── cnn_regression.py            # training script
├── predict.py                   # evaluate a saved model
├── utils/
│   ├── datasets.py              # attribute loading + 2x2 image mosaic
│   └── models.py                # CNN / MLP definitions
├── SahibindenDatasetConverter/  # raw data → training format
├── model/                       # saved experiments (weights, loss curves, metrics)
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/bbuyukyuksel/ml-house-price-prediction-cnn.git
cd ml-house-price-prediction-cnn
pip install -r requirements.txt
```

### Train

```bash
python cnn_regression.py --dataset ./mydataset --size 64
```

`--size` is the side length of each room photo; the network input is `2·size × 2·size`.
The run is saved to `model/mymodel-<size>/` together with:

- `loss.png`, `val_loss.png` – training curves
- `maxPrice.txt` – the scaling factor used for prices
- `model.txt` – dataset statistics and final MAPE

### Predict

```bash
python predict.py --dataset ./mydataset --model model/14-mymodel-256 --size 256
```

Shows sample images with their prices, then reports the MAPE on the test split.

## Results

75% / 25% train/test split, MAPE on the test set (lower is better).

| Experiment | Photo size | Mean MAPE | Std |
| --- | --- | --- | --- |
| `09-mymodel-64` | 64 | 43.87% | 31.61% |
| `13-mymodel-128` | 128 | **43.87%** | **31.60%** |
| `14-mymodel-256` | 256 | 44.12% | 32.98% |
| `000-norm-lineer-mymodel-64` | 64 | 51.64% | 31.60% |

All experiments, including the failed ones, are kept under [`model/`](model/) with their loss curves.

For reference, the dataset has an average price of **₺943,608** with a standard deviation of **₺2,108,681**,
so the price range is very wide. The error of about 44% shows that interior photos alone carry only part of
the price signal; location and size matter a lot.

## Possible Improvements

- Combine image features with tabular data (area, rooms, district) in a multi-input model
- Use a pretrained backbone (e.g. ResNet, EfficientNet) instead of training from scratch
- Predict `log(price)` to handle the wide price range

