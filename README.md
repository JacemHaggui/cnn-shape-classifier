# CNN Shape Classifier

This project is a PyTorch-based Convolutional Neural Network (CNN) for classifying simple geometric shapes (circle, rectangle, square, ellipse, triangle) from grayscale PNG images.

## Project Structure

```
.
├── dataset.py         # Custom PyTorch Dataset for loading shape images
├── labels.py          # Shape label mapping
├── model.py           # CNN model definition
├── train.py           # Model training script
├── test.py            # Model evaluation script
├── predict.py         # Single image prediction script
├── train/             # Training images (PNG)
├── test/              # Test images (PNG)
├── .gitignore         # Git ignore file
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://python-pillow.org/)

Install dependencies with:

```sh
pip install torch torchvision pillow
```

## Dataset

- The `train/` and `test/` directories contain the training images.
- Images should be 64x64 PNGs, named with the shape name and a number (e.g., `circle60.png`).

## Training

Train the model with:

```sh
python train.py
```

This will:
- Load images from `train/`
- Train the CNN for 300 epochs (modify to your liking)
- Save the trained model as `CNN_model.pth`

## Testing

Evaluate the model’s accuracy on a folder of images:

```sh
python test.py
```

This will:
- Load images from `train/` (you can change the folder in `test.py`)
- Print predictions and accuracy statistics

## Predicting a Single Image

Predict the shape in a single image:

```sh
python predict.py
```

Edit the `img_path` variable in [`predict.py`](predict.py) to point to your image.

## Label Mapping

See [`labels.py`](labels.py) for the mapping of shape names to integer labels.

## Model Architecture

See [`model.py`](model.py) for the full CNN definition.