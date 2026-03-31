![Project Demo](https://via.placeholder.com/800x400.png?text=COVID+X-Ray+Detection)
# COVID-19 Chest X-ray Detection (DenseNet121)

This project trains a transfer-learning model for chest X-ray image classification using a pretrained DenseNet121 backbone (ImageNet weights).

The implementation is provided as a notebook:
- `Pretrained_DenseNet121_on_imagenet.ipynb`

## Model Summary

- Backbone: `DenseNet121(include_top=False, weights="imagenet")`
- Input size: `224 x 224 x 3`
- Head:
  - `Flatten`
  - `Dense(1024, relu, L2 regularization)`
  - `Dense(3, sigmoid)`
- Loss: `categorical_crossentropy`
- Optimizer: `Adam(learning_rate=1e-3)`
- Metric: `accuracy`
- Training setup in notebook:
  - Epochs: `20`
  - Steps per epoch: `100`
  - Validation steps: `35`
  - Class weights: `{0:4, 1:1, 2:1}`

## Dataset Layout

The notebook uses `flow_from_directory`, so data should be organized like this:

```text
Original Dataset/
  Train/
    class_1/
    class_2/
    class_3/
  Val/
    class_1/
    class_2/
    class_3/
  Test/
    class_1/
    class_2/
    class_3/
```

In the current notebook, Google Drive paths are hard-coded to:
- `/content/gdrive/MyDrive/Original Dataset/Train`
- `/content/gdrive/MyDrive/Original Dataset/Val`
- `/content/gdrive/MyDrive/Original Dataset/Test`

Update these paths if your data is stored elsewhere.

## Requirements

Install dependencies before running:

```bash
pip install tensorflow keras matplotlib
```

## How to Run

### Option 1: Google Colab (recommended for this notebook)

1. Open the notebook in Colab.
2. Mount Google Drive:
   - `from google.colab import drive`
   - `drive.mount('/content/gdrive')`
3. Ensure dataset folders exist in the expected Drive location.
4. Run all cells in order.

### Option 2: Local Jupyter

1. Remove or comment out the Colab drive-mount cell.
2. Replace Drive dataset paths with local paths.
3. Run the notebook cells in sequence.

## Outputs

The notebook includes:
- Training and validation accuracy plots
- Training and validation loss plots
- Test evaluation (`test_acc`, `test_loss`)

## Notes

- The notebook uses older Keras methods such as `fit_generator` and `evaluate_generator`. In newer TensorFlow/Keras versions, `fit` and `evaluate` are preferred.
- Confirm class order from generators before interpreting metrics.

## License

This repository is distributed under the terms described in the `LICENSE` file.
