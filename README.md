# Face blurring project

This project detects whether an image contains a human, cat or dog.
For humans, the model localizes the face area with a bounding box and blurs it.

![Demo](assets/demo.png)
*Left: original image. Right: predicted face box (red) with blurred face. Ground-truth box shown in green.*

## Quick start to see the demo performance

```bash
# Download pre-trained weights into `weights/` (see instructions below)

# Run the interactive demo
python app.py
```

## Model card

- **Task:** Human / cat / dog classification + face bounding-box regression for humans.
- **Training data:** Custom dataset (~8k human images with face boxes, ~375 cats, ~457 dogs), balanced via weighted sampling.
- **Metrics (validation):**
  - Accuracy: 96.74%
  - Avg regression loss (Huber): 0.00081
- **Intended use:** Demonstration, education, and experimentation. Not intended for production or security-critical applications.
- **Limitations:**
  - Trained on single-face images.
  - Performance may degrade on cases where the face is located of the periphery of the image.


## Dataset
- Custom dataset of images labeled as:
    - '0': human (with face bounding box)
    - '1': cat
    - '2': dog
- Training set: ~8000 human images, ~375 cats, ~457 dogs (balanced via weighted sampling).
- Validation set: Images with the same class distribution
- The dataset itself is not included due to size.

### Data format (Open Knowledge Format)

The annotations follow the Open Knowledge Format (OKF) / frictionless data pattern:

- `data/okf/annotations.csv` – combined train/val annotations.
- `data/okf/schema.json` – JSON Schema describing the columns.
- `data/okf/datapackage.json` – data package descriptor.

Columns:

- `image_id`: image filename (e.g. `000001.jpg`).
- `x_1`, `y_1`, `width`, `height`: face bounding box in pixels (for humans).
- `category`: class label (`0` = human, `1` = cat, `2` = dog).
- `split`: data split (`train` or `val`).

## Installation

```bash
git clone https://github.com/MohamadSamaei/face-blurring.git
cd face-blurring

# Install dependencies for users
pip install -r requirements.txt

# Install dependencies for developers (optional)
pip install -r requirements-dev.txt

# Install the project in editable mode
pip install -e .
```
Then open `http://127.0.0.1:7860` in your browser and upload an image.

## Pre-trained weights
Pre-trained weights are provided for **research and educational use only**.

- Download: [v0.1.0 release](https://github.com/MohamadSamaei/face-blurring-app/releases/tag/v0.1.0)
- These weights are trained using the CelebA dataset and are subject to its non‑commercial license.  
  **Commercial use of these weights is not permitted.**
The provided weights were obtained with the following setting:
- **Regression loss:** Huber
- **Training augmentations:**
    `LongestMaxSize`, `PadIfNeeded`, `HorizontalFlip`, `BBoxSafeRandomCrop`, `Resize`, `Affine`,  
  `OneOf([MotionBlur, MedianBlur, GaussNoise, ImageCompression])`,  
  `OneOf([ColorJitter, RandomBrightnessContrast, HueSaturationValue])`
- **epochs:** 25

** Validation performance:**
- overall accuracy: **96.74%**
- Average classification loss: **0.1276**
- Average regression loss: **0.0008**
- Average total loss: **0.1284**

### Using the pre-trained weights

1. Download the weights from the [link to github release]:
   - `backbone.pth`
   - `classification_head.pth`
   - `regression_head.pth`

2. Put the weights directly in the folder weights/    

3. The code automatically loads these weights from `weights/` when running validation, visualization, or the Gradio demo.

### Training with different settings
If you want to train a new model with different settings:

- Update hyperparameters in `configs/default.yaml`.
- After training, the new weights will be saved to:
  - `weights/backbone.pth`
  - `weights/classification_head.pth`
  - `weights/regression_head.pth`

To use them in `app.py`, ensure the paths in `backbone.load_state_dict`, `classification_head.load_state_dict`, and `regression_head.load_state_dict` point to your weight files.

## Usage 
All commands are run from the project root

## Examples

A minimal demo notebook is provided to quickly test the model on a single image:

```bash
jupyter notebook examples/demo.ipynb
```

The notebook:
- Loads the pre-trained weights from `weights/`.
- Runs inference on a user-specified image.
- Draws the predicted face bounding box (for humans) and shows the blurred result.

To use it: 
1. Ensure pre-trained weights are present in `weights/`.
2. Open `examples/demo.ipynb` in Jupyter or VS Code.
3. Set `IMAGE_PATH` to the path of your test image.
4. Run all cells.


### Training
```bash
python main.py --mode train --config configs/default.yaml
```


- Reads data paths and hyperparameters from 'configs/default.yaml'
- Saves trained weights to: `weights/backbone.pth`, `weights/classification_head.pth`, `weights/regression_head.pth`.

### Validation

```bash
python main.py --mode val --config configs/default.yaml
```

- Loads weights from `weights/`
- Evaluates classification accuracy and regression loss.
- Writes predictions to `predicted_coordinates.csv`.

### Visualization (drawing bounding boxes + face blurring) 

```bash
python main.py --mode visualize --config configs/default.yaml
```

- Reads 'predicted_coordinates.csv'
- Darws:
    - Ground-truth box in green
    - predicted box in red
    - blurred face region inside the predicted box 
- Saves annotated images to 'bb_predictions/'.

### Gradio demo (interactive web UI)

```bash
python app.py
```


- Launches a local web interface (usually at `http://127.0.0.1:7860`).
- Upload an image and get:
    - Predicted class(human/cat/dog)
    - Face bounding box and blur for human images

### Configuration

Hyperparameters, paths, and visualization settings are defined in `configs/default.yaml`:

- Data paths: `image_folder_train`, `image_folder_eval`
- Training settings: `epochs`, `batch_size`, `lr_general`, `optimizer_general`
- Model input size: `width_target`, `height_target`
- Bounding box visualization: `box_color_pred`, `box_color_gt`, `box_width`

Edit the yaml file in order to change:
- Learning rate, batch size, number of epochs
- Loss functions (For regression there are choices among Huber or DIoU)
- Input resolution and other experiment settings

## Model
The architecture consists of: 

- **Backbone**: custom VGG-style CNN (5 convolutional blocks + final strided conv) producing a `[B, 512, 4, 4]` feature map
- **Classification head**: global average pooling + 2 layer MLP -> 3 classes (human/cat/dog).
- **Regression head**: a convolutional block for reducing the dimensionality from 512 to 128 + MLP -> 4 outputs (normalized `x1, y1, width, height`), constrained to `[0, 1]` using sigmoid

Training uses:
- weighted random sampler to address class imbalance
- classification loss: cross-entropy
- regression loss: Huber or DIoU (only applied to samples classified as human)

## Metrics
on validation:
- classification accuracy: **96.74%**
- average regression loss (huber): **0.00081**
- average total loss: **0.128**
The reported metrics are obtained using the pre-trained weights (v0.1.0), which are available for non‑commercial research use only.


## Project structure

├── main.py                 # CLI entry point (train/val/visualize)
├── app.py                  # Gradio frontend
├── configs/
│   └── default.yaml        # Hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── models.py           # Backbone, classification and regression heads
│   ├── dataset_train.py    # Training dataset with Albumentations
│   ├── dataset_val.py      # Validation dataset
│   ├── train.py            # Training loop
│   ├── validate.py         # Validation loop
│   ├── visualize.py        # Bounding box drawing + face blurring
│   └── loss_function_DIoU.py  # DIoU regression loss (if used)
├── weights/                 # Saved weights (not committed)
├── data/                   # Dataset (for OKF)
├── bb_predictions/         # Visualized outputs (not committed)
├── assets/                 # demo image
├── examples/
|   └── demo.ipynb          # example notebook to quickly test the functionality of the notebook
├── requirements.txt
├── pyproject.toml
└── README.md


## Extending the project
- Adapt the project to be suitable for multi-face detection
- Make the model's performance more robust in out-of-distribution images (such as images with the face at more peripherial areas and images with many details in them)

## License
This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
**Note on pre-trained weights:**  
The pre-trained weights are trained using the CelebA dataset and are provided for **non‑commercial, research and educational use only**, in accordance with the CelebA dataset license. Commercial use of the weights is not permitted.




## Acknowledgments
- Human faces with bounding boxes are from the [CelebA dataset]
(https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?select=list_bbox_celeba.csv)
- [Albumentations] (https://albumentations.ai/) is used for interactive demo interface
- [Gradio] (https://gradio.app/) is used for interactive demo interface
- [PyTorch] (https://pytorch.org/) and [torchvision] (https://pytorch.org/vision/stable/index.html)





