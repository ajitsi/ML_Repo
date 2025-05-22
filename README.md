The Python scripts in this directory were generated using the notebook 
<a href="https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb">05. Going Modular Part 2 (script mode).</a>

They breakdown as follows:

- data_setup.py - a file to prepare and download data if needed.
- engine.py - a file containing various training functions.
- model_builder.py - a file to create a PyTorch TinyVGG model.
- train.py - a file to leverage all other files and train a target PyTorch model.
- utils.py - a file dedicated to helpful utility functions.

## 1. PyTorch in the wild
In your travels, you'll see many code repositories for PyTorch-based ML projects have instructions on how to run the PyTorch code in the form of Python scripts.

For example, you might be instructed to run code like the following in a terminal/command line to train a model:
```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```
![image](https://github.com/user-attachments/assets/50eb42fd-a2db-41c2-8da7-9e4ade869345)

Running a PyTorch train.py script on the command line with various hyperparameter settings.

In this case, train.py is the target Python script, it'll likely contain functions to train a PyTorch model.

And --model, --batch_size, --lr and --num_epochs are known as argument flags.

You can set these to whatever values you like and if they're compatible with train.py, they'll work, if not, they'll error.

For example, let's say we wanted to train our TinyVGG model from notebook 04 for 10 epochs with a batch size of 32 and a learning rate of 0.001:
```
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```
This results in having a file called data that contains another directory called pizza_steak_sushi with images of pizza, steak and sushi in standard image classification format.
```
going_modular/
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   │   ├── image01.jpeg
        │   │   └── ...
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/
```

## 2. Use Python's argparse module to be able to send the train.py custom hyperparameter values for training procedures.
 * Add an argument flag for using a different:
    * Training/testing directory
    * Learning rate
    * Batch size
    * Number of epochs to train for 
    * Number of hidden units in the TinyVGG model
      * Keep the default values for each of the above arguments as what they already are (as in notebook 05).
 * For example, you should be able to run something similar to the following line to train a TinyVGG model with a learning rate of 0.003 and a batch size of 64 for 20 epochs: 
   python train.py --learning_rate 0.003 batch_size 64 num_epochs 20.

  * ```
    !python train.py --num_epochs 5 --batch_size 128 --hidden_units 128 --learning_rate 0.0003
    ```
    
  * ```
    !python predict.py --image data/pizza_steak_sushi/test/sushi/175783.jpg
    ```
## 3. Push large files (more than 100 MB) to github
Links - https://git-lfs.com/, https://dev.to/iamtekson/upload-large-file-to-github-37me


**$${\color{Darkorange} Deployed \space into \space Hugging \space face \space with \space Gradio \space Structure \space - }$$** **$${\color{orange}Link}$$** (https://huggingface.co/ajitsi)

### 4. Turning our FoodVision Mini model into a deployable app
```
 foodvision_mini/
    ├── 09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth
    ├── app.py
    ├── examples/
    │   ├── example_1.jpg
    │   ├── example_2.jpg
    │   └── example_3.jpg
    ├── model.py
    └── requirements.txt
```
Where:
* `09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth` is our trained PyTorch model file.
* `app.py` contains our Gradio app (similar to the code that launched the app).
    * **Note:** `app.py` is the default filename used for Hugging Face Spaces, if you deploy your app there, Spaces will by default look for a file called `app.py` to run. This is changeable in settings.
* `examples/` contains example images to use with our Gradio app.
* `model.py` contains the model definition as well as any transforms associated with the model.
* `requirements.txt` contains the dependencies to run our app such as `torch`, `torchvision` and `gradio`.
  
  ![image](https://github.com/user-attachments/assets/e31f5c49-831d-427f-bcc3-636c2d3cda70)
  ![image](https://github.com/user-attachments/assets/ee01b6fa-b2be-4898-a208-23f44e3c0b18)

### 5. Turning our FoodVision Big model into a deployable app

We've got a trained and saved EffNetB2 model on 20% of the Food101 dataset.

And instead of letting our model live in a folder all its life, let's deploy it!

We'll deploy our FoodVision Big model in the same way we deployed our FoodVision Mini model, as a Gradio demo on Hugging Face Spaces.

To begin, let's create a `demos/foodvision_big/` directory to store our FoodVision Big demo files as well as a `demos/foodvision_big/examples` directory to hold an example image to test the demo with.

When we're finished we'll have the following file structure:

```
demos/
  foodvision_big/
    09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth
    app.py
    class_names.txt
    examples/
      example_1.jpg
    model.py
    requirements.txt
```

Where:
* `09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth` is our trained PyTorch model file.
* `app.py` contains our FoodVision Big Gradio app.
* `class_names.txt` contains all of the class names for FoodVision Big.
* `examples/` contains example images to use with our Gradio app.
* `model.py` contains the model definition as well as any transforms associated with the model.
* `requirements.txt` contains the dependencies to run our app such as `torch`, `torchvision` and `gradio`.

  ![image](https://github.com/user-attachments/assets/16f90c7d-08d9-4dbe-942f-95ef0dc11859)

### 6. Autocolorization of gray scale images using deep CNN
This project is a **Streamlit web application** for automatic colorization of grayscale images using a deep learning model. The app allows users to upload a grayscale image and view its colorized version side-by-side, leveraging a PyTorch-based neural network.

---

## 🚀 Features

- Upload grayscale images (`.jpg`, `.jpeg`, `.png`)
- View original and colorized images side-by-side
- Simple, responsive UI built with Streamlit
- Modular code structure for easy extension

---

## 🏗️ Folder Structure

```
image_colorization_app/
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py         # Main Streamlit UI
│   ├── inference.py             # Model loading & inference logic
│   └── model/
│       ├── __init__.py
│       ├── colorization_model.py # Model architecture (PyTorch)
│       └── colorization_model.pth # Trained model weights
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Model Overview

The colorization model is a convolutional neural network (CNN) inspired by the [“Colorful Image Colorization”](https://arxiv.org/abs/1603.08511) paper. It takes a single-channel (L) grayscale image and predicts the two color channels (a, b) in the LAB color space.

### Model Architecture Diagram

```
[Input Grayscale Image (L)]
            |
            v
     [Preprocessing]
            |
            v
   [ColorizationNet (CNN)]
            |
            v
    [Output ab Channels]
            |
            v
[Postprocessing & Merge with L]
            |
            v
   [Colorized RGB Image]
```

---

## 🏋️ Model Training (Summary)

1. **Dataset:**  
   Use a dataset like [ImageNet](http://www.image-net.org/) or [COCO](https://cocodataset.org/) with color images.

2. **Preprocessing:**  
   - Convert images to LAB color space.
   - Use the L channel as input, ab channels as targets.

3. **Training Loop:**  
   - Model: Custom CNN (`ColorizationNet`)
   - Loss: MSE or cross-entropy on ab channels
   - Optimizer: Adam or SGD

4. **Saving Weights:**  
   - Save the trained model as `colorization_model.pth` in `app/model/`.

## 🖥️ Streamlit App Flow

![image](https://github.com/user-attachments/assets/c725de3f-551b-45ed-acae-da8358a2fba9)

---
## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/image_colorization_app.git
   cd image_colorization_app
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Ensure model weights are present:**
   - Place `colorization_model.pth` in `app/model/`.

4. **Run the Streamlit app:**
   ```sh
   streamlit run app/streamlit_app.py
   ```

5. **Open your browser:**  
   Visit [http://localhost:8501](http://localhost:8501)

---

