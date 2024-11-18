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

## 4. Deployed into Hugging face with Gradio Structure
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

  ![image](https://github.com/user-attachments/assets/ee01b6fa-b2be-4898-a208-23f44e3c0b18)


