The Python scripts in this directory were generated using the notebook 
<a href="https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipyn">05. Going Modular Part 2 (script mode).</a>

They breakdown as follows:

- data_setup.py - a file to prepare and download data if needed.
- engine.py - a file containing various training functions.
- model_builder.py - a file to create a PyTorch TinyVGG model.
- train.py - a file to leverage all other files and train a target PyTorch model.
- utils.py - a file dedicated to helpful utility functions.

# PyTorch in the wild
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