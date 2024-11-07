### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names =["pizza", "steak", "sushi"]

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
      num_classes=3
)
  # Load saved weights
  # Construct the full path to the model file
model_path = os.path.join(script_dir,  '09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth')
effnetb2.load_state_dict(
      torch.load(
          f = model_path,
          map_location = torch.device("cpu")
      )
)

# Create vit model
vit, vit_transforms = create_vit_model(
     num_classes=3
)
# Load saved weights
# Construct the full path to the model file
model_path = os.path.join(script_dir,  '09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth')
vit.load_state_dict(
    torch.load(
        f = model_path,
        map_location  = torch.device("cpu")
    )
)

### 3. Predict function ###

# Create predict function
def predict(img, model_choice) -> Tuple[Dict, float]:
  # Start the timer
  start_time = timer()
  
  if model_choice == "EffNetB2":
    model = effnetb2
    transforms = effnetb2_transforms
    prediction_label = "Predictions (EffNetB2 🍃)"
  elif model_choice == "ViT":
    model = vit
    transforms = vit_transforms
    prediction_label = "Predictions (ViT 🌞)" 
    
  else:
    raise gr.Error("Invalid model choice")
   
  # Transform the target image and add a batch dimension
  img = transforms(img).unsqueeze(0)

  # Put model into evaluation mode and turn on inference mode
  model.eval()
  with torch.inference_mode():
    # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
    pred_probs = torch.softmax(model(img), dim=1)

  # Create prediction label and prediction probability dictionary for each prediction class
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate the prediction time
  pred_time = round(timer() - start_time, 5)

  # Return the prediction dictionary and prediction time
  return prediction_label, pred_labels_and_probs, pred_time

def dynamic_predict(img, model_choice) -> Tuple[Dict, float]:
  prediction_label, pred_labels_and_probs, pred_time = predict(img, model_choice)
  #prediction_label = f"Predictions ({model_choice})"
  return gr.update(value=pred_labels_and_probs, label= prediction_label), pred_time

### 4. Gradio app ##

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir(os.path.join(script_dir, "examples"))]

#Custom CSS to style the dropdown
css = """
#model-choice-dropdown {
    background-color: lightpink;
    color: #333;
    font-weight: bold;
}
#model-choice-dropdown option[value="ViT"] {
    background-color: blue;
    color: white;
}
"""

# Create model choice
model_choice = gr.Dropdown(choices=["EffNetB2", "ViT"], label="Choose Model", elem_id="model-choice-dropdown")

# Create the Gradio demo
demo = gr.Interface(fn=dynamic_predict,
                    inputs=[gr.Image(type="pil"), model_choice],
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article,
                    theme="compact",
                    css=css
                    )

# Launch the demo!
demo.launch()
