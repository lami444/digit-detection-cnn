import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image , ImageOps
import io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import uvicorn
import os
import torch.nn.functional as F


app = FastAPI()

class MNISTClassifier(nn.Module):
  def __init__(self):
    super(MNISTClassifier , self).__init__()
    self.conv1 = nn.Conv2d(1, 32 , kernel_size=3)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32, 64 , kernel_size=3)
    self.fc1 = nn.Linear(64*5*5 , 128)
    self.fc2 = nn.Linear(128 , 10)

  def forward(self , x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1 , 64*5*5)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

def load_model(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name + '.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    model_loaded = torch.load(model_path, map_location=torch.device('cpu') , weights_only=False)
    model_loaded.eval()
    return model_loaded

model = load_model("model")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   
   try:
      image = Image.open(io.BytesIO(await file.read())).convert("L").resize((28, 28))
      image_contrast = ImageOps.autocontrast(image)
    #   image_contrast.save("temp_image.png")
      image_array = np.array(image_contrast).astype(np.float32) / 255.0

      plt.imshow(image_array, cmap='gray')
      plt.axis('off')
      plt.savefig("temp_image.png", bbox_inches='tight', pad_inches=0)
      
      image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)
      with torch.no_grad():
          output = model(image_tensor)
          _, predicted = torch.max(output, 1)
          prediction = predicted.item()

      return {"predicted_digit": int(prediction)}
   
   except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)