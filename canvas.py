# 1. Create a Canvas for drawing the digits. :- Tkinter Canvas
# 2. to train a neural network to recognize handwritten digits. :- PyTorch
# 3. to add these two functionalities together. :- FastAPI

import tkinter as tk
from PIL import Image, ImageDraw
from tkinter import messagebox
import matplotlib.pyplot as plt
import io
import requests

SERVER_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = "/predict"  # Change this to your FastAPI server URL

class CanvasApp:
    def __init__(self , root):
        self.root = root
        self.root.title("Digit Drawing Canvas")
        self.canavas_size = 600
        self.image_size = 28
        self.brush_size = 20

        self.canvas = tk.Canvas(self.root, width=self.canavas_size, height=self.canavas_size, bg="black")
        self.canvas.pack()

        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()


        self.clear_button = tk.Button(self.button_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.RIGHT)

        self.canvas.bind("<B1-Motion>", self.draw_digit)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        with io.BytesIO() as output:
            self.image.save(output, format="PNG")
            output.seek(0)

            url = SERVER_URL + PREDICT_ENDPOINT
            image_data = output.getvalue()
            response = requests.post(url, files={"file": ( image_data) })

            if response.status_code == 200:
                prediction = response.json().get("predicted_digit", "Unknown")
                messagebox.showinfo("Prediction", f"Here is Your Prediction: {prediction}")
            else:
                messagebox.showerror("Error", "Failed to get prediction from server.")
        
        # plt.imshow(self.image, cmap='gray')
        # plt.savefig("digit.png")
        # messagebox.showinfo("Prediction", "Here is Your Prediction : -1")
        

    def draw_digit(self, event):
        x = event.x
        y = event.y

        x1,y1 = (x-self.brush_size) , (y-self.brush_size)
        x2, y2 = (x+self.brush_size), (y+self.brush_size)

        self.canvas.create_oval(x1, y1, x2, y2, fill="yellow", outline="yellow")

        scaled_x1 , scaled_y1 = int(x1 * self.image_size / self.canavas_size), int(y1 * self.image_size / self.canavas_size)
        scaled_x2, scaled_y2 =  int(x2 * self.image_size / self.canavas_size), int(y2 * self.image_size / self.canavas_size)
        self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill="Yellow")



root = tk.Tk()
root.tk.call('tk', 'scaling', 4.0) 
app = CanvasApp(root)
root.mainloop()