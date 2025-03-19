import tkinter as tk
from tkinter import *
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw, ImageOps

# Load and preprocess the data
train_data = pd.read_csv('train.csv')
X_train = train_data.drop('label', axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(train_data['label'])

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)

# GUI setup
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Number")
        self.canvas = Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.button_predict = Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()
        self.button_clear = Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill="black", width=10)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        self.image = self.image.resize((28, 28))
        self.image = ImageOps.invert(self.image)
        img_array = np.array(self.image).reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(img_array)
        predicted_number = np.argmax(prediction)
        print(f"Predicted Number: {predicted_number}")

root = Tk()
app = App(root)
root.mainloop()
