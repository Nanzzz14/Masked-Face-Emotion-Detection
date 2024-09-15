import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOv8App:
    def __init__(self, root):
        self.root = root
        self.root.title("Masked Face Emotion Detection")

        # Set the window to open maximized
        self.root.state('zoomed')

        # Set the background color for the entire window
        self.root.configure(bg="#34495E")  # Dark blue-grey background

        # Create and place the title label with modern styling
        self.title_label = tk.Label(root, text="Masked Face Emotion Detection", font=("Helvetica", 24, "bold italic"), 
                                    fg="#ECF0F1", bg="#2C3E50", pady=10)
        self.title_label.pack(pady=20)

        # Create and place widgets
        self.choose_button = tk.Button(root, text="Browse Image File", command=self.choose_image, font=("Helvetica", 14, "bold"), 
                                       bg="#1ABC9C", fg="#ECF0F1", activebackground="#16A085", activeforeground="#ECF0F1")
        self.choose_button.pack(pady=20)

        self.image_label = tk.Label(root, bg="#34495E")
        self.image_label.pack(pady=20)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            # Load and display image
            self.display_image(file_path)
            # Perform detection
            self.perform_detection(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((800, 800))
        self.img = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.img)

    def perform_detection(self, file_path):
        # Load the YOLOv8 model
        model = YOLO("best (2).pt")  # Change to 'yolov8' if needed

        # Load image
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(img_rgb)

        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        if all(len(x) != 0 for x in confidences) and all(len(x) != 0 for x in class_ids):
            confidences = [y.item() for x in confidences for y in x]
            class_ids = [y.item() for x in class_ids for y in x]

            # Find the index of the highest confidence
            if confidences:
                max_conf_index = confidences.index(max(confidences))
                print(f"max conf: {max(confidences)}")

        if not confidences[0]:
            maxscore = 0.51
        else:
            maxscore = max(confidences)

        print(confidences)
        print(maxscore)

        if maxscore < 0.30:
            maxscore
        # Adjust confidence threshold and perform detection again
        new_results2 = model(img_rgb, conf=float(maxscore - 0.01))

        # Extract the annotated frame
        annotated_img = new_results2[0].plot()

        # Convert to PIL Image and display
        annotated_pil_image = Image.fromarray(annotated_img)
        annotated_pil_image.thumbnail((320, 320))
        self.annotated_img = ImageTk.PhotoImage(annotated_pil_image)
        self.image_label.config(image=self.annotated_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8App(root)
    root.mainloop()
