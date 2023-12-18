import tkinter as tk
from tkinter import messagebox
from video_face import *
from image_face import *

def choose_image():
    run_image()
    # print("Image")
    messagebox.showinfo("Image Saved", "Your image is saved.")

def choose_video():
    run_video()
    # print("Video")

root = tk.Tk()
root.title("Choose Image or Video")

root.geometry("400x200")

image_button = tk.Button(root, text="Choose Image", command=choose_image)
image_button.pack(side=tk.LEFT, padx=(20, 10), pady=20, expand=True, fill=tk.X)

video_button = tk.Button(root, text="Choose Video", command=choose_video)
video_button.pack(side=tk.LEFT, padx=(10, 20), pady=20, expand=True, fill=tk.X)

root.mainloop()
