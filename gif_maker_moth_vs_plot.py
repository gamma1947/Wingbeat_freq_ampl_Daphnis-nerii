# ** Disclaimer: This code was entirely written by ChatGpt. I have only made minor modifications to the code. **
# This code creates a GIF from a video and a graph. The video and graph are synchronized to show the video frame and the graph data at the same time. The GIF is saved to a file.

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VideoGraphSyncApp:
    def __init__(self, root, video_path, csv_path, gif_path):
        self.root = root
        self.root.title("Video and Graph Sync")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_paused = True
        self.is_playing = False
        self.frame_number = 0
        self.start_frame = 0
        self.end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        self.csv_path = csv_path
        self.data = self.load_csv()

        self.x_data = []
        self.y_data = []
        
        self.gif_path = gif_path
        self.frames = []

        self.create_widgets()

    def load_csv(self):
        try:
            data = pd.read_csv(self.csv_path)
            return data
        except FileNotFoundError:
            print(f"CSV file not found: {self.csv_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"CSV file is empty: {self.csv_path}")
            return None
        except pd.errors.ParserError:
            print(f"Error parsing CSV file: {self.csv_path}")
            return None

    def create_widgets(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.graph_frame = tk.Frame(self.main_frame)
        self.graph_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.line, = self.plot.plot([], [])  # Create an empty line for dynamic updating
        self.canvas = FigureCanvasTkAgg(self.figure, self.graph_frame)
        self.canvas.get_tk_widget().pack()

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack()

        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_video)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_video)
        self.pause_button.grid(row=0, column=1, padx=5, pady=5)

        self.range_frame = tk.Frame(self.control_frame)
        self.range_frame.grid(row=1, column=0, columnspan=2, pady=5)

        self.start_frame_label = ttk.Label(self.range_frame, text="Start Frame:")
        self.start_frame_label.pack(side=tk.LEFT)
        self.start_frame_entry = ttk.Entry(self.range_frame, width=10)
        self.start_frame_entry.pack(side=tk.LEFT)
        self.start_frame_entry.insert(0, "0")

        self.end_frame_label = ttk.Label(self.range_frame, text="End Frame:")
        self.end_frame_label.pack(side=tk.LEFT)
        self.end_frame_entry = ttk.Entry(self.range_frame, width=10)
        self.end_frame_entry.pack(side=tk.LEFT)
        self.end_frame_entry.insert(0, str(self.end_frame))

        self.range_button = ttk.Button(self.range_frame, text="Set Range", command=self.set_range)
        self.range_button.pack(side=tk.LEFT, padx=5)

    def set_range(self):
        try:
            self.start_frame = int(self.start_frame_entry.get())
            self.end_frame = int(self.end_frame_entry.get())
            if self.start_frame < 0 or self.end_frame > int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1:
                raise ValueError("Frame range out of bounds")
            self.frame_number = self.start_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        except ValueError as e:
            print(f"Invalid frame range: {e}")

    def start_video(self):
        if not self.is_playing:
            self.is_playing = True
            self.is_paused = False
            self.frames = []  # Reset frames for new GIF recording
            self.update_frame()

    def pause_video(self):
        self.is_paused = True
        self.is_playing = False
        self.save_gif()  # Save GIF when paused

    def update_frame(self):
        if not self.is_paused and self.start_frame <= self.frame_number <= self.end_frame:
            ret, frame = self.cap.read()
            if ret:
                self.frame_number += 1
                self.update_video_frame(frame)
                self.update_graph()
                
                # Capture the frame and add to list
                self.capture_frame(frame)
                
                self.root.after(int(1000 / self.frame_rate), self.update_frame)
            else:
                self.is_playing = False

    def update_video_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk # type: ignore
        self.video_label.configure(image=imgtk)

    def update_graph(self):
        if self.data is not None and self.frame_number < len(self.data):
            time = self.data.iloc[self.frame_number]['coords']
            data = self.data.iloc[self.frame_number]['y_r']
            self.x_data.append(time)
            self.y_data.append(data)
            self.line.set_data(self.x_data, self.y_data)
            self.plot.relim()
            self.plot.autoscale_view(True, True, True)
            self.canvas.draw()
        else:
            self.plot.text(0.5, 0.5, "End of video or error loading CSV data", horizontalalignment='center',
                           verticalalignment='center', transform=self.plot.transAxes, fontsize=12, color='red')
            self.is_playing = False

    def capture_frame(self, frame):
        # Capture the video frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        video_pil_image = Image.fromarray(frame)
        
        # Capture the canvas as an image
        self.canvas.draw()
        graph_image = self.canvas.get_renderer().buffer_rgba()
        graph_pil_image = Image.frombytes('RGBA', self.canvas.get_width_height(), graph_image)
        graph_pil_image = graph_pil_image.convert("RGB")  # Convert to RGB

        # Combine video and graph images side by side
        combined_image = Image.new('RGB', (video_pil_image.width + graph_pil_image.width, video_pil_image.height))
        combined_image.paste(video_pil_image, (0, 0))
        combined_image.paste(graph_pil_image, (video_pil_image.width, 0))

        self.frames.append(combined_image)

    def save_gif(self):
        if self.frames:
            self.frames[0].save(self.gif_path, save_all=True, append_images=self.frames[1:], duration=1000 / self.frame_rate, loop=0)
            print(f"GIF saved to {self.gif_path}")


if __name__ == "__main__":
    root = tk.Tk()
    video_file_path = r'C:\analysis\app\assets\full_flight_then_warmp_trial_3(4)DLC_resnet50_WBAJul8shuffle1_100000_labeled.mp4'  # Update with your video file path
    csv_file_path = r'C:\analysis\app\assets\full_flight_then_warmp_trial_3(4)DLC_resnet50_WBAJul8shuffle1_100000.csv'  # Update with your CSV file path
    gif_file_path = r'C:\analysis\app\assets\video_graph_sync_3.gif'  # Path to save the GIF
    app = VideoGraphSyncApp(root, video_file_path, csv_file_path, gif_file_path)
    root.mainloop()


