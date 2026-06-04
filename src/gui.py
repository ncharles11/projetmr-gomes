# src/gui.py
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import logging
from typing import Optional
from src import config

logger = logging.getLogger(__name__)

class FaceRecognitionGUI:
    """
    Handles the graphical user interface for the facial recognition application
    using Tkinter and ttk for a more modern look.
    """
    def __init__(self, root: tk.Tk):
        """
        Initializes the GUI components within the main Tkinter window.

        Args:
            root: The main Tkinter root window.
        """
        self.root = root
        self.root.title(config.GUI_WINDOW_TITLE)
        # Set minimum window size
        self.root.minsize(800, 500)
        self.root.configure(bg='#e0e0e0') # Slightly darker background

        # --- Configure ttk styles ---
        self.style = ttk.Style(self.root)
        available_themes = self.style.theme_names()
        logger.debug(f"Available ttk themes: {available_themes}")
        # Try themes in preferred order
        for theme in ['clam', 'alt', 'default']:
            try:
                self.style.theme_use(theme)
                logger.info(f"Using ttk theme: {theme}")
                break
            except tk.TclError:
                continue
        else:
            logger.warning("Could not set preferred ttk theme, using system default.")

        # General style configurations
        self.style.configure('TFrame', background='#e0e0e0')
        self.style.configure('TLabel', background='#e0e0e0', font=('Helvetica', 10))
        # Custom style for the main video display label
        self.style.configure('Video.TLabel', background='black') # Black background for video area
        # Custom style for the status label
        self.style.configure('Status.TLabel', background='#e8e8e8', font=('Helvetica', 14, 'bold'), padding=10)

        # --- Main frame ---
        self.main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # Configure grid layout for main_frame
        self.main_frame.columnconfigure(0, weight=3) # Video area takes more space
        self.main_frame.columnconfigure(1, weight=1) # Status area
        self.main_frame.rowconfigure(0, weight=1)

        # --- Video Frame (Left) ---
        self.video_frame = ttk.Frame(self.main_frame, padding="5", relief=tk.SUNKEN, borderwidth=1)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_frame.pack_propagate(False) # Prevent frame from shrinking to label size

        self.video_label = ttk.Label(self.video_frame, style='Video.TLabel', anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind('<Configure>', self._on_resize) # Handle resizing
        self._last_img_size = (0, 0) # Store last image size for resize logic

        # --- Status Frame (Right) ---
        self.status_frame = ttk.Frame(self.main_frame, padding="10")
        self.status_frame.grid(row=0, column=1, sticky="nsew")
        self.status_frame.rowconfigure(0, weight=1) # Make label expand vertically
        self.status_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            self.status_frame,
            text="Status: Initializing...",
            style='Status.TLabel',
            anchor=tk.CENTER,
            justify=tk.CENTER,
            wraplength=200 # Wrap text if status gets too long
        )
        self.status_label.grid(row=0, column=0, sticky="nsew", pady=20)

        # Callback attributes — à lier depuis main.py avant le démarrage de la boucle
        self.on_enroll = None        # callable()
        self.on_motor_manual = None  # callable(direction: str)

        # --- Bouton enrôlement ---
        self.enroll_button = ttk.Button(
            self.status_frame,
            text="Ajouter un conducteur",
            command=self._on_enroll_click,
        )
        self.enroll_button.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 8))

        # --- Pavé directionnel ---
        dpad_frame = ttk.LabelFrame(self.status_frame, text="Réglage moteur", padding="6")
        dpad_frame.grid(row=2, column=0, pady=(0, 10))
        dpad_frame.columnconfigure((0, 1, 2), weight=1)

        _w = {"width": 3}
        ttk.Button(dpad_frame, text="▲", command=lambda: self._on_motor_manual_click("up"),    **_w).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(dpad_frame, text="◄", command=lambda: self._on_motor_manual_click("left"),  **_w).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(dpad_frame, text="►", command=lambda: self._on_motor_manual_click("right"), **_w).grid(row=1, column=2, padx=2, pady=2)
        ttk.Button(dpad_frame, text="▼", command=lambda: self._on_motor_manual_click("down"),  **_w).grid(row=2, column=1, padx=2, pady=2)

        # Callback calibration — à lier depuis main.py
        self.on_calibrate_zero = None  # callable()

        # --- Bouton calibration zéro ---
        self.calibrate_button = ttk.Button(
            self.status_frame,
            text="🎯 Fixer le point Zéro",
            command=self._on_calibrate_zero_click,
        )
        self.calibrate_button.grid(row=3, column=0, sticky="ew", padx=5, pady=(0, 6))

        logger.debug("GUI initialized.")

    def _on_resize(self, event):
        """Handles resizing of the video label area to maintain aspect ratio."""
        # This basic resize just notes the event, more complex logic could go here
        # if needed, e.g., recalculating image display size.
        logger.debug(f"Video label resized to: {event.width}x{event.height}")

    def update_status(self, text: str):
        """Updates the status label text."""
        if not isinstance(text, str):
            logger.warning(f"Invalid status text type: {type(text)}. Converting to string.")
            text = str(text)
        self.status_label.config(text=f"Status:\n{text}") # Add newline for better formatting

    def update_image(self, frame: Optional[np.ndarray]):
        """
        Updates the video label with a new frame.

        Args:
            frame: The OpenCV frame (BGR format) to display, or None to clear.
        """
        if frame is None:
            # Optionally display a "No Signal" image or just clear
            self.video_label.config(image=None)
            self.video_label.image = None # Keep reference
            logger.debug("Cleared video display.")
            return

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            logger.warning("Invalid frame passed to update_image.")
            return

        try:
            # Get the dimensions of the label widget
            widget_width = self.video_label.winfo_width()
            widget_height = self.video_label.winfo_height()

            if widget_width <= 10 or widget_height <= 10: # Widget not yet rendered or very small (macOS fix)
                # Use default fallback size for initial display on Mac
                widget_width = 640
                widget_height = 480

            # Calculate aspect ratios
            img_height, img_width = frame.shape[:2]
            widget_aspect = widget_width / widget_height
            img_aspect = img_width / img_height

            # Resize image to fit widget while maintaining aspect ratio
            if img_aspect > widget_aspect:
                # Image is wider than widget aspect ratio -> fit to width
                new_width = widget_width
                new_height = int(new_width / img_aspect)
            else:
                # Image is taller or same aspect ratio -> fit to height
                new_height = widget_height
                new_width = int(new_height * img_aspect)

            # Prevent resizing to zero dimensions
            if new_width < 1 or new_height < 1:
                logger.debug("Skipping image update due to zero target dimensions.")
                return

            # Resize only if the size has changed significantly
            if abs(new_width - self._last_img_size[0]) > 2 or abs(new_height - self._last_img_size[1]) > 2:
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self._last_img_size = (new_width, new_height)
                logger.debug(f"Resized frame to {new_width}x{new_height}")
            else:
                # Use previously resized dimensions if available and close enough
                resized_frame = cv2.resize(frame, self._last_img_size, interpolation=cv2.INTER_AREA)


            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            # Convert PIL Image to PhotoImage for Tkinter
            imgtk = ImageTk.PhotoImage(image=img_pil)

            # Update the label widget
            self.video_label.config(image=imgtk)
            # IMPORTANT: Keep a reference to the image object to prevent garbage collection!
            self.video_label.image = imgtk

        except Exception as e:
            logger.error(f"Error updating GUI image: {e}", exc_info=True)
            # Clear image on error
            self.video_label.config(image=None)
            self.video_label.image = None

    def _on_enroll_click(self):
        """Déclenché par le bouton 'Ajouter un conducteur'. Lier self.on_enroll dans main.py."""
        if self.on_enroll:
            self.on_enroll()

    def _on_motor_manual_click(self, direction: str):
        """Déclenché par le pavé directionnel. direction ∈ {'up','down','left','right'}. Lier self.on_motor_manual dans main.py."""
        if self.on_motor_manual:
            self.on_motor_manual(direction)

    def _on_calibrate_zero_click(self):
        """Déclenché par le bouton 'Fixer le point Zéro'. Lier self.on_calibrate_zero dans main.py."""
        if self.on_calibrate_zero:
            self.on_calibrate_zero()