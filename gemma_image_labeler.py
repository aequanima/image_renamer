# Disable tensorflow logging if present
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
import re
import logging
import base64
import datetime # <-- Fixed: Import datetime
from typing import Optional # <-- Fixed: Import Optional for type hint compatibility
import shutil # <-- Added for shutil.copy2

# --- Dependencies Check ---
try:
    # <-- Fixed: Import specific components for clarity and safety
    from ollama import Client, ResponseError
except ImportError:
    messagebox.showerror("Dependency Error",
                         "The 'ollama' Python library is not installed.\n"
                         "Please install it: pip install ollama")
    exit()
# --- Dependencies Check End ---


# --- Configuration ---
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_PROMPT = ("Describe the most salient object in this image. If it contains a real human, respond "
                 "only with the word 'person'. Otherwise, respond only with the single name of the "
                 "most salient object (e.g., 'cat', 'car', 'tree', 'flower', 'building'). "
                 "Be concise and provide only the single word.")

# Other Settings
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
PERSON_SUBDIR = "person" # Subdirectory name within the *output* directory for person images

# Ollama API parameters
OLLAMA_TIMEOUT = 120 # Increased timeout for potentially slower operations
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_TOKENS = 20
# --- Configuration End ---

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to sanitize filenames
def sanitize_filename(name):
    name = name.strip().lower()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    # Handle empty names after sanitization
    if not name:
        name = "unknown"
    return name[:50]

# Function to convert image file to base64 bytes
def image_to_base64_bytes(file_path: Path) -> Optional[bytes]:
    """Reads an image file and returns its content as bytes."""
    try:
        with open(file_path, "rb") as image_file:
            return image_file.read()
    except Exception as e:
        logging.error(f"Error reading image file {file_path.name}: {e}")
        return None

class ImageRenamerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.title(f"Image Renamer (Ollama) - {formatted_date_time}")

        self.geometry("750x800") # Increased height for output dir

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(6, weight=1) # Adjust log area row

        # --- Configuration Frame ---
        self.config_frame = ctk.CTkFrame(self)
        self.config_frame.grid(row=0, column=0, padx=10, pady=(10,5), sticky="ew")
        self.config_frame.grid_columnconfigure(1, weight=1)

        # Ollama Model Name
        ctk.CTkLabel(self.config_frame, text="Ollama Model Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_name_entry = ctk.CTkEntry(self.config_frame, width=400)
        self.model_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.model_name_entry.insert(0, DEFAULT_OLLAMA_MODEL)
        ctk.CTkLabel(self.config_frame, text="(e.g., gemma3:4b, llava)", text_color="gray").grid(row=1, column=1, padx=5, pady=(0,5), sticky="w")

        # Custom Prompt Override
        ctk.CTkLabel(self.config_frame, text="Custom Prompt (Optional):").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        self.prompt_textbox = ctk.CTkTextbox(self.config_frame, height=80, wrap="word")
        self.prompt_textbox.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.prompt_textbox.insert("0.0", DEFAULT_PROMPT)

        # --- Input Selection Frame ---
        self.selection_frame = ctk.CTkFrame(self)
        self.selection_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.selection_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.selection_frame, text="Select Input Source:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=5, pady=(5,0), sticky="w")

        self.select_file_button = ctk.CTkButton(self.selection_frame, text="Select Single Image", command=self.select_file)
        self.select_file_button.grid(row=1, column=0, padx=5, pady=10, sticky="ew")

        self.select_dir_button = ctk.CTkButton(self.selection_frame, text="Select Directory", command=self.select_directory)
        self.select_dir_button.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        self.selected_path_label = ctk.CTkLabel(self.selection_frame, text="Selected Input: None", wraplength=700, justify="left")
        self.selected_path_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # --- Output Directory Frame --- Added
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.output_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.output_frame, text="Select Output Directory:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, padx=5, pady=(5,0), sticky="w")

        ctk.CTkLabel(self.output_frame, text="Output Path:").grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.output_dir_entry = ctk.CTkEntry(self.output_frame, placeholder_text="Select directory where renamed files will be copied")
        self.output_dir_entry.grid(row=1, column=1, padx=5, pady=10, sticky="ew")
        self.browse_output_button = ctk.CTkButton(self.output_frame, text="Browse...", command=self.browse_output_dir)
        self.browse_output_button.grid(row=1, column=2, padx=5, pady=10)

        # --- Action Frame ---
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew") # Row updated

        self.start_button = ctk.CTkButton(self.action_frame, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=10)

        # --- Progress Bar ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=4, column=0, padx=10, pady=0, sticky="ew") # Row updated
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Progress: 0 / 0")
        self.progress_label.pack(side="left", padx=5)
        self.progressbar = ctk.CTkProgressBar(self.progress_frame)
        self.progressbar.pack(side="left", fill="x", expand=True, padx=5)
        self.progressbar.set(0)
        self.progress_frame.grid_remove() # Hide initially

        # --- Log Area ---
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", wrap="word")
        self.log_textbox.grid(row=6, column=0, padx=10, pady=10, sticky="nsew") # Row updated

        self.selected_images = []
        self.selected_output_dir = None # Store selected output dir Path object
        self.processing_thread = None
        self.ollama_client = Client(timeout=OLLAMA_TIMEOUT)

    def update_log(self, message):
        def _update():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.configure(state="disabled")
            self.log_textbox.see("end")
        self.after(0, _update)
        logging.info(message)

    def update_progress(self, current_value, max_value):
        def _update():
            if max_value > 0:
                progress_float = current_value / max_value
                self.progressbar.set(progress_float)
                self.progress_label.configure(text=f"Progress: {current_value} / {max_value}")
            else:
                self.progressbar.set(0)
                self.progress_label.configure(text="Progress: 0 / 0")
        self.after(0, _update)

    def check_start_button_state(self):
        """Enable start button only if input and output are selected."""
        if self.selected_images and self.selected_output_dir:
            self.start_button.configure(state="normal")
        else:
            self.start_button.configure(state="disabled")

    def select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", SUPPORTED_IMAGE_EXTENSIONS), ("All Files", "*.*")]
        )
        if filepath:
            p = Path(filepath)
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                self.selected_images = [p]
                self.selected_path_label.configure(text=f"Selected Input: {filepath}")
                self.update_log(f"Selected single input file: {filepath}")
                self.update_progress(0, 0)
                self.progress_frame.grid_remove()
                self.check_start_button_state() # Check if ready to start
            else:
                messagebox.showerror("Error", f"Selected file is not a supported image type: {p.suffix}\nSupported: {SUPPORTED_IMAGE_EXTENSIONS}")
                # Don't clear selection label here, maybe they want to try again
                self.selected_images = []
                self.check_start_button_state()

    def select_directory(self):
        dirpath = filedialog.askdirectory(title="Select Directory Containing Images")
        if dirpath:
            self.selected_images = []
            p_dir = Path(dirpath)
            count = 0
            for item in p_dir.iterdir():
                if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    self.selected_images.append(item)
                    count += 1
            if count > 0:
                self.selected_path_label.configure(text=f"Selected Input Directory: {dirpath} ({count} images)")
                self.update_log(f"Selected input directory: {dirpath}, found {count} supported images.")
                self.update_progress(0, 0)
                self.progress_frame.grid_remove()
                self.check_start_button_state() # Check if ready to start
            else:
                messagebox.showwarning("No Images Found", f"No supported image files found in the selected directory.\nSupported: {SUPPORTED_IMAGE_EXTENSIONS}")
                # Clear selection
                self.selected_path_label.configure(text="Selected Input: None")
                self.selected_images = []
                self.check_start_button_state()

    def browse_output_dir(self):
        """Opens directory dialog to select output folder."""
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.selected_output_dir = Path(dirpath)
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(self.selected_output_dir))
            self.update_log(f"Selected output directory: {self.selected_output_dir}")
            self.check_start_button_state() # Check if ready to start

    def set_ui_state(self, processing: bool):
        """Enable/disable UI elements during processing."""
        state = "disabled" if processing else "normal"
        self.select_file_button.configure(state=state)
        self.select_dir_button.configure(state=state)
        self.browse_output_button.configure(state=state) # Disable output browse too
        self.model_name_entry.configure(state=state)
        self.prompt_textbox.configure(state=state)
        self.output_dir_entry.configure(state=state) # Disable output entry

        # Start button logic is now handled by check_start_button_state
        # But we need to update its text and disable it explicitly when processing starts
        if processing:
            self.start_button.configure(state="disabled", text="Processing...")
            self.progress_frame.grid() # Show progress bar
        else:
            # Re-enable based on selections, or disable if nothing selected
            self.check_start_button_state()
            self.start_button.configure(text="Start Processing") # Reset text
            self.progress_frame.grid_remove() # Hide progress bar


    def check_ollama_connection(self):
        try:
            self.ollama_client.list()
            self.update_log("Successfully connected to Ollama service.")
            return True
        except Exception as e:
            self.update_log(f"Error connecting to Ollama: {e}")
            if "connection refused" in str(e).lower():
                errmsg = ("Could not connect to the Ollama service at its default address.\n"
                          "Please ensure Ollama is installed and running.\n\n")
            else:
                errmsg = "An error occurred while trying to communicate with Ollama.\n\n"
            messagebox.showerror("Ollama Connection Error", f"{errmsg}Error details: {e}")
            return False

    def start_processing(self):
        """Validates input/output and starts the image processing thread."""
        ollama_model_name = self.model_name_entry.get().strip()
        custom_prompt = self.prompt_textbox.get("0.0", "end-1c").strip()
        output_dir_path = self.selected_output_dir # Already stored as Path object

        # Validations
        if not ollama_model_name:
            messagebox.showerror("Error", "Please enter the Ollama model name to use.")
            return
        if not self.selected_images:
             messagebox.showerror("Error", "No input image file or directory selected.")
             return
        if not output_dir_path:
             messagebox.showerror("Error", "Please select an output directory.")
             return

        # Ensure output directory exists
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             messagebox.showerror("Error", f"Could not create or access output directory:\n{output_dir_path}\nError: {e}")
             return

        prompt_to_use = custom_prompt if custom_prompt else DEFAULT_PROMPT

        if not self.check_ollama_connection():
            return

        self.update_log(f"Starting processing using Ollama model: {ollama_model_name}")
        self.update_log(f"Output directory: {output_dir_path}")
        self.set_ui_state(processing=True)
        self.update_progress(0, len(self.selected_images))

        # Start thread
        self.processing_thread = threading.Thread(
            target=self.process_images_thread,
            args=(list(self.selected_images), output_dir_path, ollama_model_name, prompt_to_use),
            daemon=True
        )
        self.processing_thread.start()


    def run_llm_inference(self, ollama_model_name: str, image_path: Path, prompt: str) -> Optional[str]:
        """Runs inference using the Ollama API."""
        self.update_log(f"Analysing: {image_path.name} (using {ollama_model_name})")

        image_bytes = image_to_base64_bytes(image_path)
        if not image_bytes:
            self.update_log(f"Skipping {image_path.name}: Failed to read image file.")
            return None

        messages = [{'role': 'user', 'content': prompt, 'images': [image_bytes]}]

        try:
            response = self.ollama_client.chat(
                model=ollama_model_name, messages=messages,
                options={
                    'temperature': OLLAMA_TEMPERATURE,
                    'num_predict': OLLAMA_MAX_TOKENS,
                    'stop': ['<end_of_turn>', '<eos>', '\n']
                }
            )

            if response and 'message' in response and 'content' in response['message']:
                content = response['message']['content']
                cleaned_content = content.strip().split('\n')[0].strip('\'" .,')
                first_word = cleaned_content.split()[0] if cleaned_content else ""
                if not first_word:
                     self.update_log(f"Warning: LLM response for {image_path.name} was empty after cleaning.")
                     return None
                self.update_log(f"-> Detected object: '{first_word.lower()}'")
                return first_word.lower()
            else:
                self.update_log(f"Warning: Unexpected Ollama response structure for {image_path.name}: {response}")
                return None
        except ResponseError as e:
             self.update_log(f"Ollama API error for {image_path.name}: {e.status_code} - {e.error}")
             if "model" in str(e.error).lower() and "not found" in str(e.error).lower():
                  self.after(0, messagebox.showwarning, "Model Not Found", f"The Ollama model '{ollama_model_name}' was not found by the Ollama service.\nMake sure you have pulled or created it.")
             return None
        except Exception as e:
            self.update_log(f"Error during Ollama inference for {image_path.name}: {e}")
            return None


    def process_single_image(self, output_dir: Path, ollama_model_name: str, prompt: str, image_path: Path):
        """Processes one image: analyse, determine target path, copy with new name."""
        if not image_path.exists():
            self.update_log(f"Skipping {image_path.name}: Source file no longer exists.")
            return

        object_name = self.run_llm_inference(ollama_model_name, image_path, prompt)

        if not object_name:
            self.update_log(f"Skipping {image_path.name}: Failed to get object name.")
            return

        sanitized_object_name = sanitize_filename(object_name)
        original_stem = image_path.stem # Keep original name part
        extension = image_path.suffix

        # Determine target directory (within output_dir)
        target_sub_dir = output_dir
        if sanitized_object_name == "person":
            target_sub_dir = output_dir / PERSON_SUBDIR
            target_sub_dir.mkdir(exist_ok=True) # Create 'person' subdir if needed

        # Construct potential new filename parts
        new_filename_base = f"{sanitized_object_name}_{original_stem}"
        new_filename = f"{new_filename_base}{extension}"
        new_path = target_sub_dir / new_filename

        # --- Filename Collision Handling in Target Directory ---
        counter = 1
        while new_path.exists():
            # Generate a new name with a counter
            new_filename = f"{new_filename_base}_{counter}{extension}"
            new_path = target_sub_dir / new_filename
            counter += 1
            if counter > 100: # Safety break
                 self.update_log(f"Error: Too many filename collisions for base '{new_filename_base}' in '{target_sub_dir.name}'. Skipping copy of {image_path.name}.")
                 return
        # --- End Collision Handling ---

        try:
            # Copy the original file to the new path, preserving metadata
            shutil.copy2(str(image_path), str(new_path))

            # Log success based on whether it was moved to 'person' subdir
            if target_sub_dir != output_dir:
                 self.update_log(f"Copied and Renamed: {image_path.name} -> {target_sub_dir.name}/{new_filename}")
            else:
                 self.update_log(f"Copied and Renamed: {image_path.name} -> {new_filename}")

        except Exception as e:
             self.update_log(f"Error copying {image_path.name} to {new_path}: {e}")


    def process_images_thread(self, image_paths: list[Path], output_dir: Path, ollama_model_name: str, prompt: str):
        """Worker thread: iterates images, calls processing function."""
        try:
            total_images = len(image_paths)
            processed_count = 0
            self.update_log(f"Starting batch processing for {total_images} images...")
            self.update_progress(processed_count, total_images)

            for image_path in image_paths:
                # Pass output_dir, model name, and prompt to single image processor
                self.process_single_image(output_dir, ollama_model_name, prompt, image_path)
                processed_count += 1
                self.update_progress(processed_count, total_images) # Update progress

            self.update_log("Batch processing finished.")

        except Exception as e:
            self.update_log(f"Fatal error during processing thread: {e}")
            self.after(0, messagebox.showerror, "Processing Error", f"An unexpected error occurred in the processing thread: {e}")
        finally:
            self.after(0, self.set_ui_state, False) # Re-enable UI


if __name__ == "__main__":
    app = ImageRenamerApp()
    app.mainloop()

