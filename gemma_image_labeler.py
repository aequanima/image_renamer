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
PERSON_SUBDIR = "person"

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
# <-- Fixed: Use Optional for type hint compatibility
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

        # Get current date and time for title
        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        self.title(f"Image Renamer (Ollama) - {formatted_date_time}")

        self.geometry("750x700") # Increased size for prompt box and progress bar

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        # Adjust row weights if needed, log area takes most space
        self.grid_rowconfigure(5, weight=1)

        # --- Configuration Frame ---
        self.config_frame = ctk.CTkFrame(self)
        self.config_frame.grid(row=0, column=0, padx=10, pady=(10,5), sticky="ew")
        self.config_frame.grid_columnconfigure(1, weight=1)

        # Ollama Model Name
        ctk.CTkLabel(self.config_frame, text="Ollama Model Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_name_entry = ctk.CTkEntry(self.config_frame, width=400)
        self.model_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.model_name_entry.insert(0, DEFAULT_OLLAMA_MODEL)
        ctk.CTkLabel(self.config_frame, text="(e.g., gemma3:4b, llava, llama3.2-vision)", text_color="gray").grid(row=1, column=1, padx=5, pady=(0,5), sticky="w")

        # Custom Prompt Override (Enhancement)
        ctk.CTkLabel(self.config_frame, text="Custom Prompt (Optional):").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        self.prompt_textbox = ctk.CTkTextbox(self.config_frame, height=80, wrap="word")
        self.prompt_textbox.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.prompt_textbox.insert("0.0", DEFAULT_PROMPT)

        # --- Selection Frame ---
        self.selection_frame = ctk.CTkFrame(self)
        self.selection_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.selection_frame.grid_columnconfigure(0, weight=1) # Equal weight for buttons

        self.select_file_button = ctk.CTkButton(self.selection_frame, text="Select Single Image", command=self.select_file)
        self.select_file_button.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.select_dir_button = ctk.CTkButton(self.selection_frame, text="Select Directory", command=self.select_directory)
        self.select_dir_button.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.selected_path_label = ctk.CTkLabel(self.selection_frame, text="Selected: None", wraplength=700, justify="left")
        self.selected_path_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # --- Action Frame ---
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.start_button = ctk.CTkButton(self.action_frame, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=10)

        # --- Progress Bar (Enhancement) ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=3, column=0, padx=10, pady=0, sticky="ew")
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Progress: 0 / 0")
        self.progress_label.pack(side="left", padx=5)
        self.progressbar = ctk.CTkProgressBar(self.progress_frame)
        self.progressbar.pack(side="left", fill="x", expand=True, padx=5)
        self.progressbar.set(0) # Initial value
        self.progress_frame.grid_remove() # Hide initially

        # --- Log Area ---
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", wrap="word")
        self.log_textbox.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")

        self.selected_images = []
        self.processing_thread = None
        # Initialize Ollama client once
        self.ollama_client = Client(timeout=OLLAMA_TIMEOUT)


    def update_log(self, message):
        """Appends a message to the log textbox in a thread-safe way."""
        def _update():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.configure(state="disabled")
            self.log_textbox.see("end")
        # Use after() to schedule the update on the main GUI thread
        self.after(0, _update)
        logging.info(message)

    def update_progress(self, current_value, max_value):
        """Updates the progress bar and label in a thread-safe way."""
        def _update():
            if max_value > 0:
                progress_float = current_value / max_value
                self.progressbar.set(progress_float)
                self.progress_label.configure(text=f"Progress: {current_value} / {max_value}")
            else:
                self.progressbar.set(0)
                self.progress_label.configure(text="Progress: 0 / 0")
        self.after(0, _update)


    def select_file(self):
        # (No changes needed in select_file)
        filepath = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", SUPPORTED_IMAGE_EXTENSIONS), ("All Files", "*.*")]
        )
        if filepath:
            p = Path(filepath)
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                self.selected_images = [p]
                self.selected_path_label.configure(text=f"Selected: {filepath}")
                self.start_button.configure(state="normal")
                self.update_log(f"Selected single file: {filepath}")
                self.update_progress(0, 0) # Reset progress
                self.progress_frame.grid_remove() # Hide progress bar
            else:
                messagebox.showerror("Error", f"Selected file is not a supported image type: {p.suffix}\nSupported: {SUPPORTED_IMAGE_EXTENSIONS}")
                self.selected_path_label.configure(text="Selected: None")
                self.start_button.configure(state="disabled")
                self.selected_images = []


    def select_directory(self):
        # (No changes needed in select_directory)
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
                self.selected_path_label.configure(text=f"Selected Directory: {dirpath} ({count} images)")
                self.start_button.configure(state="normal")
                self.update_log(f"Selected directory: {dirpath}, found {count} supported images.")
                self.update_progress(0, 0) # Reset progress
                self.progress_frame.grid_remove() # Hide progress bar
            else:
                messagebox.showwarning("No Images Found", f"No supported image files found in the selected directory.\nSupported: {SUPPORTED_IMAGE_EXTENSIONS}")
                self.selected_path_label.configure(text="Selected: None")
                self.start_button.configure(state="disabled")
                self.selected_images = []


    def set_ui_state(self, processing: bool):
        """Enable/disable UI elements during processing."""
        state = "disabled" if processing else "normal"
        self.select_file_button.configure(state=state)
        self.select_dir_button.configure(state=state)
        self.model_name_entry.configure(state=state)
        self.prompt_textbox.configure(state=state) # Disable prompt box during processing

        if not processing and self.selected_images:
             self.start_button.configure(state="normal", text="Start Processing")
        else:
            if processing:
                self.start_button.configure(state="disabled", text="Processing...")
                self.progress_frame.grid() # Show progress bar when processing starts
            else:
                self.start_button.configure(state="disabled", text="Start Processing")
                self.progress_frame.grid_remove() # Hide progress bar when done


    def check_ollama_connection(self):
        """Tries to connect to Ollama and list models."""
        try:
            # Use the client instance
            self.ollama_client.list()
            self.update_log("Successfully connected to Ollama service.")
            return True
        except Exception as e:
            self.update_log(f"Error connecting to Ollama: {e}")
            # Check if it's specifically a connection error
            if "connection refused" in str(e).lower():
                errmsg = ("Could not connect to the Ollama service at its default address.\n"
                          "Please ensure Ollama is installed and running.\n\n")
            else:
                errmsg = "An error occurred while trying to communicate with Ollama.\n\n"

            messagebox.showerror("Ollama Connection Error", f"{errmsg}Error details: {e}")
            return False


    def start_processing(self):
        """Validates input and starts the image processing in a separate thread."""
        ollama_model_name = self.model_name_entry.get().strip()
        custom_prompt = self.prompt_textbox.get("0.0", "end-1c").strip() # Get prompt text

        if not ollama_model_name:
            messagebox.showerror("Error", "Please enter the Ollama model name to use.")
            return
        if not self.selected_images:
             messagebox.showerror("Error", "No image file or directory selected.")
             return

        # Use custom prompt if provided, otherwise default
        prompt_to_use = custom_prompt if custom_prompt else DEFAULT_PROMPT

        # Check Ollama connection before starting thread
        if not self.check_ollama_connection():
            return

        self.update_log(f"Starting processing using Ollama model: {ollama_model_name}")
        self.set_ui_state(processing=True)
        self.update_progress(0, len(self.selected_images)) # Initialize progress bar

        # Start processing in a new thread
        self.processing_thread = threading.Thread(
            target=self.process_images_thread,
            # Pass the prompt to use
            args=(list(self.selected_images), ollama_model_name, prompt_to_use),
            daemon=True
        )
        self.processing_thread.start()


    def run_llm_inference(self, ollama_model_name: str, image_path: Path, prompt: str) -> Optional[str]:
        """Runs inference using the Ollama API."""
        self.update_log(f"Processing image: {image_path.name} with model {ollama_model_name}")

        # Get image bytes
        image_bytes = image_to_base64_bytes(image_path)
        if not image_bytes:
            self.update_log(f"Skipping {image_path.name}: Failed to read image file.")
            return None

        # Construct messages payload for ollama.chat
        messages = [
            {
                'role': 'user',
                'content': prompt, # Use the provided prompt
                'images': [image_bytes] # Pass raw bytes directly
                # Note: The ollama library handles encoding the bytes as needed for the API call.
                # If issues arise, the alternative is manually encoding:
                # 'images': [base64.b64encode(image_bytes).decode('utf-8')]
            }
        ]

        try:
            # Use the initialized client's chat method
            response = self.ollama_client.chat(
                model=ollama_model_name,
                messages=messages,
                options={
                    'temperature': OLLAMA_TEMPERATURE,
                    'num_predict': OLLAMA_MAX_TOKENS,
                    'stop': ['<end_of_turn>', '<eos>', '\n'] # Added newline as potential stop
                }
            )

            if response and 'message' in response and 'content' in response['message']:
                content = response['message']['content']
                # Clean up potential extra text or quotes
                cleaned_content = content.strip().split('\n')[0].strip('\'" .,')
                first_word = cleaned_content.split()[0] if cleaned_content else ""
                # Additional check: if the response is empty after cleaning, return None
                if not first_word:
                     self.update_log(f"Warning: LLM response for {image_path.name} was empty after cleaning.")
                     return None
                self.update_log(f"LLM raw output: '{content[:50]}...' -> Parsed: '{first_word.lower()}'")
                return first_word.lower()
            else:
                self.update_log(f"Warning: Unexpected Ollama response structure for {image_path.name}: {response}")
                return None

        # <-- Fixed: Catch specific ResponseError
        except ResponseError as e:
             self.update_log(f"Ollama API error for {image_path.name}: {e.status_code} - {e.error}")
             if "model" in str(e.error).lower() and "not found" in str(e.error).lower():
                  # Schedule messagebox on main thread
                  self.after(0, messagebox.showwarning, "Model Not Found", f"The Ollama model '{ollama_model_name}' was not found by the Ollama service.\nMake sure you have pulled or created it.")
             return None
        except Exception as e:
            self.update_log(f"Error during Ollama inference for {image_path.name}: {e}")
            return None


    def process_single_image(self, ollama_model_name: str, prompt: str, image_path: Path):
        """Processes a single image: gets object name, renames, and moves if 'person'."""
        if not image_path.exists():
            self.update_log(f"Skipping {image_path.name}: File no longer exists.")
            return

        # Pass the prompt down
        object_name = self.run_llm_inference(ollama_model_name, image_path, prompt)

        if not object_name:
            self.update_log(f"Failed to get object name for {image_path.name}. Skipping rename.")
            return

        sanitized_object_name = sanitize_filename(object_name)
        # No need to check for empty sanitized_object_name here, sanitize_filename handles it

        original_stem = image_path.stem
        extension = image_path.suffix
        base_dir = image_path.parent

        # Basic check to avoid simple re-runs adding obj_obj_filename
        # This checks if the *current* filename *already* starts with the *newly detected* tag.
        if "_" in original_stem and original_stem.split('_')[0] == sanitized_object_name:
             potential_original = '_'.join(original_stem.split('_')[1:])
             # Check if there was something meaningful after the assumed tag
             if len(potential_original) > 0 and potential_original != original_stem :
                self.update_log(f"Skipping {image_path.name}: Appears to be already renamed with tag '{sanitized_object_name}'.")
                return

        # Define the base for the new filename *before* adding counters
        new_filename_base = f"{sanitized_object_name}_{original_stem}"
        new_filename = f"{new_filename_base}{extension}"

        target_dir = base_dir
        if sanitized_object_name == "person":
            target_dir = base_dir / PERSON_SUBDIR
            target_dir.mkdir(exist_ok=True) # Create 'person' subdir if needed

        new_path = target_dir / new_filename

        # --- Filename Collision Handling (Using Counters) ---
        # This logic prevents overwriting files by adding _1, _2, etc.
        # It runs *before* the actual rename/move operation.
        counter = 1
        while new_path.exists():
            # Important: Check if we are trying to rename the file to itself
            # (can happen if script is re-run after partial success with counters)
            try:
                if new_path.samefile(image_path):
                    self.update_log(f"Skipping {image_path.name}: Target path is the same file.")
                    return
            except FileNotFoundError:
                pass # If new_path doesn't exist yet, samefile check isn't needed/possible here

            # If it's not the same file, generate a new name with a counter
            new_filename = f"{new_filename_base}_{counter}{extension}"
            new_path = target_dir / new_filename
            counter += 1
            if counter > 100: # Safety break
                 self.update_log(f"Error: Too many filename collisions for {image_path.name} with tag '{sanitized_object_name}' in '{target_dir.name}'. Skipping.")
                 return
        # --- End Collision Handling ---

        try:
            # Perform the move/rename using the determined unique new_path
            # Use os.rename which works across filesystems on some OS if shutil fails
            try:
                os.rename(str(image_path), str(new_path))
            except OSError: # Fallback to shutil.move if os.rename fails (e.g., cross-device)
                import shutil
                shutil.move(str(image_path), str(new_path))

            if target_dir != base_dir:
                self.update_log(f"Moved and Renamed: {image_path.name} -> {target_dir.name}/{new_filename}")
            else:
                self.update_log(f"Renamed: {image_path.name} -> {new_filename}")

        except OSError as e:
            self.update_log(f"Error renaming/moving {image_path.name} to {new_path}: {e}")
        except Exception as e:
             self.update_log(f"Unexpected error processing file {image_path.name}: {e}")


    def process_images_thread(self, image_paths: list[Path], ollama_model_name: str, prompt: str):
        """Worker thread function to process all selected images via Ollama."""
        try:
            total_images = len(image_paths)
            processed_count = 0
            self.update_log(f"Starting Ollama processing for {total_images} images...")
            self.update_progress(processed_count, total_images) # Initial progress update

            for image_path in image_paths:
                # Pass the prompt down to the single image processor
                self.process_single_image(ollama_model_name, prompt, image_path)
                processed_count += 1
                # Update progress after each image
                self.update_progress(processed_count, total_images)

            self.update_log("Processing finished.")

        except Exception as e:
            self.update_log(f"Fatal error during processing thread: {e}")
            self.after(0, messagebox.showerror, "Processing Error", f"An unexpected error occurred in the processing thread: {e}")
        finally:
            # Re-enable UI elements on the main thread
            self.after(0, self.set_ui_state, False)


if __name__ == "__main__":
    app = ImageRenamerApp()
    app.mainloop()