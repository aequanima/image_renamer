# image_renamer
gemma_image_file_renamer


This is a python script that uses Google's Gemma 3 model to rename single image files or whole image directories according to the most salient feature of that image (or using your custom instructions). 

Install ollama, install ollama python bindings (pip install ollama), start your ollama server, use the intuitive GUI. 

$env:OLLAMA_USE_GPU = 1
ollama run gemma3:4b
python ./gemma_image_labeler.py
