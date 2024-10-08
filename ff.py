import subprocess
import os
import torch

if torch.cuda.is_available():
  device="cuda"
  print("Using GPU")
else:
  device="cpu"
  print("Using CPU")

# Check if facefusion directory exists
if not os.path.exists("facefusion"):
    # Clone the repository
    subprocess.run(["git", "clone", "https://github.com/AahirAbbas/ff", "--single-branch"], check=True)
else:
    print("facefusion directory already exists. Skipping git clone.")

# Change directory to facefusion to run UI
os.chdir("facefusion")

# Installation
# ... (rest of your code)

# Clone the repository
subprocess.run(["git", "clone", "https://github.com/AahirAbbas/ff", "--single-branch"], check=True)
# chande directory to face fusion to run ui
os.chdir("facefusion")


# installation
if device == "cuda":
    subprocess.run(["python", "install.py", "--onnxruntime", "cuda", "--skip-conda"], check=True)
elif device == "cpu":
    subprocess.run(["python", "install.py", "--onnxruntime", "default", "--skip-conda"], check=True)


# Run the ui
if device == "cuda":
    subprocess.run(["python", "ff.py", "run", "--execution-providers", "cuda"], check=True)
elif device == "cpu":
    subprocess.run(["python", "ff.py", "run", "--execution-providers", "cpu"], check=True)


# Launch the interface with share=True
demo.launch(share=True)
