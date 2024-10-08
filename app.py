import subprocess
import os
import torch

if torch.cuda.is_available():
  device="cuda"
  print("Using GPU")
else:
  device="cpu"
  print("Using CPU")


# Clone the repository
subprocess.run(["git", "clone", "https://github.com/facefusion/facefusion", "--single-branch"], check=True)
# chande directory to face fusion to run ui
os.chdir("facefusion")


# installation
if device == "cuda":
    subprocess.run(["python", "install.py", "--onnxruntime", "cuda", "--skip-conda"], check=True)
elif device == "cpu":
    subprocess.run(["python", "install.py", "--onnxruntime", "default", "--skip-conda"], check=True)


# Run the ui
if device == "cuda":
    subprocess.run(["python", "facefusion.py", "run", "--execution-providers", "cuda"], check=True)
elif device == "cpu":
    subprocess.run(["python", "facefusion.py", "run", "--execution-providers", "cpu"], check=True)