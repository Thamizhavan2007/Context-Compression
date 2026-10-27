import sys

import numpy
import pandas
import sklearn
import networkx
import torch
import transformers
import sentence_transformers
import datasets
import streamlit

print("=" * 50)
print("CONTEXT COMPRESSION PROJECT")
print("=" * 50)

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("NetworkX:", networkx.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)
print("Datasets:", datasets.__version__)
print("Streamlit:", streamlit.__version__)

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")