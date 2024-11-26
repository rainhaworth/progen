# This file implements the fine tuning using the custom ProGen model and tokenizer
from Bio import SeqIO
from datasets import load_dataset

file_path = "/Users/buenosnachos/Downloads/dataset.json.gz"
dataset = load_dataset(file_path)
