import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
import torch
from .propositional_score import compute_df_score
# Download once
nltk.download('punkt', quiet=True)

# Get device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load once (global)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)


def compute_features(passage: str):
    """
    Input: raw passage with |T1|, |T2|, |S|
    Output: len, prop, coh (exact values used in training)
    """
    # Extract clean text & sentences
    # Remove special tokens but keep structure
    clean_text = re.sub(r'\|[TS]\d*\|', ' ', passage)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Split into sentences (respect |S|)
    sentences = [s.strip() for s in passage.split('|S|') if s.strip()]
    sentences = [re.sub(r'\|[TS]\d*\|', '', s).strip() for s in sentences]

    # Text Length bucket
    word_count = len(clean_text.split())
    length = "short" if word_count < 60 else "medium" if word_count < 140 else "long"

    # Propositional Density (real heuristic)
    # Count verbs + key conjunctions as proxy for propositions

    prop_density = compute_df_score(clean_text)

    # Cohesion (real sentence embeddings)
    if len(sentences) > 1:
        embs = model.encode(sentences, convert_to_tensor=True)
        cosines = util.cos_sim(embs[:-1], embs[1:]).diagonal().cpu().numpy()
        cohesion = round(float(cosines.mean()), 2)
    else:
        cohesion = 0.80  # default for single sentence

    return length, prop_density, cohesion
