import spacy
from underthesea import text_normalize, pos_tag
from transformers import pipeline
from typing import Union, List
import torch

nlp = spacy.load("en_core_web_sm")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ckpt = "papluca/xlm-roberta-base-language-detection"
pipe = pipeline("text-classification", model=model_ckpt, device=device)


def compute_df_score(text):
    language = classify_Language(text)

    if language == 'en':
        pd_score = en_calculate_propositional_density(text)
    elif language == 'vi':
        pd_score = vi_calculate_propositional_density(text)
    
    return pd_score


def en_count_propositions(doc):
    """
    Count propositions based on verbs, subordinating conjunctions, and relative pronouns.
    Each verb typically represents a proposition (clause).
    """
    proposition_count = 0
    
    for token in doc:
        # Count main verbs (excluding auxiliaries and modals)
        if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
            proposition_count += 1
        
        # Count subordinating conjunctions (that, because, if, when, etc.)
        elif token.pos_ == "SCONJ":
            proposition_count += 1
        
        # Count relative pronouns (who, which, that as relative)
        elif token.dep_ == "relcl":
            proposition_count += 1
    
    return proposition_count


def en_calculate_propositional_density(text):
    """
    Calculate propositional density: propositions / total words for English
    """
    doc = nlp(text)
    
    # Count words (excluding punctuation)
    word_count = len([token for token in doc if not token.is_punct and not token.is_space])
    
    # Count propositions
    prop_count = en_count_propositions(doc)
    
    # Calculate density
    density = prop_count / word_count if word_count > 0 else 0
    
    return {
        "text": text,
        "word_count": word_count,
        "proposition_count": prop_count,
        "propositional_density": density
    }


def vi_count_propositions(doc: list):
    """
    Count propositions based on verbs, subordinating conjunctions, and relative pronouns.
    Each verb typically represents a proposition (clause).
    """
    proposition_count = 0
    
    for token in doc:
        # Count main verbs 
        if token[1] in ['V', 'A', 'R', 'E', 'C', 'T'] :
            proposition_count += 1
    
    return proposition_count


def vi_calculate_propositional_density(text):
    """
    Calculate propositional density: propositions / total words
    """
    doc = pos_tag(text_normalize(text))
    
    # Count words (excluding punctuation)
    word_count = len([token for token in doc])
    
    # Count propositions
    prop_count = vi_count_propositions(doc)
    
    # Calculate density
    density = prop_count / word_count if word_count > 0 else 0
    
    return {
        "text": text,
        "word_count": word_count,
        "proposition_count": prop_count,
        "propositional_density": density
    }
    

def classify_Language(text: Union[str, List[str]]):

    if isinstance(text, str):
        item = pipe(text, top_k=1, truncation=True)
        return item[0]['label']
    elif isinstance(text, list):
        item = pipe(text, top_k=1, truncation=True)
        labels = [x[0]['label'] for x in item]
        return labels
    else:
        return None