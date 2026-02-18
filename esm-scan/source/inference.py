"""
ESM-Scan SageMaker Inference Handler
Predicts the impact of mutations on a protein using ESM-1v model.
Uses wt-marginals scoring strategy for efficient batch prediction.
"""

import json
import os
import time
import logging
import torch
import numpy as np
from esm import pretrained

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global model objects (loaded once)
MODEL = None
ALPHABET = None
BATCH_CONVERTER = None
DEVICE = None


def model_fn(model_dir):
    """Load the ESM-1v model from the model directory."""
    global MODEL, ALPHABET, BATCH_CONVERTER, DEVICE

    logger.info("Loading ESM-1v model...")
    start = time.time()

    # Check for GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # Load the pretrained model
    # The model will be downloaded automatically by fair-esm if not present
    model_path = os.path.join(model_dir, "esm1v_t33_650M_UR90S_1.pt")
    if os.path.exists(model_path):
        logger.info(f"Loading model from local file: {model_path}")
        MODEL, ALPHABET = pretrained.load_model_and_alphabet(model_path)
    else:
        logger.info("Loading model from pretrained hub: esm1v_t33_650M_UR90S_1")
        MODEL, ALPHABET = pretrained.load_model_and_alphabet("esm1v_t33_650M_UR90S_1")

    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    BATCH_CONVERTER = ALPHABET.get_batch_converter()

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    return MODEL


def input_fn(request_body, request_content_type):
    """Parse the incoming request payload."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")


def generate_all_mutations(sequence):
    """Generate all possible single amino acid mutations for a sequence."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mutations = []
    for i in range(len(sequence)):
        for aa in amino_acids:
            mutations.append(f"{sequence[i]}{i+1}{aa}")
    return mutations


def label_row(mutation, sequence, token_probs, alphabet, offset_idx=1):
    """Score a single mutation using wt-marginals."""
    wt = mutation[0]
    mt = mutation[-1]
    idx = int(mutation[1:-1]) - offset_idx

    if idx < 0 or idx >= len(sequence):
        return None
    if sequence[idx] != wt:
        return None

    wt_encoded = alphabet.get_idx(wt)
    mt_encoded = alphabet.get_idx(mt)

    # +1 for BOS token
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()


def predict_fn(data, model):
    """Run ESM-Scan prediction on the input data."""
    global ALPHABET, BATCH_CONVERTER, DEVICE

    start_time = time.time()
    logger.info(f"Inference started at: {time.ctime(start_time)}")

    sequence = data.get("sequence", "")
    if not sequence:
        return {"error": "No sequence provided in the input payload."}

    # Validate sequence
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.strip().upper()
    if not all(aa in valid_aas for aa in sequence):
        invalid = [aa for aa in sequence if aa not in valid_aas]
        return {"error": f"Invalid amino acids found in sequence: {set(invalid)}"}

    logger.info(f"Sequence length: {len(sequence)}")

    # Check sequence length limit (ESM-1v max is 1022 tokens)
    if len(sequence) > 1022:
        return {"error": f"Sequence length {len(sequence)} exceeds ESM-1v maximum of 1022 residues."}

    # Get mutations to score
    mutations = data.get("mutations", None)
    scoring_strategy = data.get("scoring_strategy", "wt-marginals")
    offset_idx = data.get("offset_idx", 1)
    scan_all = mutations is None

    if scan_all:
        mutations = generate_all_mutations(sequence)
        logger.info(f"Generated {len(mutations)} possible mutations (full scan)")
    else:
        logger.info(f"Scoring {len(mutations)} user-provided mutations")

    # Encode the sequence
    batch_data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER(batch_data)
    batch_tokens = batch_tokens.to(DEVICE)

    # Compute token probabilities using wt-marginals strategy
    logger.info(f"Computing token probabilities ({scoring_strategy})...")
    with torch.no_grad():
        if scoring_strategy == "wt-marginals":
            token_probs = torch.log_softmax(
                model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],
                dim=-1
            )
        elif scoring_strategy == "masked-marginals":
            all_token_probs = []
            for i in range(batch_tokens.size(1)):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, i] = ALPHABET.mask_idx
                tok_probs = torch.log_softmax(
                    model(batch_tokens_masked)["logits"], dim=-1
                )
                all_token_probs.append(tok_probs[:, i])
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        else:
            return {"error": f"Unsupported scoring strategy: {scoring_strategy}. Use 'wt-marginals' or 'masked-marginals'."}

    token_probs = token_probs.cpu()

    # Score each mutation
    logger.info("Scoring mutations...")
    results = []
    skipped = 0
    for mut in mutations:
        try:
            score = label_row(mut, sequence, token_probs, ALPHABET, offset_idx)
            if score is not None:
                results.append({"mutant": mut, "score": round(score, 6)})
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Error scoring mutation {mut}: {e}")
            skipped += 1

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Inference finished at: {time.ctime(end_time)}")
    logger.info(f"Total duration: {duration:.1f}s")
    logger.info(f"Scored {len(results)} mutations, skipped {skipped}")

    return {
        "results": results,
        "metadata": {
            "sequence_length": len(sequence),
            "total_mutations_scored": len(results),
            "skipped": skipped,
            "scoring_strategy": scoring_strategy,
            "duration_seconds": round(duration, 1),
            "scan_all": scan_all
        }
    }


def output_fn(prediction, accept):
    """Serialize the prediction output."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    return json.dumps(prediction), "application/json"
