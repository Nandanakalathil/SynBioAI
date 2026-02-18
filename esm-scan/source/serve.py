"""
ESM-Scan SageMaker Serving - Flask-based inference server.
Exposes /ping and /invocations endpoints on port 8080.
"""

import json
import os
import time
import logging
import traceback
import torch
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model objects
MODEL = None
ALPHABET = None
BATCH_CONVERTER = None
DEVICE = None
MODEL_LOADED = False


def load_model():
    """Load the ESM-1v model."""
    global MODEL, ALPHABET, BATCH_CONVERTER, DEVICE, MODEL_LOADED

    if MODEL_LOADED:
        return

    logger.info("Loading ESM-1v model...")
    start = time.time()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    from esm import pretrained

    model_path = "/opt/ml/model/esm1v_t33_650M_UR90S_1.pt"
    if os.path.exists(model_path):
        logger.info(f"Loading from local: {model_path}")
        MODEL, ALPHABET = pretrained.load_model_and_alphabet(model_path)
    else:
        logger.info("Loading from pretrained hub...")
        MODEL, ALPHABET = pretrained.load_model_and_alphabet("esm1v_t33_650M_UR90S_1")

    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    BATCH_CONVERTER = ALPHABET.get_batch_converter()
    MODEL_LOADED = True

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s on {DEVICE}")


def generate_all_mutations(sequence):
    """Generate all possible single amino acid mutations."""
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

    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()


@app.route("/ping", methods=["GET"])
def ping():
    """Health check - SageMaker calls this to verify the container is alive."""
    # Load model on first ping
    try:
        load_model()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Ping failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    """Main inference endpoint."""
    try:
        load_model()

        start_time = time.time()
        logger.info(f"Inference started at: {time.ctime(start_time)}")

        data = json.loads(request.data.decode("utf-8"))
        sequence = data.get("sequence", "").strip().upper()

        if not sequence:
            return jsonify({"error": "No sequence provided"}), 400

        # Validate
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_aas for aa in sequence):
            invalid = set(aa for aa in sequence if aa not in valid_aas)
            return jsonify({"error": f"Invalid amino acids: {invalid}"}), 400

        if len(sequence) > 1022:
            return jsonify({"error": f"Sequence length {len(sequence)} exceeds max 1022"}), 400

        logger.info(f"Sequence length: {len(sequence)}")

        # Get mutations
        mutations = data.get("mutations", None)
        scoring_strategy = data.get("scoring_strategy", "wt-marginals")
        offset_idx = data.get("offset_idx", 1)
        scan_all = mutations is None

        if scan_all:
            mutations = generate_all_mutations(sequence)
            logger.info(f"Full scan: {len(mutations)} mutations")
        else:
            logger.info(f"Specific mutations: {len(mutations)}")

        # Encode sequence
        batch_data = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = BATCH_CONVERTER(batch_data)
        batch_tokens = batch_tokens.to(DEVICE)

        # Compute token probabilities
        logger.info(f"Computing probabilities ({scoring_strategy})...")
        with torch.no_grad():
            if scoring_strategy == "wt-marginals":
                token_probs = torch.log_softmax(
                    MODEL(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],
                    dim=-1
                )
            elif scoring_strategy == "masked-marginals":
                all_token_probs = []
                for i in range(batch_tokens.size(1)):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = ALPHABET.mask_idx
                    tok_probs = torch.log_softmax(
                        MODEL(batch_tokens_masked)["logits"], dim=-1
                    )
                    all_token_probs.append(tok_probs[:, i])
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            else:
                return jsonify({"error": f"Unsupported strategy: {scoring_strategy}"}), 400

        token_probs = token_probs.cpu()

        # Score mutations
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
                logger.warning(f"Error scoring {mut}: {e}")
                skipped += 1

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Done: {len(results)} scored, {skipped} skipped, {duration:.1f}s")

        response = {
            "results": results,
            "metadata": {
                "sequence_length": len(sequence),
                "total_mutations_scored": len(results),
                "skipped": skipped,
                "scoring_strategy": scoring_strategy,
                "duration_seconds": round(duration, 1),
                "scan_all": scan_all,
                "device": str(DEVICE)
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Inference error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting ESM-Scan inference server on port 8080...")
    # Pre-load the model before starting the server
    load_model()
    app.run(host="0.0.0.0", port=8080, threaded=False)
