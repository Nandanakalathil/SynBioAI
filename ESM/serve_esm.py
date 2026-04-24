import os
import io
import json
import torch
import pandas as pd
from flask import Flask, request, Response, jsonify
from esm import pretrained, FastaBatchedDataset
from gevent.pywsgi import WSGIServer
from utils import parse_fasta, embeddings_to_csv

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_NAME = "esm2_t33_650M_UR50D"
PORT = 8080

# Load model and alphabet globally for caching
print(f"[*] Loading {MODEL_NAME} model and alphabet...")
model, alphabet = pretrained.load_model_and_alphabet(MODEL_NAME)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print("[*] Model moved to GPU")
else:
    print("[!] GPU NOT available, using CPU")

batch_converter = alphabet.get_batch_converter()
repr_layers = [model.num_layers]  # Default to last layer

@app.route('/ping', methods=['GET'])
def ping():
    """SageMaker Health Check."""
    health = model is not None
    status = 200 if health else 500
    return Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invoke():
    """Generate embeddings for protein sequences."""
    try:
        # Check if input is JSON or raw FASTA
        if request.content_type == 'application/json':
            payload = request.get_json()
            fasta_content = payload.get("fasta", "")
        else:
            fasta_content = request.data.decode('utf-8')

        if not fasta_content:
            return jsonify({"error": "Empty input"}), 400

        # Parse FASTA
        sequences = parse_fasta(fasta_content)
        if not sequences:
            return jsonify({"error": "No sequences found in FASTA"}), 400

        labels, strs = zip(*sequences)
        
        # Process in batches
        # We can use fair-esm's internal utilities or just manual batching
        # To keep it simple and efficient, we use the batch_converter
        _, _, toks = batch_converter(sequences)
        
        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)

        with torch.no_grad():
            results = model(toks, repr_layers=repr_layers, return_contacts=False)
            token_representations = results["representations"][model.num_layers]

        # Extract mean representations (excluding BOS/EOS if applicable)
        # ESM2 uses [CLS] sequence [SEP] style padding? 
        # Actually alphabet.get_batch_converter() handles it.
        # Mean pooling over the sequence length (excluding padding)
        
        # Calculate mean representations
        # toks: (N, L)
        # token_representations: (N, L, D)
        embeddings = []
        for i, (_, seq) in enumerate(sequences):
            # The tokens for sequence i are at toks[i, 1 : len(seq) + 1]
            # (ESM adds <cls> at 0 and potentially <eos>)
            emb = token_representations[i, 1 : len(seq) + 1].mean(0)
            embeddings.append(emb)
        
        full_embeddings = torch.stack(embeddings)
        csv_output = embeddings_to_csv(labels, full_embeddings)

        return Response(response=csv_output, status=200, mimetype='text/csv')

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"[*] Starting ESM-2 Inference Server on port {PORT}...")
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()
