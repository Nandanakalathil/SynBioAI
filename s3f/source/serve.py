"""
S3F SageMaker Serving - Flask-based async inference server.
Sequence-Structure-Surface Fitness Model for protein mutation scoring.
Exposes /ping and /invocations endpoints on port 8080.
"""

import json
import os
import sys
import time
import logging
import traceback
import pickle
import io
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from flask import Flask, request, jsonify
from Bio.PDB import PDBParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add S3F to path
sys.path.insert(0, "/opt/s3f/S3F")

from torchdrug import core, data, utils, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R

# Import S3F modules (registers classes with torchdrug)
from s3f import dataset as s3f_dataset
from s3f import task as s3f_task
from s3f import model as s3f_model
from s3f import gvp as s3f_gvp

app = Flask(__name__)

# Global model objects
TASK = None
DEVICE = None
MODEL_LOADED = False

# S3F config matching s3f.yaml
S3F_CONFIG = {
    "task": {
        "class": "ResidueTypePrediction",
        "mask_rate": 0.15,
        "dropout": 0.5,
        "plddt_threshold": 70,
        "model": {
            "class": "FusionNetwork",
            "sequence_model": {
                "class": "MyESM",
                "path": "/opt/ml/model/esm-model-weights/",
                "model": "ESM-2-650M",
            },
            "structure_model": {
                "class": "SurfGVP",
                "node_in_dim": [1280, 0],
                "node_h_dim": [256, 16],
                "edge_in_dim": [16, 1],
                "edge_h_dim": [64, 1],
                "surf_in_dim": [42, 0],
                "surf_edge_in_dim": [16, 1],
                "num_surf_res_neighbor": 3,
                "num_surf_graph_neighbor": 16,
                "num_layers": 5,
                "vector_gate": True,
                "readout": "mean",
                "drop_rate": 0.1,
            },
        },
        "graph_construction_model": {
            "class": "GraphConstruction",
            "node_layers": [{"class": "AlphaCarbonNode"}],
            "edge_layers": [{"class": "SpatialEdge", "radius": 10.0, "min_distance": 0}],
            "edge_feature": None,
        },
    }
}


def load_model():
    """Load the S3F model."""
    global TASK, DEVICE, MODEL_LOADED

    if MODEL_LOADED:
        return

    logger.info("Loading S3F model...")
    start = time.time()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # Build task from config
    from easydict import EasyDict
    cfg = EasyDict(S3F_CONFIG)

    TASK = core.Configurable.load_config_dict(cfg.task)
    TASK.preprocess(None, None, None)

    # Load checkpoint
    ckpt_path = "/opt/ml/model/s3f.pth"
    logger.info(f"Loading checkpoint from {ckpt_path}")
    model_dict = torch.load(ckpt_path, map_location="cpu")["model"]
    TASK.load_state_dict(model_dict)

    TASK = TASK.to(DEVICE)
    TASK.eval()

    MODEL_LOADED = True
    elapsed = time.time() - start
    logger.info(f"S3F model loaded in {elapsed:.1f}s on {DEVICE}")


def parse_pdb_to_protein(pdb_content):
    """Parse PDB content string into a torchdrug Protein object."""
    # Write PDB to temp file for BioPython
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb_content)
        pdb_path = f.name

    try:
        protein, seq = s3f_dataset.bio_load_pdb(pdb_path)
        return protein, seq
    finally:
        os.unlink(pdb_path)


def process_surface(pdb_content):
    """Process surface features from PDB content.
    
    Computes surface points, normals, HKS features, curvatures,
    and residue-to-surface correspondence.
    """
    from s3f import surface

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb_content)
        pdb_path = f.name

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        residues = list(structure.get_residues())
        num_residue = len(residues)

        # Get backbone atom positions (N, CA, C)
        atom_positions = []
        for residue in residues:
            res_atoms = []
            for atom_name in ["N", "CA", "C"]:
                if atom_name in residue:
                    res_atoms.append(residue[atom_name].get_vector().get_array())
                else:
                    # Use CA position as fallback
                    if "CA" in residue:
                        res_atoms.append(residue["CA"].get_vector().get_array())
                    else:
                        res_atoms.append(np.zeros(3))
            atom_positions.append(res_atoms)

        atom_position = torch.tensor(np.array(atom_positions), dtype=torch.float32)
        atom_type = torch.tensor([3, 0, 0], dtype=torch.long)[None, :].repeat(num_residue, 1)
        atom_position = atom_position.flatten(0, 1).to(DEVICE)
        atom_type = F.one_hot(atom_type.flatten(0, 1), num_classes=6).to(DEVICE)
        num_atom = len(atom_position)
        batch = torch.zeros((num_atom,), dtype=torch.long).to(DEVICE)

        # Compute surface
        surf_points, surf_normals, _ = surface.atoms_to_points_normals(
            atom_position, batch, atomtypes=atom_type
        )
        num_surf_points = len(surf_points)

        # Compute residue-to-surface correspondence
        res2surf, _ = surface.knn_atoms(atom_position, surf_points, k=20)
        res2surf = res2surf.view(num_residue, 3, -1)

        # Compute curvatures
        batch_surf = torch.zeros((num_surf_points,), dtype=torch.long).to(DEVICE)
        surf_curvatures = surface.compute_curvatures(
            surf_points, surf_normals, batch=batch_surf,
            curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0]
        )

        # Compute eigenvectors for HKS
        surf_points_np = surf_points.cpu().detach().numpy()
        eigs_ratio = 0.01 if num_surf_points > 20000 else 0.06
        surf_eig_vals, surf_eig_vecs, _ = surface.compute_eigens(
            num_surf_points, surf_points_np, min_n_eigs=50, eigs_ratio=eigs_ratio
        )

        # Compute HKS features
        surf_hks = surface.compute_HKS(
            surf_eig_vecs, surf_eig_vals, num_t=32, t_min=0.1, t_max=1000, scale=1000
        )

        surf_data = {
            "surf_points": surf_points_np.astype(np.float32),
            "surf_normals": surf_normals.cpu().detach().numpy().astype(np.float32),
            "surf_hks": surf_hks.astype(np.float32),
            "surf_curvatures": surf_curvatures.cpu().detach().numpy().astype(np.float32),
            "res2surf": res2surf.cpu().detach().numpy(),
        }
        return surf_data

    finally:
        os.unlink(pdb_path)


def load_surface_graph(surf_data):
    """Load surface graph from surface data dict."""
    return s3f_dataset.load_surface(surf_data)


def graph_concat(graphs):
    """Concatenate multiple graphs."""
    if len(graphs) == 1:
        return graphs[0]
    graph = graphs[0].pack(graphs)
    if isinstance(graph, data.Protein):
        _graph = data.Protein(
            edge_list=graph.edge_list, atom_type=graph.atom_type,
            bond_type=graph.bond_type, residue_type=graph.residue_type,
            atom_name=graph.atom_name, atom2residue=graph.atom2residue,
            residue_feature=graph.residue_feature, b_factor=graph.b_factor,
            bond_feature=None, node_position=graph.node_position,
            num_node=graph.num_atom, num_residue=graph.num_residue,
        )
    else:
        _graph = data.Graph(edge_list=graph.edge_list, num_node=graph.num_node, num_relation=1)
        with _graph.node():
            _graph.node_position = graph.node_position
            _graph.node_feature = graph.node_feature
            _graph.normals = graph.normals
    return _graph


def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    """Get optimal window around mutation position."""
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [
            max(0, mutation_position_relative - half_model_window),
            min(seq_len_wo_special, mutation_position_relative + half_model_window),
        ]


def prepare_masked_sequences(sequence, mutations, protein):
    """Prepare masked sequences for S3F inference."""
    mask_id = TASK.model.sequence_model.alphabet.get_idx("<mask>")

    # Parse mutations and sort by position
    parsed = []
    for mut_str in mutations:
        parts = mut_str.split(":")
        sites = tuple(int(p[1:-1]) - 1 for p in parts)
        parsed.append((sites, parts, 0.0))  # target=0 placeholder
    parsed = sorted(parsed)

    sequences = []
    offsets = []
    mutation_list = parsed

    for i, mut in enumerate(parsed):
        if i > 0 and mut[0] == parsed[i - 1][0]:
            continue

        masked_seq = protein.clone()
        _mutation_site = mut[0]
        node_index = torch.tensor(_mutation_site, dtype=torch.long)

        # Truncate long sequences
        if masked_seq.num_residue > 1022:
            seq_len = masked_seq.num_residue
            start, end = get_optimal_window(
                mutation_position_relative=mut[0][0],
                seq_len_wo_special=seq_len,
                model_window=1022,
            )
        else:
            start, end = 0, masked_seq.num_residue

        node_index = node_index - start
        residue_mask = torch.zeros((masked_seq.num_residue,), dtype=torch.bool)
        residue_mask[start:end] = 1
        masked_seq = masked_seq.subresidue(residue_mask)

        with masked_seq.graph():
            masked_seq.start = torch.as_tensor(start)
            masked_seq.end = torch.as_tensor(end)

        offsets.append(start)

        with masked_seq.residue():
            masked_seq.residue_feature[node_index] = 0
            masked_seq.residue_type[node_index] = mask_id

        sequences.append(masked_seq)

    return sequences, mutation_list, offsets


def predict(sequences, wild_type, surf_graph, batch_size=2):
    """Run S3F model inference."""
    _dataset = s3f_dataset.MutantDataset(sequences, wild_type, surf_graph=surf_graph)
    dataloader = data.DataLoader(_dataset, batch_size, shuffle=False, num_workers=0)

    TASK.eval()
    seq_prob = []
    for batch in dataloader:
        batch = utils.cuda(batch, device=DEVICE)
        with torch.no_grad():
            prob, sizes = TASK.inference(batch)
        cum_sizes = sizes.cumsum(dim=0)
        for i in range(len(sizes)):
            seq_prob.append(prob[cum_sizes[i] - sizes[i] : cum_sizes[i]])
    return seq_prob


def score_mutations(seq_prob, mutations, offsets):
    """Score mutations from model probabilities."""
    results = []
    i = 0
    last_sites = None

    for j, item in enumerate(mutations):
        sites, muts, _ = item
        if j > 0 and sites != last_sites:
            i += 1

        node_index = torch.tensor(sites, dtype=torch.long)
        offset = offsets[i]
        node_index = node_index - offset

        mt_target = [data.Protein.residue_symbol2id.get(m[-1], -1) for m in muts]
        wt_target = [data.Protein.residue_symbol2id.get(m[0], -1) for m in muts]

        log_prob = torch.log_softmax(seq_prob[i], dim=-1)
        mt_log_prob = log_prob[node_index, mt_target]
        wt_log_prob = log_prob[node_index, wt_target]
        log_prob_diff = mt_log_prob - wt_log_prob
        score = log_prob_diff.sum(dim=0)

        mut_str = ":".join(muts)
        results.append({"mutant": mut_str, "score": round(score.item(), 6)})
        last_sites = sites

    return results


def generate_all_single_mutations(sequence):
    """Generate all possible single amino acid substitution strings."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mutations = []
    for i, wt_aa in enumerate(sequence):
        if wt_aa not in amino_acids:
            continue
        for mt_aa in amino_acids:
            if mt_aa != wt_aa:
                mutations.append(f"{wt_aa}{i+1}{mt_aa}")
    return mutations


@app.route("/ping", methods=["GET"])
def ping():
    """Health check."""
    try:
        load_model()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Ping failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    """Main inference endpoint.
    
    Expected JSON payload:
    {
        "sequence": "MKTL...",           # Required: protein sequence
        "pdb": "ATOM ...",               # Required: PDB file content as string
        "mutations": ["A1G", "L5V"],     # Optional: specific mutations to score
        "pdb_range": "1-100"             # Optional: PDB range (default: full)
    }
    """
    try:
        load_model()

        start_time = time.time()
        logger.info(f"S3F inference started at: {time.ctime(start_time)}")

        payload = json.loads(request.data.decode("utf-8"))

        sequence = payload.get("sequence", "").strip().upper()
        pdb_content = payload.get("pdb", "")
        mutations = payload.get("mutations", None)
        pdb_range_str = payload.get("pdb_range", None)

        if not sequence:
            return jsonify({"error": "No sequence provided"}), 400
        if not pdb_content:
            return jsonify({"error": "No PDB structure provided"}), 400

        logger.info(f"Sequence length: {len(sequence)}")

        # Parse wild-type from sequence
        protein = data.Protein.from_sequence(sequence, atom_feature=None, bond_feature=None)
        protein.view = "residue"

        # Parse PDB structure
        logger.info("Parsing PDB structure...")
        wild_type, pdb_seq = parse_pdb_to_protein(pdb_content)
        ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
        wild_type = wild_type.subgraph(ca_index)
        wild_type.view = "residue"

        # Set PDB range
        if pdb_range_str:
            parts = pdb_range_str.split("-")
            start_range, end_range = int(parts[0]) - 1, int(parts[-1])
        else:
            start_range, end_range = 0, wild_type.num_residue

        with wild_type.graph():
            wild_type.start = torch.as_tensor(start_range)
            wild_type.end = torch.as_tensor(end_range)

        # Process surface
        logger.info("Computing surface features...")
        surf_data = process_surface(pdb_content)
        surf_graph = load_surface_graph(surf_data)
        res2surf = torch.as_tensor(surf_data["res2surf"])
        with wild_type.residue():
            wild_type.res2surf = res2surf

        # Get mutations
        scan_all = mutations is None
        if scan_all:
            mutations = generate_all_single_mutations(sequence)
            logger.info(f"Full scan mode: {len(mutations)} possible single mutations")
        else:
            logger.info(f"Scoring {len(mutations)} user-provided mutations")

        # Prepare masked sequences
        logger.info("Preparing masked sequences...")
        masked_sequences, mutation_list, offsets = prepare_masked_sequences(
            sequence, mutations, protein
        )
        logger.info(f"Number of unique mutation sites: {len(masked_sequences)}")

        # Run inference
        logger.info("Running S3F inference...")
        seq_prob = predict(masked_sequences, wild_type, surf_graph, batch_size=2)

        # Score mutations
        logger.info("Scoring mutations...")
        results = score_mutations(seq_prob, mutation_list, offsets)

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Done: {len(results)} scored, {duration:.1f}s")

        response = {
            "results": results,
            "metadata": {
                "sequence_length": len(sequence),
                "total_mutations_scored": len(results),
                "scan_all": scan_all,
                "duration_seconds": round(duration, 1),
                "device": str(DEVICE),
            },
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Inference error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting S3F inference server on port 8080...")
    load_model()
    app.run(host="0.0.0.0", port=8080, threaded=False)
