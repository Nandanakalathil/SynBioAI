import torch
import pandas as pd
import os
import argparse


def extract_proposals(project, round_num, master_csv=None):
    base_path = f"{project}/data"
    results_path = f"{project}/results/round{round_num}"
    
    if not os.path.exists(results_path):
        print(f"Error: Results folder not found at {results_path}")
        return
    
    # Find the indices file
    indices_files = [f for f in os.listdir(results_path) if f.endswith('indices.pt')]
    if not indices_files:
        print(f"Error: No indices.pt file found in {results_path}")
        return
    
    # Load data
    all_combos = pd.read_csv(f"{base_path}/all_combos.csv")
    selected_indices = torch.load(f"{results_path}/{indices_files[0]}")
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.tolist()
    
    # Get the current round's batch (last 96)
    batch_size = 96
    current_batch_indices = selected_indices[-batch_size:]
    
    # Build results
    results = []
    for idx in current_batch_indices:
        if hasattr(idx, 'item'):
            idx = idx.item()
        idx = int(idx)
        combo = all_combos.iloc[idx]['Combo']
        row = {'Index': idx, 'Combo': combo}
        results.append(row)
    
    df_res = pd.DataFrame(results)
    
    # ALWAYS include a Fitness column
    df_res['Fitness'] = None
    
    # Optionally map True Fitness from a master CSV if provided
    if master_csv and os.path.exists(master_csv):
        master_df = pd.read_csv(master_csv)
        # Auto-detect sequence and fitness columns
        seq_col = next((c for c in ['AAs', 'Combo', 'Sequence'] if c in master_df.columns), None)
        fit_col = next((c for c in ['fitness', 'Fitness', 'FITNESS'] if c in master_df.columns), None)
        
        if seq_col and fit_col:
            master_map = dict(zip(master_df[seq_col], master_df[fit_col]))
            df_res['Fitness'] = df_res['Combo'].map(master_map)
            print(f"Mapped fitness values from {master_csv}")
    
    # Save only the Combo column as requested
    output_file = f"{results_path}/round{round_num}_proposals.csv"
    df_res[['Combo']].to_csv(output_file, index=False)
    
    print(f"\n--- Round {round_num} AI Proposals for '{project}' ---")
    print(df_res.head(20))
    print(f"\nTotal proposals: {len(df_res)}")
    print(f"Saved to: {output_file}")
    return df_res

def merge_rounds(project, current_fitness_csv, proposals_csv, output_csv):
    """Merge current training data with new round proposals to create the next round's input."""
    df_current = pd.read_csv(current_fitness_csv)
    df_new = pd.read_csv(proposals_csv)
    
    # Standardize column names
    if 'True_Fitness' in df_new.columns:
        df_new = df_new[['Combo', 'True_Fitness']].rename(columns={'True_Fitness': 'Fitness'})
    
    combined = pd.concat([df_current, df_new], ignore_index=True)
    combined.to_csv(output_csv, index=False)
    print(f"\nMerged {len(df_current)} + {len(df_new)} = {len(combined)} rows")
    print(f"Saved to: {output_csv}")

def clean_initial_samples(project):
    """Clean the initial_samples_to_test.csv to only have the Combo column."""
    base_path = f"{project}/data"
    file_path = f"{base_path}/initial_samples_to_test.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    df = pd.read_csv(file_path)
    if 'Combo' in df.columns:
        df[['Combo']].to_csv(file_path, index=False)
        print(f"Cleaned {file_path} (removed extra columns/Fitness).")
    else:
        print(f"Error: 'Combo' column not found in {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALDE Proposal Extractor & Merger")
    subparsers = parser.add_subparsers(dest="command")

    # Extract proposals
    p_extract = subparsers.add_parser("extract", help="Extract proposals from a round")
    p_extract.add_argument("--name", required=True, help="Project name")
    p_extract.add_argument("--round", type=int, required=True, help="Round number")
    p_extract.add_argument("--master_csv", default=None, help="Path to master CSV with true fitness values")

    # Clean initial samples
    p_clean = subparsers.add_parser("clean-initial", help="Clean initial samples CSV")
    p_clean.add_argument("--name", required=True, help="Project name")

    # Merge rounds
    p_merge = subparsers.add_parser("merge", help="Merge current data with new proposals")
    p_merge.add_argument("--name", required=True, help="Project name")
    p_merge.add_argument("--current", required=True, help="Current fitness CSV (e.g. fitness_initial.csv)")
    p_merge.add_argument("--proposals", required=True, help="Proposals CSV from the round")
    p_merge.add_argument("--output", required=True, help="Output CSV for the next round")

    args = parser.parse_args()

    if args.command == "extract":
        extract_proposals(args.name, args.round, args.master_csv)
    elif args.command == "clean-initial":
        clean_initial_samples(args.name)
    elif args.command == "merge":
        merge_rounds(args.name, args.current, args.proposals, args.output)
    else:
        parser.print_help()
