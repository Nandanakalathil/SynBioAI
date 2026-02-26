import os
import sys
import pandas as pd
import torch
import numpy as np
import random
from io import StringIO
import time
import json
from flask import Flask, request, Response

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.encoding_utils import generate_onehot, generate_all_combos
import src.utils as utils
from src.objectives import Objective
from src.optimize import BayesianOptimization, BO_ARGS

app = Flask(__name__)

class EndpointProduction(Objective):
    def __init__(self, df, obj_col):
        train_combos = df['Combo'].tolist()
        self.nsamples = len(train_combos)
        self.ytrain = df[obj_col].values
        self.Xtrain = generate_onehot(train_combos)
        self.Xtrain = torch.reshape(self.Xtrain, (self.Xtrain.shape[0], -1))
        nsites = len(train_combos[0])
        print(f"Generating all combos for {nsites} sites...")
        self.all_combos = generate_all_combos(nsites)
        print(f"Total design space size: {len(self.all_combos)}")
        
        # Optimize index lookup
        combo_to_idx = {combo: i for i, combo in enumerate(self.all_combos)}
        self.train_indices = [combo_to_idx[combo] for combo in train_combos]
        
        print(f"Encoding design space...")
        X_full = generate_onehot(self.all_combos)
        print("Reshaping encodings...")
        self.X = torch.reshape(X_full, (len(self.all_combos), -1))
        self.y = np.zeros(len(self.all_combos))
        self.y[self.train_indices] = self.ytrain
        self.ytrain = torch.tensor(self.ytrain)
        self.y = torch.tensor(self.y)
        self.train_indices = torch.tensor(self.train_indices)

    def objective(self, x: torch.Tensor, noise=0.) -> tuple[torch.Tensor, torch.Tensor]:
        qx, qy = utils.query_discrete(self.X, self.y, x)
        return qx.double(), qy.double()

    def get_max(self) -> torch.Tensor:
        return torch.max(self.y).double()

    def get_domain(self) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = utils.domain_discrete(self.X)
        return lower.double(), upper.double()

    def get_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X.double(), self.y.double()

@app.route('/ping', methods=['GET'])
def ping():
    return Response(response='OK', status=200)

@app.route('/invocations', methods=['POST'])
def invocations():
    print("Invocations start")
    try:
        if request.content_type != 'text/csv':
            print(f"Bad content type: {request.content_type}")
            return Response(response='Unsupported content type', status=415)
        
        print("Reading request data...")
        data = request.data.decode('utf-8')
        if not data:
            print("Empty request data!")
            return Response(response='Empty input data', status=400)
            
        print(f"First 100 chars of data: {data[:100]}")
        df = pd.read_csv(StringIO(data))
        print(f"Dataframe loaded. Shape: {df.shape}")
        
        if 'Combo' not in df.columns:
            print("Missing Combo column")
            return Response(response='Missing Combo column', status=400)
        
        obj_col = None
        cols = [c for c in df.columns if c != 'Combo']
        if len(cols) > 0:
            obj_col = cols[0]
        else:
            print("Missing fitness column")
            return Response(response='Missing fitness column', status=400)
            
        print(f"Optimizing: {obj_col}")
        
        print("Initializing EndpointProduction...")
        obj = EndpointProduction(df, obj_col)
        print("EndpointProduction initialized.")
        
        obj_fn = obj.objective
        domain = obj.get_domain()
        disc_X, disc_y = obj.get_points()

        batch_size = 96
        
        # Determine budget from incoming request (SageMaker passes this in a specific header)
        custom_attrs = request.headers.get('X-Amzn-Sagemaker-Custom-Attributes', '')
        print(f"Custom Attributes received: {custom_attrs}")
        
        rounds = 4 # Default
        if 'X-Opt-Rounds=' in custom_attrs:
            try:
                # Extract value from 'X-Opt-Rounds=N'
                for attr in custom_attrs.split(','):
                    if attr.startswith('X-Opt-Rounds='):
                        rounds = int(attr.split('=')[1])
                        break
            except Exception as e:
                print(f"Error parsing rounds from custom attributes: {e}")
            
        budget = batch_size * rounds
        print(f"Requested {rounds} rounds -> Budget: {budget}")
        
        seed = 42

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        mtype = "DNN_ENSEMBLE"
        acq_fn = "TS"
        kernel = "RBF"
        dropout = 0.0
        lr = 1e-3
        activation = "lrelu"
        
        input_dim = domain[0].size(-1)
        arc = [input_dim, 50, 30, 1]
        print(f"Architecture: {arc}")
        
        temp_dir = "/tmp/alde_results/"
        os.makedirs(temp_dir, exist_ok=True)
        fname = f"{mtype}-DO-{dropout}-{kernel}-{acq_fn}-{arc[-2:]}_1"
        savedir = os.path.join(temp_dir, fname)
        
        print(f"Target savedir: {savedir}")
        print(f"Budget: {budget} ({budget // batch_size} rounds of {batch_size})")
        
        run_args = BO_ARGS(
            mtype=mtype, kernel=kernel, acq_fn=acq_fn, xi=4.0, architecture=arc,
            activation=activation, min_noise=1e-6, trainlr=lr, train_iter=300,
            dropout=dropout, mcdropout=0, verbose=2, bb_fn=obj_fn, domain=domain,
            disc_X=disc_X, disc_y=disc_y, noise_std=0, n_rand_init=0, budget=budget,
            query_cost=1, queries_x=obj.Xtrain, queries_y=obj.ytrain,
            indices=obj.train_indices, savedir=savedir,
            batch_size=batch_size
        )

        print("Starting BayesianOptimization.run...")
        BayesianOptimization.run(run_args, seed)
        print("BayesianOptimization.run finished.")
        
        indices_path = savedir + 'indices.pt'
        print(f"Loading indices from: {indices_path}")
        if not os.path.exists(indices_path):
            print(f"Indices file not found at {indices_path}!")
            return Response(response='Failure in BO loop - no indices generated', status=500)
            
        all_indices = torch.load(indices_path)
        print(f"Total indices saved: {len(all_indices)}, training size: {obj.nsamples}")
        
        # The saved indices contain ALL indices (original training + new recommendations)
        # We only want the NEW recommendations (last `budget` entries)
        new_indices = all_indices[obj.nsamples:]
        num_rounds = len(new_indices) // batch_size
        print(f"New recommended indices: {len(new_indices)} ({num_rounds} rounds)")
        
        recommended_combos = [obj.all_combos[int(i)] for i in new_indices.tolist()]
        
        # Label each recommendation with its round number
        rounds = []
        for r in range(num_rounds):
            rounds.extend([f"Round_{r+1}"] * batch_size)
        # Handle any remaining indices
        remaining = len(recommended_combos) - len(rounds)
        if remaining > 0:
            rounds.extend([f"Round_{num_rounds+1}"] * remaining)
        
        output_df = pd.DataFrame({
            'Recommended_Combo': recommended_combos,
            'Round': rounds
        })
        
        print(f"Returning {len(output_df)} recommendations across {num_rounds} rounds.")
        return Response(response=output_df.to_csv(index=False), status=200, mimetype='text/csv')
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"CRITICAL ERROR in invocations: {str(e)}")
        print(err)
        return Response(response=f"Internal Error: {str(e)}\n{err}", status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
