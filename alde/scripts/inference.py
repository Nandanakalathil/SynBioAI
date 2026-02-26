import os
import sys
import pandas as pd
import torch
import numpy as np
import random
from io import StringIO
import time
import json

# Add current directory to path so src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.encoding_utils import generate_onehot, generate_all_combos
import src.utils as utils
from src.objectives import Objective
from src.optimize import BayesianOptimization, BO_ARGS

class EndpointProduction(Objective):
    """
    Custom objective class that doesn't rely on files saved to disk, 
    so it can be run dynamically inside a SageMaker Endpoint.
    """
    def __init__(self, df, obj_col):
        train_combos = df['Combo'].tolist()
        self.nsamples = len(train_combos)
        self.ytrain = df[obj_col].values
        
        # Generate onehot for training data
        self.Xtrain = generate_onehot(train_combos)
        self.Xtrain = torch.reshape(self.Xtrain, (self.Xtrain.shape[0], -1))
        
        # Geberate full design space dynamically
        nsites = len(train_combos[0])
        self.all_combos = generate_all_combos(nsites)
        
        # Get indices of training variants within the full design space
        self.train_indices = [self.all_combos.index(combo) for combo in train_combos]

        # Generate onehot for full design space
        X_full = generate_onehot(self.all_combos)
        self.X = torch.reshape(X_full, (len(self.all_combos), -1))

        # filler array, used to measure regret, does not affect outcome
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


# --- SAGEMAKER ENDPOINT HANDLERS ---

def model_fn(model_dir):
    """
    Load the model. In ALDE, models are trained on the fly, 
    so we just return a dummy initialized string.
    """
    return "ALDE_Dynamic_Model"

def input_fn(request_body, request_content_type):
    """
    Parse the incoming request. Expected format is text/csv.
    """
    if request_content_type == 'text/csv':
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Expected text/csv.")

def predict_fn(input_data, model):
    """
    Run the Active Learning Bayesian Optimization to predict the next variants.
    """
    df = input_data
    if 'Combo' not in df.columns:
        raise ValueError("Input CSV must contain a 'Combo' column representing the protein sequence.")
    
    # Identify the objective/fitness column
    obj_col = None
    cols = [c for c in df.columns if c != 'Combo']
    if len(cols) > 0:
        obj_col = cols[0]
    else:
        raise ValueError("Input CSV must contain a fitness score column alongside 'Combo'.")
        
    print(f"Starting prediction. Optimizing fitness column: {obj_col}")
    
    # 1. Setup objective and domain space
    obj = EndpointProduction(df, obj_col)
    
    obj_fn = obj.objective
    domain = obj.get_domain()
    disc_X, disc_y = obj.get_points()

    # 2. Config arguments
    batch_size = 90
    budget = batch_size
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mtype = "DNN_ENSEMBLE"
    acq_fn = "TS" # Thompson Sampling
    kernel = "RBF"
    dropout = 0.0
    lr = 1e-3
    activation = "lrelu"
    
    # Architecture for DNN_ENSEMBLE with onehot
    arc = [domain[0].size(-1), 50, 30, 1]
    
    # Create temporary directory to save model results
    temp_dir = "/tmp/alde_results/"
    os.makedirs(temp_dir, exist_ok=True)
    fname = f"{mtype}-DO-{dropout}-{kernel}-{acq_fn}-{arc[-2:]}_1"
    
    run_args = BO_ARGS(
        mtype=mtype,
        kernel=kernel,
        acq_fn=acq_fn,
        xi=4.0,
        architecture=arc,
        activation=activation,
        min_noise=1e-6,
        trainlr=lr,
        train_iter=300,
        dropout=dropout,
        mcdropout=0,
        verbose=2,
        bb_fn=obj_fn,
        domain=domain,
        disc_X=disc_X,
        disc_y=disc_y,
        noise_std=0,
        n_rand_init=0,
        budget=budget,
        query_cost=1,
        queries_x=obj.Xtrain,
        queries_y=obj.ytrain,
        indices=obj.train_indices,
        savedir=os.path.join(temp_dir, fname),
        batch_size=batch_size
    )

    # 3. Run Bayesian Optimization Training and querying
    print("Running Bayesian Optimization. This may take a few minutes...")
    BayesianOptimization.run(run_args, seed)
    
    # 4. Read the generated recommended indices
    indices_path = os.path.join(temp_dir, fname + 'indices.pt')
    if not os.path.exists(indices_path):
        raise RuntimeError("Failed to generate indices. The Bayesian Optimization loop did not output the expected tensor.")
        
    recommended_indices = torch.load(indices_path)
    
    # 5. Map indices back to combinations
    # FIX: cast float indices to int
    recommended_combos = [obj.all_combos[int(i)] for i in recommended_indices.tolist()]
    
    # 6. Format output as a dataframe
    output_df = pd.DataFrame({
        'Recommended_Combo': recommended_combos,
        'Round': 'Next_Round'
    })
    
    return output_df

def output_fn(prediction_df, accept):
    """
    Format output. Returns a CSV string.
    """
    if accept == 'text/csv':
        return prediction_df.to_csv(index=False), accept
    else:
        return prediction_df.to_csv(index=False), 'text/csv'
