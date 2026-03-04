import boto3
import time
import os
import subprocess
import sys

INSTANCE_ID = "i-016cc83a1f7b6a481"
REGION = "us-east-1"
EC2_USER = "ec2-user"
LOCAL_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519.pub")

ec2 = boto3.client('ec2', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)

BUCKET_NAME = "alde-sagemaker-data"
S3_KEY_PREFIX = "simulation_results/"

def get_instance_info():
    response = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
    instance = response['Reservations'][0]['Instances'][0]
    return instance['State']['Name'], instance.get('PublicIpAddress')

def push_key():
    try:
        subprocess.run([
            "aws", "ec2-instance-connect", "send-ssh-public-key",
            "--instance-id", INSTANCE_ID,
            "--instance-os-user", EC2_USER,
            "--ssh-public-key", "file://" + LOCAL_SSH_KEY,
            "--region", REGION
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def start_instance():
    state, ip = get_instance_info()
    if state == 'running':
        return ip
    
    print(f" Starting EC2 Instance ({INSTANCE_ID})...")
    ec2.start_instances(InstanceIds=[INSTANCE_ID])
    
    while True:
        state, ip = get_instance_info()
        if state == 'running' and ip:
            print(f"Instance is LIVE at {ip}")
            print("Waiting 30s for services...")
            time.sleep(30)
            return ip
        if state == 'stopped':
            ec2.start_instances(InstanceIds=[INSTANCE_ID])
        print(f"   Current state: {state}...")
        time.sleep(10)

def stop_instance():
    ec2.stop_instances(InstanceIds=[INSTANCE_ID])

def scp_robust(src, dest, ip):
    """Retries scp up to 3 times with quoted paths for safety"""
    for i in range(3):
        push_key()
        # Wrap paths in quotes to handle spaces/special chars
        cmd = f'scp -o StrictHostKeyChecking=no -o ConnectTimeout=15 "{src}" "{dest}"'
        if subprocess.run(cmd, shell=True).returncode == 0:
            time.sleep(1) # FS sync
            return True
        print(f"    (SCP failed for {os.path.basename(src)}, retrying...)")
        time.sleep(5)
    return False

def run_remote_op(ip, command, silent=False):
    with open("temp_remote_cmd.sh", "w") as f:
        f.write("#!/bin/bash\n" + command + "\n")
    
    if not scp_robust("temp_remote_cmd.sh", f"{EC2_USER}@{ip}:~/temp_remote_cmd.sh", ip):
        if os.path.exists("temp_remote_cmd.sh"): os.remove("temp_remote_cmd.sh")
        return False
    
    ssh_cmd = f"ssh -o StrictHostKeyChecking=no {EC2_USER}@{ip} 'bash ~/temp_remote_cmd.sh'"
    if silent:
        result = subprocess.run(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = (result.returncode == 0)
    else:
        print("  Executing command on EC2...")
        process = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(f"    [EC2] {line.strip()}")
        process.wait()
        res = (process.returncode == 0)
    
    if os.path.exists("temp_remote_cmd.sh"): os.remove("temp_remote_cmd.sh")
    return res

def upload_data(ip, local_path):
    print(f" Uploading dataset: {local_path}")
    remote_dir = "/home/ec2-user/ALDE/data/custom_run"
    run_remote_op(ip, f"mkdir -p {remote_dir} && rm -f {remote_dir}/fitness.csv", silent=True)
    if scp_robust(local_path, f"{EC2_USER}@{ip}:{remote_dir}/fitness.csv", ip):
        print("Data uploaded successfully.")
        return True
    return False

def upload_to_s3(local_file, s3_filename):
    print(f" Uploading results to S3: s3://{BUCKET_NAME}/{S3_KEY_PREFIX}{s3_filename}")
    try:
        s3.upload_file(local_file, BUCKET_NAME, S3_KEY_PREFIX + s3_filename)
        print(" S3 Upload SUCCESS!")
        return True
    except Exception as e:
        print(f" S3 Upload Failed: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o", default="final_results.csv")
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    print("\n--- ALDE AUTOMATION START ---")
    print(f"LOCAL WORKING DIRECTORY: {os.getcwd()}")
    print(f"INTENDED LOCAL OUTPUT: {os.path.abspath(output_csv)}")
    
    while True:
        state, _ = get_instance_info()
        if state != 'stopping': break
        print("   Waiting for vCPU capacity (instance stopping)...")
        time.sleep(15)

    ip = start_instance()

    protein_type = "custom_run" if input_csv else "GB1"
    
    if protein_type == "custom_run":
        encoding_script = """
import pandas as pd, torch, numpy as np, os
AA_SCALES = {
    'A': [1.28, 0.05, 1.00, 0.31, 6.11], 'C': [1.77, 0.13, 0.21, 0.15, 5.07],
    'D': [1.60, 0.11, 0.44, 0.56, 5.07], 'E': [1.56, 0.15, 0.31, 0.56, 5.07],
    'F': [2.94, 0.29, 0.00, 0.03, 5.07], 'G': [0.00, 0.00, 0.00, 0.00, 6.11],
    'H': [2.99, 0.23, 0.13, 0.19, 5.07], 'I': [4.19, 0.19, 0.00, 0.03, 5.07],
    'K': [1.89, 0.22, 0.24, 0.18, 5.07], 'L': [4.19, 0.19, 0.00, 0.03, 5.07],
    'M': [2.35, 0.22, 0.21, 0.15, 5.07], 'N': [1.60, 0.13, 0.34, 0.56, 5.07],
    'P': [2.67, 0.00, 0.31, 0.19, 5.07], 'Q': [1.56, 0.18, 0.24, 0.56, 5.07],
    'R': [2.34, 0.17, 0.33, 0.18, 5.07], 'S': [1.31, 0.06, 0.31, 0.13, 5.07],
    'T': [3.03, 0.11, 0.31, 0.13, 5.07], 'V': [4.07, 0.15, 0.00, 0.03, 5.07],
    'W': [3.21, 0.41, 0.00, 0.03, 5.07], 'Y': [2.94, 0.36, 0.00, 0.03, 5.07]
}
df = pd.read_csv('data/custom_run/fitness.csv')
combos = df['Combo'].values
encoded = [ [b for a in s for b in AA_SCALES.get(a, [0]*5)] for s in combos ]
X = torch.tensor(encoded, dtype=torch.float64)
torch.save(X, 'data/custom_run/AA_x.pt')
"""
        setup_enc_cmd = "cat > /home/ec2-user/ALDE/pre_encode.py << 'REF_EOF'\n" + encoding_script + "\nREF_EOF\n"
        run_remote_op(ip, setup_enc_cmd, silent=True)
        
    py_lines = [
        "from __future__ import annotations",
        "import argparse, numpy as np, pandas as pd, torch, random, os, time, warnings",
        "from src.optimize import BayesianOptimization, BO_ARGS",
        "import src.objectives as objectives",
        "import src.utils as utils",
        "if __name__ == '__main__':",
        "    warnings.filterwarnings('ignore')",
        "    protein = '" + protein_type + "'",
        "    encoding = 'AA'",
        "    mtype = 'DKL_BOTORCH'",
        "    acq_fn = 'TS'",
        "    batch_size = 96",
        "    budget = 384",
        "    output_path = 'results/best_brain_simulation/'",
        "    seed = 3939496635",
        "    obj = objectives.Combo(protein, encoding)",
        "    obj_fn = obj.objective",
        "    domain = obj.get_domain()",
        "    disc_X, disc_y = obj.get_points()",
        "    torch.manual_seed(seed)",
        "    np.random.seed(seed)",
        "    random.seed(seed)",
        "    subdir = output_path + protein + '/' + encoding + '/'",
        "    os.makedirs(subdir, exist_ok=True)",
        "    start_x, start_y, start_indices = utils.samp_discrete(96, obj, seed)",
        "    sim_args = BO_ARGS(mtype=mtype, kernel='RBF', acq_fn=acq_fn, architecture=[domain[0].size(-1), 12, 8, 1], bb_fn=obj_fn, domain=domain, disc_X=disc_X, disc_y=disc_y, budget=budget, batch_size=96, queries_x=start_x, queries_y=start_y, indices=start_indices, savedir=subdir + 'BEST_BRAIN_RUN', verbose=2)",
        "    BayesianOptimization.run(sim_args, seed)"
    ]
    py_script = "\n".join(py_lines)

    setup_cmd = "cat > /home/ec2-user/ALDE/execute_best_brain_auto.py << 'REF_EOF'\n" + py_script + "\nREF_EOF\n"
    setup_cmd += """
cat > /home/ec2-user/ALDE/run_logic_auto.sh << 'REF_EOF'
#!/bin/bash
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh && conda activate alde
cd /home/ec2-user/ALDE
# Clean old results to avoid downloading ghost files from previous runs
rm -f round_*.csv final_auto_results.csv simulation.log
[ -f pre_encode.py ] && python pre_encode.py
python execute_best_brain_auto.py 2>&1 | tee simulation.log
/home/ec2-user/miniconda3/envs/alde/bin/python -s << 'PY_EOF'
import torch, pandas as pd, os, math
protein = '""" + protein_type + """'
indices_path = f'results/best_brain_simulation/{protein}/AA/BEST_BRAIN_RUNindices.pt'
if not os.path.exists(indices_path): exit(1)

indices = torch.load(indices_path).long()
df = pd.read_csv(f'data/{protein}/fitness.csv')

# 1. Track round assignments (batches of 96)
batch_size = 96
num_rounds = math.ceil(len(indices) / batch_size)
round_assignment = {}
for i, idx in enumerate(indices):
    round_assignment[int(idx)] = (i // batch_size) + 1

# 2. Save round-wise results (batches of 96)
for r in range(1, num_rounds + 1):
    end_idx = r * batch_size
    round_indices = indices[:end_idx]
    round_df = df.iloc[round_indices].copy()
    round_df['Round'] = round_df.index.map(lambda x: round_assignment.get(x, r))
    round_df = round_df.sort_values(by='fitness', ascending=False)
    round_df.to_csv(f'round_{r}_results.csv', index=False)

# 3. Save final complete result with Round column
res_final = df.iloc[indices].copy()
res_final['Round'] = res_final.index.map(lambda x: round_assignment.get(x, 0))
res_final = res_final.sort_values(by='fitness', ascending=False)
res_final.to_csv('final_auto_results.csv', index=False)
PY_EOF
REF_EOF
chmod +x /home/ec2-user/ALDE/run_logic_auto.sh
"""
    run_remote_op(ip, setup_cmd, silent=True)

    if input_csv:
        if not upload_data(ip, input_csv):
            stop_instance()
            return

    if run_remote_op(ip, "cd /home/ec2-user/ALDE && ./run_logic_auto.sh"):
        print("\n Simulation Complete! Downloading results...")
        
        # Dynamically determine which rounds exist (up to a safe limit, e.g., 10)
        potential_rounds = [f"round_{r}_results.csv" for r in range(1, 6)]
        files_to_check = potential_rounds + ["final_auto_results.csv"]
        
        for remote_file in files_to_check:
            if remote_file == "final_auto_results.csv":
                local_file = os.path.abspath(output_csv)
            else:
                prefix = output_csv.replace(".csv", "")
                local_file = os.path.abspath(f"{prefix}_{remote_file}")

            remote_path = f"/home/ec2-user/ALDE/{remote_file}"
            print(f"   Downloading {remote_file}...")
            
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            
            if scp_robust(f"{EC2_USER}@{ip}:{remote_path}", local_file, ip):
                if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                    print(f" SUCCESS! File verified at: {local_file}")
                    upload_to_s3(local_file, os.path.basename(local_file))
                else:
                    print(f" WARNING: File is missing or empty at {local_file}")
            elif remote_file == "final_auto_results.csv":
                print(f" ERROR: Could not find final results on server.")
    else:
        print(" Simulation failed.")

    print("Waiting 5s for disk sync...")
    time.sleep(5)
    stop_instance()
    
    print("\n--- FINAL LOCAL FILE LIST ---")
    current_dir = os.path.dirname(os.path.abspath(output_csv))
    for f in os.listdir(current_dir):
        if f.endswith(".csv"):
            print(f" FOUND: {os.path.join(current_dir, f)}")
    print("--- ALDE AUTOMATION FINISHED ---\n")

if __name__ == "__main__":
    main()
