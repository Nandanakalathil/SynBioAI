import boto3
import sagemaker
import subprocess
import os
import sys
import time
from sagemaker.model import Model
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig

def deploy_async():
    # 1. AWS Context
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole')['Role']['Arn']

    region = boto3.Session().region_name
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    repo_name = "esm2-embedding-async"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
    
    print(f"[*] Deploying to Region: {region}")
    print(f"[*] Account ID: {account_id}")

    # 2. Build and Push to ECR
    print("[*] Logging into ECR...")
    subprocess.run(f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com", shell=True, check=True)
    
    ecr = boto3.client('ecr')
    try:
        ecr.create_repository(repositoryName=repo_name)
        print(f"[+] Created ECR repository: {repo_name}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass

    print(f"[*] Building Docker Image: {image_uri}")
    # Run from the ESM directory
    esm_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        f"docker build --platform linux/amd64 --provenance=false -t {repo_name} -f {esm_dir}/Dockerfile.esm {esm_dir}",
        shell=True, check=True
    )
    subprocess.run(f"docker tag {repo_name}:latest {image_uri}", shell=True, check=True)

    print(f"[*] Pushing to ECR...")
    subprocess.run(f"docker push {image_uri}", shell=True, check=True)

    # 3. SageMaker Infrastructure
    sm_client = boto3.client('sagemaker', region_name=region)
    sm_session = sagemaker.Session()
    default_bucket = sm_session.default_bucket()
    
    model_name = "ESM2-650M-Model-Async"
    endpoint_config_name = "ESM2-650M-Config-Async"
    endpoint_name = "ESM2-650M-Endpoint-Async"

    # Cleanup existing resources
    for name, method in [(model_name, sm_client.delete_model), 
                         (endpoint_config_name, sm_client.delete_endpoint_config),
                         (endpoint_name, sm_client.delete_endpoint)]:
        try:
            if method == sm_client.delete_endpoint:
                sm_client.describe_endpoint(EndpointName=name)
                print(f"[*] Deleting endpoint {name}...")
                sm_client.delete_endpoint(EndpointName=name)
            elif method == sm_client.delete_endpoint_config:
                sm_client.describe_endpoint_config(EndpointConfigName=name)
                sm_client.delete_endpoint_config(EndpointConfigName=name)
            else:
                sm_client.describe_model(ModelName=name)
                sm_client.delete_model(ModelName=name)
        except sm_client.exceptions.ClientError:
            pass

    # Wait for endpoint deletion if needed
    try:
        while True:
            sm_client.describe_endpoint(EndpointName=endpoint_name)
            print("    Waiting for endpoint deletion...")
            time.sleep(10)
    except sm_client.exceptions.ClientError:
        pass

    # Define Model
    model = Model(
        image_uri=image_uri,
        role=role,
        sagemaker_session=sm_session,
        name=model_name
    )

    # Async Config
    async_config = AsyncInferenceConfig(
        output_path=f"s3://{default_bucket}/esm2/async-outputs/",
        max_concurrent_invocations_per_instance=4
    )

    print(f"[*] Deploying Async Endpoint: {endpoint_name} (Instance: ml.g5.xlarge)")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge", 
        async_inference_config=async_config,
        endpoint_name=endpoint_name
    )

    # 4. Auto Scaling (Scale-to-Zero)
    print("[*] Configuring Scale-to-Zero Auto Scaling...")
    as_client = boto3.client('application-autoscaling')
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    
    as_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=0,
        MaxCapacity=2
    )
    
    as_client.put_scaling_policy(
        PolicyName='HasBacklogWithoutCapacity-ScalingPolicy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 1.0,
            'CustomizedMetricSpecification': {
                'MetricName': 'ApproximateBacklogSizePerInstance',
                'Namespace': 'AWS/SageMaker',
                'Dimensions': [{'Name': 'EndpointName', 'Value': endpoint_name}],
                'Statistic': 'Average',
            },
            'ScaleInCooldown': 300,
            'ScaleOutCooldown': 60,
        }
    )
    
    print(f"\n[✔] Async Deployment Complete!")
    print(f"Endpoint: {endpoint_name}")
    print(f"S3 Output Path: s3://{default_bucket}/esm2/async-outputs/")

if __name__ == "__main__":
    deploy_async()
