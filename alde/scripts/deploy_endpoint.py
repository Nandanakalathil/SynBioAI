import sagemaker
from sagemaker.model import Model
import boto3
from sagemaker.async_inference import AsyncInferenceConfig

# Define your AWS parameters
role = "arn:aws:iam::119492371915:role/sagemakerexe"
region = "us-east-1"
account_id = "119492371915"

# Define ECR config
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/alde-sagemaker:latest"
model_name = "alde-active-learning-model"
endpoint_config_name = "alde-async-endpoint-v2"  # Using upscaled config name for consistency
endpoint_name = "alde-async-endpoint-v2"
s3_bucket = "sagemaker-us-east-1-119492371915"
s3_prefix = "alde-async-output"

sagemaker_session = sagemaker.Session()
sm_client = boto3.client('sagemaker', region_name=region)

# Cleanup existing resources if they exist
print("Cleaning up existing model/config if any...")
for name in [model_name, endpoint_config_name, endpoint_name]:
    try:
        sm_client.delete_model(ModelName=name)
        print(f"Deleted existing model: {name}")
    except:
        pass
    
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=name)
        print(f"Deleted existing config: {name}")
    except:
        pass
    
    try:
        sm_client.delete_endpoint(EndpointName=name)
        print(f"Deleted existing endpoint: {name}")
    except:
        pass

print(f"Creating SageMaker Model using image: {image_uri}")

# 1. Create Model
alde_model = Model(
    image_uri=image_uri,
    role=role,
    name=model_name,
    sagemaker_session=sagemaker_session,
    env={
        "SAGEMAKER_PROGRAM": "scripts/inference.py"
    }
)

# 2. Setup Async Configuration
async_config = AsyncInferenceConfig(
    output_path=f"s3://{s3_bucket}/{s3_prefix}/output",
    max_concurrent_invocations_per_instance=1,
)

print(f"Deploying asynchronous endpoint: {endpoint_name}...")
# 3. Deploy
predictor = alde_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.2xlarge", # Using upscaled instance type
    endpoint_name=endpoint_name,
    async_inference_config=async_config
)

print(f"Deployment complete. Endpoint Name: {endpoint_name}")

print("Configuring auto-scaling to scale down to 0 after 15 minutes of inactivity...")
asg_client = boto3.client('application-autoscaling', region_name=region)

resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

response = asg_client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=resource_id,
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=0,
    MaxCapacity=1
)

response = asg_client.put_scaling_policy(
    PolicyName=f"{endpoint_name}-scaling-policy",
    ServiceNamespace='sagemaker',
    ResourceId=resource_id,
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 1.0, 
        'CustomizedMetricSpecification': {
            'MetricName': 'ApproximateBacklogSizePerInstance',
            'Namespace': 'AWS/SageMaker',
            'Dimensions': [
                {'Name': 'EndpointName', 'Value': endpoint_name}
            ],
            'Statistic': 'Average',
        },
        'ScaleInCooldown': 900, # 15 minutes
        'ScaleOutCooldown': 120
    }
)
print("Auto-scaling configured successfully!")
