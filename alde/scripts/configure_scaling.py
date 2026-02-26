import boto3

region = "us-east-1"
endpoint_name = "alde-async-endpoint-v2"

print(f"Configuring auto-scaling for {endpoint_name} to scale down to 0 after 15 minutes of inactivity...")
asg_client = boto3.client('application-autoscaling', region_name=region)

resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

try:
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
except Exception as e:
    print(f"Error: {e}")
