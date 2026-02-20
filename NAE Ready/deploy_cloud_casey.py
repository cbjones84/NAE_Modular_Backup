# Cloud Casey AWS Lambda Deployment
# Deploy Cloud Casey as a persistent AWS Lambda function

import json
import boto3
import zipfile
import os
from pathlib import Path

def create_lambda_deployment_package():
    """Create deployment package for AWS Lambda"""
    
    # Files to include in the package
    files_to_include = [
        "cloud_casey_deployment.py",
        "agents/enhanced_casey.py",
        "config/cloud_casey_config.json",
        "goal_manager.py"
    ]
    
    # Create deployment package
    package_name = "cloud_casey_deployment.zip"
    
    with zipfile.ZipFile(package_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files_to_include:
            if os.path.exists(file_path):
                zip_file.write(file_path)
                print(f"Added {file_path} to deployment package")
    
    print(f"Deployment package created: {package_name}")
    return package_name

def deploy_to_aws_lambda():
    """Deploy Cloud Casey to AWS Lambda"""
    
    # Create deployment package
    package_path = create_lambda_deployment_package()
    
    # Initialize AWS clients
    lambda_client = boto3.client('lambda')
    iam_client = boto3.client('iam')
    
    # Lambda function configuration
    function_name = "cloud-casey-agent"
    role_name = "cloud-casey-lambda-role"
    
    try:
        # Create IAM role for Lambda
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            role_response = iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Role for Cloud Casey Lambda function"
            )
            
            # Attach basic execution policy
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
            )
            
            print(f"IAM role created: {role_name}")
            
        except iam_client.exceptions.EntityAlreadyExistsException:
            print(f"IAM role already exists: {role_name}")
        
        # Get role ARN
        role_response = iam_client.get_role(RoleName=role_name)
        role_arn = role_response['Role']['Arn']
        
        # Read deployment package
        with open(package_path, 'rb') as f:
            zip_content = f.read()
        
        # Create or update Lambda function
        try:
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='cloud_casey_deployment.lambda_handler',
                Code={'ZipFile': zip_content},
                Description='Cloud Casey Agent for continuous NAE analysis',
                Timeout=900,  # 15 minutes
                MemorySize=1024,
                Environment={
                    'Variables': {
                        'NAE_REPO_URL': 'https://github.com/your_username/neural-agency-engine',
                        'LOG_LEVEL': 'INFO',
                        'ANALYSIS_MODE': 'continuous'
                    }
                }
            )
            print(f"Lambda function created: {function_name}")
            
        except lambda_client.exceptions.ResourceConflictException:
            # Update existing function
            response = lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            print(f"Lambda function updated: {function_name}")
        
        # Create CloudWatch Events rule for continuous execution
        events_client = boto3.client('events')
        
        rule_name = f"{function_name}-continuous-execution"
        
        try:
            events_client.put_rule(
                Name=rule_name,
                ScheduleExpression='rate(1 hour)',  # Run every hour
                Description='Continuous execution for Cloud Casey Agent',
                State='ENABLED'
            )
            
            # Add Lambda function as target
            events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': response['FunctionArn'],
                        'Input': json.dumps({'continuous_mode': True})
                    }
                ]
            )
            
            # Add permission for CloudWatch Events to invoke Lambda
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId='cloudwatch-events',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=f"arn:aws:events:us-east-1:123456789012:rule/{rule_name}"
            )
            
            print(f"CloudWatch Events rule created: {rule_name}")
            
        except Exception as e:
            print(f"Error creating CloudWatch Events rule: {e}")
        
        print(f"\n🚀 Cloud Casey Agent deployed successfully!")
        print(f"Function Name: {function_name}")
        print(f"Function ARN: {response['FunctionArn']}")
        print(f"Execution Schedule: Every 1 hour")
        
        return response['FunctionArn']
        
    except Exception as e:
        print(f"Error deploying to AWS Lambda: {e}")
        return None

def create_docker_deployment():
    """Create Docker deployment for Cloud Casey"""
    
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cloud_casey_deployment.py .
COPY agents/ ./agents/
COPY config/ ./config/
COPY goal_manager.py .

# Create necessary directories
RUN mkdir -p logs/cloud_analysis logs/nae_reports

# Set environment variables
ENV PYTHONPATH=/app
ENV ANALYSIS_MODE=continuous
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import cloud_casey_deployment; print('Cloud Casey is healthy')"

# Run Cloud Casey
CMD ["python", "cloud_casey_deployment.py"]
"""
    
    with open("Dockerfile.cloud_casey", "w") as f:
        f.write(dockerfile_content)
    
    docker_compose_content = """
version: "3.9"

services:
  cloud-casey:
    build:
      context: .
      dockerfile: Dockerfile.cloud_casey
    container_name: cloud_casey_agent
    restart: unless-stopped
    environment:
      - NAE_REPO_URL=https://github.com/your_username/neural-agency-engine
      - LOG_LEVEL=INFO
      - ANALYSIS_MODE=continuous
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    networks:
      - nae_network
    healthcheck:
      test: ["CMD", "python", "-c", "import cloud_casey_deployment; print('healthy')"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  nae_network:
    driver: bridge
"""
    
    with open("docker-compose.cloud_casey.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("Docker deployment files created:")
    print("- Dockerfile.cloud_casey")
    print("- docker-compose.cloud_casey.yml")
    
    print("\nTo deploy with Docker:")
    print("docker-compose -f docker-compose.cloud_casey.yml up -d")

if __name__ == "__main__":
    print("🌩️ Cloud Casey Deployment Options")
    print("=" * 50)
    
    print("\n1. AWS Lambda Deployment")
    print("2. Docker Deployment")
    print("3. Both")
    
    choice = input("\nSelect deployment option (1-3): ").strip()
    
    if choice == "1":
        deploy_to_aws_lambda()
    elif choice == "2":
        create_docker_deployment()
    elif choice == "3":
        deploy_to_aws_lambda()
        create_docker_deployment()
    else:
        print("Invalid choice. Creating Docker deployment files...")
        create_docker_deployment()
