#!/usr/bin/env python3
"""
Script to create an IAM role for the CDK agent using AgentCore.
This role will have the necessary permissions for the CDK agent to function properly.
"""

import argparse
import boto3
import json
import sys
import os
import logging
from helper_funcs import create_agentcore_role

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to create the CDK agent role.
    """
    parser = argparse.ArgumentParser(description='Create IAM role for CDK Agent')
    parser.add_argument('--agent-name', type=str, default='cdk-agent',
                        help='Name of the agent (default: cdk-agent)')
    parser.add_argument('--profile', type=str, help='AWS profile to use')
    parser.add_argument('--region', type=str, help='AWS region to use')
    
    args = parser.parse_args()
    
    # Set AWS profile if provided
    if args.profile:
        os.environ['AWS_PROFILE'] = args.profile
        logger.info(f"Using AWS profile: {args.profile}")
    
    # Set AWS region if provided
    if args.region:
        os.environ['AWS_DEFAULT_REGION'] = args.region
        logger.info(f"Using AWS region: {args.region}")
    
    try:
        # Create the AgentCore role for the CDK agent
        logger.info(f"Creating AgentCore role for {args.agent_name}...")
        role = create_agentcore_role(args.agent_name)
        
        # Print role information
        logger.info(f"Successfully created role: {role['Role']['RoleName']}")
        logger.info(f"Role ARN: {role['Role']['Arn']}")
        
        # Save role information to a file
        role_info = {
            'role_name': role['Role']['RoleName'],
            'role_arn': role['Role']['Arn']
        }
        
        with open(f"{args.agent_name}_role_info.json", 'w') as f:
            json.dump(role_info, f, indent=2)
            logger.info(f"Role information saved to {args.agent_name}_role_info.json")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error creating CDK agent role: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())