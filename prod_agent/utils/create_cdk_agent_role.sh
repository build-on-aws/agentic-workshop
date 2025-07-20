#!/bin/bash
# Script to create a CDK agent role using the Python script

# Default values
AGENT_NAME="cdk-agent"
PROFILE=""
REGION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --agent-name)
      AGENT_NAME="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --agent-name NAME   Name of the agent (default: cdk-agent)"
      echo "  --profile PROFILE   AWS profile to use"
      echo "  --region REGION     AWS region to use"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build the command with optional arguments
CMD="python3 create_cdk_agent_role.py --agent-name \"$AGENT_NAME\""

if [ -n "$PROFILE" ]; then
  CMD="$CMD --profile \"$PROFILE\""
fi

if [ -n "$REGION" ]; then
  CMD="$CMD --region \"$REGION\""
fi

# Execute the command
echo "Creating CDK agent role with command: $CMD"
eval $CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
  echo "CDK agent role created successfully!"
else
  echo "Failed to create CDK agent role. Check the logs for details."
  exit 1
fi