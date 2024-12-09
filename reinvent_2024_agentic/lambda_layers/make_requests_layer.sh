#!/bin/bash

# Create a directory for our project
mkdir -p requests-lambda-layer
cd requests-lambda-layer

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM amazon/aws-lambda-python:3.13

# Create directory for the layer
RUN mkdir -p /opt/python

# Install Requests
RUN pip install requests -t /opt/python/

# Remove unnecessary files to reduce size
RUN cd /opt/python && \
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type f -name "*.pyc" -delete && \
    find . -type f -name "*.pyo" -delete && \
    find . -type f -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
EOF

# Build the Docker image
echo "Building Docker image..."
docker build -t requests-lambda-layer .

# Create a container and copy the layer contents
echo "Creating container and extracting layer..."
docker create --name temp_container requests-lambda-layer
mkdir -p python
docker cp temp_container:/opt/python/. python/
docker rm temp_container

# Create the ZIP file
echo "Creating ZIP file..."
zip -r requests-layer.zip python/

# Clean up
echo "Cleaning up..."
rm -rf python

echo "Layer has been created as requests-layer.zip"
echo "You can now upload this to AWS Lambda as a layer"