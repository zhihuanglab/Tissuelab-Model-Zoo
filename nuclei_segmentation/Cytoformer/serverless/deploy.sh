#!/bin/bash
# Deployment script for StarDist Modal serverless

echo "=================================="
echo "StarDist Modal Deployment"
echo "=================================="
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

echo "✅ Modal CLI is installed"
echo ""

# Check authentication
echo "Checking Modal authentication..."
if modal token current &> /dev/null; then
    echo "✅ Modal authentication configured"
else
    echo "⚠️  Modal not authenticated. Run: modal token new"
    exit 1
fi

echo ""

# Deploy the app
echo "Deploying StarDist app to Modal..."
modal deploy modal_app.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "Next steps:"
    echo "1. Initialize models: modal run modal_app.py::main"
    echo "2. Test deployment: python client_example.py"
    echo "3. Monitor logs: modal app logs stardist-segmentation-v1"
else
    echo ""
    echo "❌ Deployment failed"
    exit 1
fi

