#!/bin/bash
# Launch script for Gemma-2-27B-it on Lambda instance
# Instance IP: 158.101.120.239

set -e

HOST="158.101.120.239"
USER="ubuntu"  # Default Lambda user

echo "=================================="
echo "GEMMA-2-27B-IT LAUNCHER"
echo "=================================="
echo "Instance: $HOST"
echo ""

# Upload files to instance
echo "Uploading files to instance..."
scp data/eia_scenarios/scenarios.json $USER@$HOST:~/empathy-probes/data/eia_scenarios/
scp src/generate_opensource_vllm.py $USER@$HOST:~/empathy-probes/src/
scp lambda_setup.sh $USER@$HOST:~/empathy-probes/

echo ""
echo "Files uploaded. Now SSH into instance and run:"
echo ""
echo "  ssh $USER@$HOST"
echo "  cd ~/empathy-probes"
echo "  bash lambda_setup.sh"
echo "  cd src"
echo "  nohup python3 generate_opensource_vllm.py --model gemma-2-27b-it > ../gemma27b.log 2>&1 &"
echo ""
echo "Monitor with:"
echo "  tail -f ~/empathy-probes/gemma27b.log"
echo ""
