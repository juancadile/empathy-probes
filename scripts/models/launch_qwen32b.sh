#!/bin/bash
# Launch script for Qwen-2.5-32B-Instruct on Lambda instance
# Instance IP: 150.136.209.214

set -e

HOST="150.136.209.214"
USER="ubuntu"  # Default Lambda user

echo "=================================="
echo "QWEN-2.5-32B-INSTRUCT LAUNCHER"
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
echo "  nohup python3 generate_opensource_vllm.py --model qwen-2.5-32b-instruct > ../qwen32b.log 2>&1 &"
echo ""
echo "Monitor with:"
echo "  tail -f ~/empathy-probes/qwen32b.log"
echo ""
