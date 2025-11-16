# Vast.ai Setup - Faster Alternative (~30 min, ~$1)

## Prerequisites

1. **Create account**: [vast.ai](https://vast.ai) (needs credit card for ~$5 minimum)
2. **Push code to GitHub**:
   ```bash
   cd /Users/juancadile/Documents/samuel-anthropic/empathy-action-probes
   git add .
   git commit -m "Add cross-model validation scripts"
   git push origin cloud-strengthening
   ```

---

## Step-by-Step

### 1. Find Instance

Go to [vast.ai/console/create](https://vast.ai/console/create/) and use these filters:

**Recommended filters:**
- GPU: RTX 3090 or A5000 (good price/performance)
- VRAM: ≥20GB
- Disk: ≥50GB
- Reliability: ≥95%

**Or search directly:**
```
gpu_name: RTX 3090 reliability > 0.95 num_gpus=1 disk_space > 50
```

**Expected cost**: $0.30-0.50/hour

---

### 2. Launch Instance

1. Click **Rent** on a good option
2. **Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
3. **Disk Space**: 50GB
4. **Options**:
   - ✅ Enable Direct SSH
   - ✅ Run Jupyter (optional)
5. Click **Create**

Wait ~2-3 min for instance to start.

---

### 3. SSH Into Instance

Click **Connect** → Copy SSH command, should look like:
```bash
ssh -p 12345 root@12.34.56.78
```

Paste into your terminal.

---

### 4. Setup on Instance

Once connected, run these commands:

```bash
# Install dependencies
pip install transformers accelerate bitsandbytes datasets scikit-learn

# Clone repo
git clone https://github.com/juancadile/empathy-probes.git
cd empathy-probes
git checkout cloud-strengthening

# Verify GPU
nvidia-smi

# Check data
ls data/contrastive_pairs/
wc -l data/contrastive_pairs/*.jsonl
```

---

### 5. Run Experiments

```bash
# Run both models (~30 min on RTX 3090)
python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b
```

**Monitor progress**: You'll see progress bars for each layer.

---

### 6. Download Results

**Option A: SCP (from your local machine)**

Open a NEW terminal window (don't close SSH):

```bash
# Replace PORT and IP with your instance details
scp -P 12345 -r root@12.34.56.78:~/empathy-probes/results/cross_model_validation ~/Downloads/
```

**Option B: GitHub**

In the SSH session:
```bash
# Commit results
git add results/cross_model_validation/
git commit -m "Cross-model validation results: Qwen + Dolphin"
git push origin cloud-strengthening

# Then pull locally
```

---

### 7. Stop Instance

**IMPORTANT**: Don't forget to stop the instance!

1. Go to [vast.ai/console/instances](https://vast.ai/console/instances/)
2. Click **Destroy** on your instance
3. Verify billing stopped

**Cost**: ~$0.30/hour × 0.5 hours = **~$0.15-0.50 total**

---

## Quick Commands Summary

```bash
# On vast.ai instance
pip install transformers accelerate bitsandbytes datasets scikit-learn
git clone https://github.com/juancadile/empathy-probes.git
cd empathy-probes && git checkout cloud-strengthening
python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b

# Download (from local machine, separate terminal)
scp -P <PORT> -r root@<IP>:~/empathy-probes/results/cross_model_validation ~/Downloads/

# Don't forget to destroy instance!
```

---

## Time & Cost Estimate

| GPU | Time (2 models) | Cost/hour | Total Cost |
|-----|-----------------|-----------|------------|
| RTX 3090 | ~30 min | $0.30-0.40 | **~$0.20** |
| A5000 | ~35 min | $0.50-0.60 | **~$0.35** |
| RTX 4090 | ~20 min | $0.70-0.80 | **~$0.30** |

---

## Troubleshooting

**Can't SSH?**
- Check instance status is "running"
- Try "Reset SSH" button in vast.ai console
- Verify SSH port in connection string

**Out of disk space?**
- Models + results need ~40GB
- Increase disk space allocation when renting

**Slow download speeds?**
- Models are 7-8GB each (first time download)
- Subsequent runs use cached models (much faster)

**Instance disconnects?**
- Use `screen` or `tmux`:
  ```bash
  screen -S experiment
  python src/probe_extraction_cross_model.py --models qwen2.5-7b dolphin-llama-3.1-8b
  # Press Ctrl+A, then D to detach
  # Reconnect: screen -r experiment
  ```

---

## Comparison: Colab vs Vast.ai

| Factor | Colab | Vast.ai |
|--------|-------|---------|
| **Time** | ~1.5 hours | ~30 min |
| **Cost** | Free | ~$0.50 |
| **Setup** | Easy (browser) | Medium (SSH) |
| **Disconnect Risk** | Higher | Lower |
| **Best For** | First run, testing | Production, speed |

**Recommendation**: Try Colab first. If it disconnects or is too slow, switch to vast.ai.

---

## Next Steps After Results

1. Download results to local machine
2. Analyze: `cat cross_model_validation/all_models_results.json`
3. Compare with Phi-3 baseline (AUROC 1.0, Layer 12)
4. Update paper with cross-model validation findings
