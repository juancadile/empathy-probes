# Lambda Labs Open-Source Model Generation

Complete setup guide for generating 1,500 contrastive pairs across 3 open-source models in parallel.

## Instance Configuration

| Model | Instance IP | GPU | Status |
|-------|------------|-----|--------|
| Llama-3.1-70B-Instruct | 150.136.112.32 | A100 40GB | Booting |
| Gemma-2-27B-it | 158.101.120.239 | A100 40GB | Booting |
| Qwen-2.5-32B-Instruct | 150.136.209.214 | A100 40GB | Booting |

## Quick Start (Parallel Launch)

### 1. Make scripts executable
```bash
chmod +x launch_*.sh monitor_lambda_progress.sh download_results.sh lambda_setup.sh
```

### 2. Launch all 3 instances in parallel

**Terminal 1 - Llama-70B:**
```bash
./launch_llama70b.sh
# Follow on-screen instructions to SSH and start generation
```

**Terminal 2 - Gemma-27B:**
```bash
./launch_gemma27b.sh
# Follow on-screen instructions to SSH and start generation
```

**Terminal 3 - Qwen-32B:**
```bash
./launch_qwen32b.sh
# Follow on-screen instructions to SSH and start generation
```

### 3. Monitor progress (Terminal 4)
```bash
./monitor_lambda_progress.sh
```

### 4. Download results when complete
```bash
./download_results.sh
```

---

## Detailed Instructions

### Step 1: Upload Files to Each Instance

For each instance, run the corresponding launcher script:

```bash
./launch_llama70b.sh   # Uploads to 150.136.112.32
./launch_gemma27b.sh   # Uploads to 158.101.120.239
./launch_qwen32b.sh    # Uploads to 150.136.209.214
```

### Step 2: SSH Into Each Instance and Setup

**Llama-70B Instance:**
```bash
ssh ubuntu@150.136.112.32
cd ~/empathy-probes
bash lambda_setup.sh
```

**Gemma-27B Instance:**
```bash
ssh ubuntu@158.101.120.239
cd ~/empathy-probes
bash lambda_setup.sh
```

**Qwen-32B Instance:**
```bash
ssh ubuntu@150.136.209.214
cd ~/empathy-probes
bash lambda_setup.sh
```

### Step 3: Start Generation (on each instance)

**Llama-70B:**
```bash
cd ~/empathy-probes/src
nohup python3 generate_opensource_vllm.py --model llama-3.1-70b-instruct > ../llama70b.log 2>&1 &
```

**Gemma-27B:**
```bash
cd ~/empathy-probes/src
nohup python3 generate_opensource_vllm.py --model gemma-2-27b-it > ../gemma27b.log 2>&1 &
```

**Qwen-32B:**
```bash
cd ~/empathy-probes/src
nohup python3 generate_opensource_vllm.py --model qwen-2.5-32b-instruct > ../qwen32b.log 2>&1 &
```

**Note:** After running `nohup`, you can safely close the SSH connection. The process continues running.

### Step 4: Monitor Progress

From your local machine:
```bash
./monitor_lambda_progress.sh
```

Or check individual instances:
```bash
ssh ubuntu@150.136.112.32 "tail -f ~/empathy-probes/llama70b.log"
ssh ubuntu@158.101.120.239 "tail -f ~/empathy-probes/gemma27b.log"
ssh ubuntu@150.136.209.214 "tail -f ~/empathy-probes/qwen32b.log"
```

### Step 5: Download Results

When generation completes (or to get intermediate results):
```bash
./download_results.sh
```

Results will be saved to `data/contrastive_pairs/`:
- `generation_progress_llama-70b.jsonl`
- `generation_progress_gemma-27b.jsonl`
- `generation_progress_qwen-32b.jsonl`

---

## Expected Timeline

| Model | Pairs | Est. Time | Status |
|-------|-------|-----------|--------|
| Llama-70B | 500 | ~6 hours | ⏳ Running |
| Gemma-27B | 500 | ~2.5 hours | ⏳ Running |
| Qwen-32B | 500 | ~3 hours | ⏳ Running |

**Total wall-clock time:** ~6 hours (parallel execution)
**Total cost:** ~$23 (6 hrs × $3.87/hr)

---

## Features

### Crash Resistance
- Saves every 10 pairs automatically
- Resume from last checkpoint on restart
- Progress tracked in JSONL (append-only)

### Temperature Cycling
- Cycles through [0.7, 0.8, 0.9, 1.0]
- Ensures diverse responses
- Matches closed-source model strategy

### vLLM Acceleration
- 10-20× faster than transformers
- Batched inference (2 completions at once)
- Optimized GPU memory usage

---

## Troubleshooting

### Instance won't connect
```bash
# Check if instance is up
ping 150.136.112.32

# Try with verbose SSH
ssh -v ubuntu@150.136.112.32
```

### Generation stopped
```bash
# SSH into instance and check logs
ssh ubuntu@150.136.112.32
tail -50 ~/empathy-probes/llama70b.log

# Restart if needed
cd ~/empathy-probes/src
nohup python3 generate_opensource_vllm.py --model llama-3.1-70b-instruct > ../llama70b.log 2>&1 &
```

### Out of memory error
vLLM is configured with `gpu_memory_utilization=0.90`. If OOM occurs:
1. Edit `generate_opensource_vllm.py`
2. Change `gpu_memory_utilization` to `0.85`
3. Restart generation

### Download fails
```bash
# Download individual files
scp ubuntu@150.136.112.32:~/empathy-probes/data/contrastive_pairs/generation_progress_llama-70b.jsonl data/contrastive_pairs/
```

---

## After Completion

1. **Download all results:**
   ```bash
   ./download_results.sh
   ```

2. **Verify pair counts:**
   ```bash
   wc -l data/contrastive_pairs/generation_progress_*.jsonl
   ```

3. **Terminate instances** (to stop billing):
   - Log into Lambda Labs dashboard
   - Terminate all 3 instances
   - **Important:** Do this promptly to avoid charges

4. **Update dataset builder:**
   - Add new models to `build_final_dataset.py`
   - Run selection if needed (models already at 500 pairs)

---

## Cost Tracking

| Resource | Rate | Time | Cost |
|----------|------|------|------|
| 3× A100 40GB | $3.87/hr | 6 hrs | $23.22 |
| Storage (minimal) | ~$0.01/hr | 6 hrs | $0.06 |
| **Total** | | | **~$23.28** |

Very efficient for 1,500 high-quality pairs!
