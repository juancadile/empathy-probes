# Quick Start: Launch All 3 Models

## Instances Ready:
- ✅ Llama-70B: `150.136.112.32`
- ✅ Gemma-27B: `158.101.120.239`
- ✅ Qwen-32B: `150.136.209.214`

---

## Step 1: Upload files to all instances (run locally)

```bash
# Upload to Llama instance
./launch_llama70b.sh

# Upload to Gemma instance
./launch_gemma27b.sh

# Upload to Qwen instance
./launch_qwen32b.sh
```

---

## Step 2: Setup and start each instance

### Open 3 terminal windows:

**Terminal 1 - Llama-70B:**
```bash
ssh ubuntu@150.136.112.32
cd ~/empathy-probes && bash lambda_setup.sh
cd src
nohup python3 generate_opensource_vllm.py --model llama-3.1-70b-instruct > ../llama70b.log 2>&1 &
exit  # Can close after nohup starts
```

**Terminal 2 - Gemma-27B:**
```bash
ssh ubuntu@158.101.120.239
cd ~/empathy-probes && bash lambda_setup.sh
cd src
nohup python3 generate_opensource_vllm.py --model gemma-2-27b-it > ../gemma27b.log 2>&1 &
exit  # Can close after nohup starts
```

**Terminal 3 - Qwen-32B:**
```bash
ssh ubuntu@150.136.209.214
cd ~/empathy-probes && bash lambda_setup.sh
cd src
nohup python3 generate_opensource_vllm.py --model qwen-2.5-32b-instruct > ../qwen32b.log 2>&1 &
exit  # Can close after nohup starts
```

---

## Step 3: Monitor progress (run locally)

```bash
./monitor_lambda_progress.sh
```

This will show live updates every 30 seconds showing:
- Pair counts for each model
- Last 3 log lines
- Connection status

Press `Ctrl+C` to exit monitoring (doesn't stop generation)

---

## Step 4: Download results when done (~6 hours)

```bash
./download_results.sh
```

Files will be saved to `data/contrastive_pairs/`:
- `generation_progress_llama-70b.jsonl` (500 pairs)
- `generation_progress_gemma-27b.jsonl` (500 pairs)
- `generation_progress_qwen-32b.jsonl` (500 pairs)

---

## Expected Timeline

- **Gemma-27B**: Finishes first (~2.5 hours)
- **Qwen-32B**: Finishes second (~3 hours)
- **Llama-70B**: Finishes last (~6 hours)

**Total cost:** ~$23 for all 1,500 pairs

---

## Important: Terminate instances when done!

After downloading results:
1. Go to Lambda Labs dashboard
2. Terminate all 3 instances
3. Saves money (~$4/hr if left running)

---

## Troubleshooting

**If upload fails:** Instances might still be booting. Wait 2-3 minutes and retry.

**If SSH fails:** Check instance status in Lambda dashboard.

**To check if running:**
```bash
ssh ubuntu@150.136.112.32 "ps aux | grep generate_opensource"
```

**To restart on crash:**
```bash
ssh ubuntu@150.136.112.32
cd ~/empathy-probes/src
nohup python3 generate_opensource_vllm.py --model llama-3.1-70b-instruct > ../llama70b.log 2>&1 &
```

See [LAMBDA_INSTRUCTIONS.md](LAMBDA_INSTRUCTIONS.md) for full details.
