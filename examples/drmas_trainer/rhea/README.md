# 在 Rhea 上復現 Dr. MAS（A6000、general partition）

最簡單的復現路線是 **Math 任務**：只需要下載 HuggingFace 資料集，不用像 Search 任務那樣另外架 faiss retriever server（要額外的 conda 環境、約 60GB 的 wiki index，每張 GPU 還要多吃約 6GB 顯存）。

## 和原始設定的差別

| | 論文 / `run_math.sh` | 這裡（`run_math_a6000.slurm`） |
| :-- | :-- | :-- |
| GPU | 4 × H100 80GB | 4 × A6000 48GB |
| 模型 | 2 × Qwen3-4B（non-sharing） | 1 × Qwen3-1.7B（sharing，兩個 agent 共用） |
| micro batch / GPU | 4 | 2 |
| logger | console + wandb | console（可用 `LOGGER` 打開 wandb） |
| `ray_init.num_cpus` | 沒設 | 設成 `$SLURM_CPUS_PER_TASK`（在 SLURM 上沒設會卡住） |

演算法本身（GRPO + agent-wise normalization `group_by_agent_id=True`、group size 8、batch 32、Solver/Verifier 最多 2 輪）都沒改。

## 步驟

```bash
# 0. clone
git clone https://github.com/langfengQ/DrMAS.git && cd DrMAS

# 1. 開一個互動式 GPU job 來裝環境（flash-attn 安裝需要 CUDA）
srun -p general --gres=gpu:A6000:1 -c 8 --mem=64G -t 2:00:00 --pty bash
bash examples/drmas_trainer/rhea/setup_env.sh      # 建 conda env + 下載/處理資料到 ~/data/drmas_math
exit

# 2. 送訓練 job
mkdir -p logs
sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm
tail -f logs/drmas-math-<jobid>.out

# 3. 訓練完後用完整 test set 評估（pass@16 / avg@16）
sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm eval
```

`trainer.resume_mode` 預設是 `auto`，所以如果 job 被時間上限砍掉，直接再 `sbatch` 一次就會從 `checkpoints/DrMAS_math/<experiment_name>/` 最新的 checkpoint 接著跑（每 50 step 存一次）。

## 常用調整

```bash
# 用 wandb（先 `wandb login`）
LOGGER="['console','wandb']" sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm

# 更接近論文：Qwen3-4B、non-sharing（顯存比較緊，OOM 的話把 micro batch 改成 [1,1]）
MODEL=Qwen/Qwen3-4B MODEL_SHARING=False sbatch examples/drmas_trainer/rhea/run_math_a6000.slurm

# 只拿到 2 張卡：改 --gres，NGPUS 會自動跟著 SLURM 變
sbatch --gres=gpu:A6000:2 examples/drmas_trainer/rhea/run_math_a6000.slurm
```

## 可能要依 Rhea 實際設定修改的地方

- `--gres=gpu:A6000:4`：GPU type 字串要跟 Rhea 一致（用 `sinfo -p general -o "%N %G"` 看）。
- `--time=2-00:00:00`、`--mem=192G`、`--cpus-per-task=32`：看 general partition 的上限（`scontrol show partition general`）。
- CUDA module：如果系統需要 `module load cuda/...`，把兩個腳本裡註解掉的那行打開。
- `DRMAS_CACHE`：預設 `~/drmas_cache`（HF 模型快取、pip 快取）。如果 home 有 quota，改成 scratch 路徑，例如 `export DRMAS_CACHE=/scratch/$USER/drmas_cache`。

## 常見問題

- **卡在 Ray 啟動不動**：確認 `ray_init.num_cpus` 有設、而且不超過 `--cpus-per-task`。
- **OOM**：先把 `actor_ppo_micro_batch_size_per_gpu` 改成 `'[1,1]'`、`log_prob_micro_batch_size_per_gpu` 改成 2，或把 `gpu_memory_utilization` 從 0.5 降到 0.4。
- **flash-attn 安裝失敗**：通常是找不到對應的 prebuilt wheel 而開始自己編譯；確認 python 3.12、torch 2.6（由 sglang 0.4.6.post5 帶進來），並在 GPU 節點上裝。
