"""Generate training loss curve and benchmark results visualizations."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

REPO = "/mnt/lustre-grete/projects/nii00224/mahmoud/Discrete-OCR-Diffusion-Models"
ASSETS = os.path.join(REPO, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Training Loss Curve (from trainer_state.json)
# ──────────────────────────────────────────────
trainer_state_path = "/mnt/lustre-grete/projects/nii00224/mahmoud/DiffuQwen/checkpoints/diffuqwen-hf-20260127-013517/checkpoint-20000/trainer_state.json"

print("Loading trainer_state.json...")
with open(trainer_state_path) as f:
    state = json.load(f)

log_history = state["log_history"]
train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
eval_entries  = [e for e in log_history if "eval_loss" in e]

steps  = [e["step"] for e in train_entries]
losses = [e["loss"] for e in train_entries]

eval_steps  = [e["step"]      for e in eval_entries]
eval_losses = [e["eval_loss"] for e in eval_entries]

min_train_loss = min(losses)
min_train_step = steps[losses.index(min_train_loss)]
min_eval_loss  = min(eval_losses)
min_eval_step  = eval_steps[eval_losses.index(min_eval_loss)]

# Detect extreme spikes (> mean + 5*std of smoothed region)
window = 50
smoothed_full = np.convolve(losses, np.ones(window)/window, mode='same')
residual = np.array(losses) - smoothed_full
threshold = np.mean(residual) + 5 * np.std(residual)
spike_mask = residual > threshold
spike_steps  = [steps[i]  for i in range(len(steps))  if spike_mask[i]]
spike_losses = [losses[i] for i in range(len(losses)) if spike_mask[i]]

# ── Combined figure: 2 subplots ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
fig.suptitle(f'Training Metrics — Step 20000 (Epoch 9.55)', fontsize=14, fontweight='bold', color='#1a237e')

# ── Subplot 1: Training Loss ────────────────────────────
ax1.plot(steps, losses, color='#5bc8f5', linewidth=0.6, alpha=0.5, label='Train Loss')

# Moving average
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
smoothed_steps = steps[window-1:]
ax1.plot(smoothed_steps, smoothed, color='#f5a623', linewidth=2.0, label=f'Moving Avg (w={window})')

# Eval loss overlaid on train subplot
ax1.plot(eval_steps, eval_losses, color='#00c853', linewidth=1.5,
         marker='o', markersize=3, linestyle='--', label='Eval Loss')

# Extreme spikes
ax1.scatter(spike_steps, spike_losses, marker='X', color='#c62828', s=60, zorder=5,
            label=f'Extreme Spikes ({len(spike_steps)})')

# Min annotation
ax1.axhline(y=min_train_loss, color='#00c853', linestyle=':', linewidth=1, alpha=0.6)
ax1.annotate(f'Min: {min_train_loss:.4f} @ step {min_train_step}',
             xy=(min_train_step, min_train_loss),
             xytext=(min_train_step - 4000, min_train_loss + 0.5),
             color='#00c853', fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#00c853', lw=1))

ax1.set_title('Training Loss', fontsize=11, fontweight='bold', color='#1a237e')
ax1.set_xlabel('Step', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.set_xlim(0, 20000)
ax1.set_ylim(0, 200)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.25)

# ── Subplot 2: Evaluation Loss ──────────────────────────
ax2.plot(eval_steps, eval_losses, color='#00c853', linewidth=2.2,
         marker='o', markersize=5, linestyle='--', label='Eval Loss')

ax2.axhline(y=min_eval_loss, color='#00c853', linestyle=':', linewidth=1, alpha=0.6)
ax2.annotate(f'Min: {min_eval_loss:.4f} @ step {min_eval_step}',
             xy=(min_eval_step, min_eval_loss),
             xytext=(min_eval_step - 7000, min_eval_loss + 0.01),
             color='#00c853', fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#00c853', lw=1))

ax2.set_title('Evaluation Loss', fontsize=11, fontweight='bold', color='#1a237e')
ax2.set_xlabel('Step', fontsize=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.set_xlim(eval_steps[0] - 200, 20500)
ax2.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "diffuqwen_training_loss.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {ASSETS}/diffuqwen_training_loss.png")
plt.close()

# Keep a legacy zoomed single-subplot version as well
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
zoom_mask = [i for i, s in enumerate(steps) if s >= 500]
zoom_steps  = [steps[i]  for i in zoom_mask]
zoom_losses = [losses[i] for i in zoom_mask]
ax.plot(zoom_steps, zoom_losses, color='#5bc8f5', linewidth=0.7, alpha=0.45, label='Train Loss')
if len(zoom_losses) > window:
    sm = np.convolve(zoom_losses, np.ones(window)/window, mode='valid')
    ax.plot(zoom_steps[window-1:], sm, color='#f5a623', linewidth=2.0, label=f'Moving Avg (w={window})')
ax.plot(eval_steps, eval_losses, color='#00c853', linewidth=1.5,
        marker='o', markersize=4, linestyle='--', label='Eval Loss')
ax.axvline(x=10000, color='#e53935', linestyle='--', linewidth=1.5, alpha=0.7, label='Anneal complete (step 10k)')
ax.annotate(f'Min train: {min_train_loss:.4f} @ step {min_train_step}',
            xy=(min_train_step, min_train_loss), xytext=(14000, 3.5),
            color='#00c853', fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00c853', lw=1))
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('DiffuQwen-VL Training Loss (Zoomed, steps 500–20k)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(500, 20000)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "diffuqwen_training_loss_zoomed.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {ASSETS}/diffuqwen_training_loss_zoomed.png")
plt.close()

# ──────────────────────────────────────────────
# 2. Benchmark Results Comparison Bar Chart
# ──────────────────────────────────────────────
categories = ['Absent', 'Baseline', 'Math', 'Order', 'Present', 'Table']
diffuqwen =  [98.3,    99.9,      0.7,   1.5,    4.2,     2.8]
lavida =     [96.0,    97.0,      1.4,   11.9,   11.4,    17.5]
olmocr =     [96.0,    99.9,      85.9,  72.8,   60.5,    82.4]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
bars1 = ax.bar(x - width, diffuqwen, width, label='DiffuQwen-VL', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x, lavida, width, label='LaViDa-OCR (no pool)', color='#FF9800', edgecolor='white')
bars3 = ax.bar(x + width, olmocr, width, label='olmOCR (AR Baseline)', color='#4CAF50', edgecolor='white')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('olmOCR-bench Results: Diffusion vs. Autoregressive', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 110)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 5:
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "benchmark_comparison.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {ASSETS}/benchmark_comparison.png")
plt.close()

# ──────────────────────────────────────────────
# 3. LaViDa Per-JSONL Breakdown
# ──────────────────────────────────────────────
jsonl_cats = ['arxiv_math', 'baseline', 'headers\n_footers', 'long_tiny\n_text', 'multi\n_column', 'old_scans', 'old_scans\n_math', 'table_tests']
lavida_jsonl = [1.3, 97.1, 95.7, 9.7, 13.8, 21.3, 2.2, 17.6]
lavida_tests = [2927, 1394, 760, 442, 884, 526, 458, 1022]

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
colors = ['#F44336' if v < 20 else '#FF9800' if v < 50 else '#4CAF50' for v in lavida_jsonl]
bars = ax.bar(range(len(jsonl_cats)), lavida_jsonl, color=colors, edgecolor='white')
ax.set_xticks(range(len(jsonl_cats)))
ax.set_xticklabels(jsonl_cats, fontsize=9)
ax.set_ylabel('Pass Rate (%)', fontsize=12)
ax.set_title('LaViDa-OCR Per-JSONL Breakdown (checkpoint-10750, no pooling)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(True, axis='y', alpha=0.3)

for bar, val, n in zip(bars, lavida_jsonl, lavida_tests):
    ax.annotate(f'{val}%\n({n} tests)', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "lavida_jsonl_breakdown.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {ASSETS}/lavida_jsonl_breakdown.png")
plt.close()

# ──────────────────────────────────────────────
# 4. LaViDa Pooling Comparison
# ──────────────────────────────────────────────
pool_cats = ['Absent', 'Baseline', 'Math', 'Order', 'Present', 'Table']
with_pool =    [99.5, 17.5, 0.0, 0.0, 0.0, 0.0]
without_pool = [96.0, 97.0, 1.4, 11.9, 11.4, 17.5]

x = np.arange(len(pool_cats))
width = 0.35

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.bar(x - width/2, with_pool, width, label='With 2×2 Pooling (980 tokens)', color='#F44336', edgecolor='white')
ax.bar(x + width/2, without_pool, width, label='Without Pooling (3,645 tokens)', color='#4CAF50', edgecolor='white')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('LaViDa-OCR: Impact of Visual Pooling on OCR', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pool_cats, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 110)
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "lavida_pooling_comparison.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {ASSETS}/lavida_pooling_comparison.png")
plt.close()

# ──────────────────────────────────────────────
# 6. LaViDa Training Loss Curve
# ──────────────────────────────────────────────
lavida_state_path = "/mnt/lustre-grete/projects/nii00224/mahmoud/Lavida-experiment/LaViDa/checkpoints/lavida-stage2-olmocr-opt-nopool/checkpoint-10750/trainer_state.json"

print("Loading LaViDa trainer_state.json...")
with open(lavida_state_path) as f:
    lv_state = json.load(f)

lv_log = lv_state["log_history"]
lv_train = [e for e in lv_log if "loss" in e and "eval_loss" not in e]
lv_steps  = [e["step"] for e in lv_train]
lv_losses = [e["loss"] for e in lv_train]

lv_min_loss = min(lv_losses)
lv_min_step = lv_steps[lv_losses.index(lv_min_loss)]

lv_window = 50
lv_smoothed_full = np.convolve(lv_losses, np.ones(lv_window)/lv_window, mode='same')
lv_residual = np.array(lv_losses) - lv_smoothed_full
lv_threshold = np.mean(lv_residual) + 5 * np.std(lv_residual)
lv_spike_mask = lv_residual > lv_threshold
lv_spike_steps  = [lv_steps[i]  for i in range(len(lv_steps))  if lv_spike_mask[i]]
lv_spike_losses = [lv_losses[i] for i in range(len(lv_losses)) if lv_spike_mask[i]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
fig.suptitle(f'LaViDa-OCR Training — Step 10750 (no pooling)', fontsize=14, fontweight='bold', color='#1a237e')

# Top: full training loss
ax1.plot(lv_steps, lv_losses, color='#5bc8f5', linewidth=0.4, alpha=0.4, label='Train Loss')
lv_sm = np.convolve(lv_losses, np.ones(lv_window)/lv_window, mode='valid')
lv_sm_steps = lv_steps[lv_window-1:]
ax1.plot(lv_sm_steps, lv_sm, color='#f5a623', linewidth=2.0, label=f'Moving Avg (w={lv_window})')
ax1.scatter(lv_spike_steps, lv_spike_losses, marker='X', color='#c62828', s=50, zorder=5,
            label=f'Extreme Spikes ({len(lv_spike_steps)})')
ax1.axhline(y=lv_min_loss, color='#00c853', linestyle=':', linewidth=1, alpha=0.6)
ax1.annotate(f'Min: {lv_min_loss:.4f} @ step {lv_min_step}',
             xy=(lv_min_step, lv_min_loss),
             xytext=(lv_min_step - 3000, lv_min_loss + 0.3),
             color='#00c853', fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#00c853', lw=1))
ax1.set_title('Training Loss (full)', fontsize=11, fontweight='bold', color='#1a237e')
ax1.set_xlabel('Step', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.set_xlim(0, max(lv_steps) + 100)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.25)

# Bottom: zoomed (after initial transient)
zoom_start = 500
zm = [(s, l) for s, l in zip(lv_steps, lv_losses) if s >= zoom_start]
zm_steps, zm_losses = zip(*zm)
ax2.plot(zm_steps, zm_losses, color='#5bc8f5', linewidth=0.5, alpha=0.4, label='Train Loss')
if len(zm_losses) > lv_window:
    zm_sm = np.convolve(zm_losses, np.ones(lv_window)/lv_window, mode='valid')
    ax2.plot(list(zm_steps)[lv_window-1:], zm_sm, color='#f5a623', linewidth=2.0, label=f'Moving Avg (w={lv_window})')
ax2.axhline(y=lv_min_loss, color='#00c853', linestyle=':', linewidth=1, alpha=0.6)
ax2.annotate(f'Min: {lv_min_loss:.4f} @ step {lv_min_step}',
             xy=(lv_min_step, lv_min_loss),
             xytext=(lv_min_step - 3000, lv_min_loss + 0.15),
             color='#00c853', fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#00c853', lw=1))
ax2.set_title('Training Loss (zoomed, steps 500+)', fontsize=11, fontweight='bold', color='#1a237e')
ax2.set_xlabel('Step', fontsize=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.set_xlim(zoom_start, max(lv_steps) + 100)
ax2.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "lavida_training_loss.png"), dpi=150, bbox_inches='tight')
print(f"Saved: {ASSETS}/lavida_training_loss.png")
plt.close()

print("\nAll visualizations generated successfully!")
print(f"Files in {ASSETS}:")
for f in sorted(os.listdir(ASSETS)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(ASSETS, f))
        print(f"  {f} ({size/1024:.1f} KB)")
