# Neural Observer 📉

**Failure-Aware Diagnostics for Deep Neural Models.**

`neural-observer` is a lightweight instrumentation library designed to diagnose training instabilities (Gradient Explosions, Loss Spikes, NaNs) in deep multi-modal models. Unlike standard loggers (W&B, TensorBoard) which only record *averages*, Neural Observer acts as a **Flight Recorder**: it maintains a high-frequency circular buffer of per-layer statistics in RAM and only dumps the history to disk when a crash is detected.

![Crash Analysis](assets/demo_crash.png)

*Above: A capture of a Gradient Explosion on a standard ConvNet. Note how Layer 6 (Green) explodes exponentially at step 359, while Layer 8 (Red) remains stable. The system captured the exact moment of failure automatically.*

##  Key Features

* **Zero-Config Instrumentation:** Auto-attaches to `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, and `nn.Embedding` via PyTorch hooks.
* **Low-Overhead Ring Buffer:** Stores the last `N` steps of training dynamics (gradients, activations) in CPU RAM. Zero VRAM penalty.
* **Smart Trigger Engine:** Automatically halts training and saves a crash report when:
  * Loss spikes > 3x the moving average.
  * Gradient norms exceed a safety threshold.
  * NaNs or Infs appear in the loss.
* **Crash Forensics:** Dumps a JSON report containing the exact sequence of layer stats leading up to the failure.

##  Installation

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

##  How to Use

Instrumenting your model takes just a few lines of code.

```python
import torch
from observer import HookManager, RingBuffer, TriggerEngine

# 1. Setup Your Model
model = MyDeepModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Attach Observer (The Eyes)
# log_every=10 reduces overhead by only sampling every 10th step
observer = HookManager(log_every=10)
observer.attach(model)

# 3. Setup Recorder & Triggers (The Memory & Brain)
buffer = RingBuffer(capacity=100)
triggers = TriggerEngine(grad_threshold=100.0, spike_factor=3.0)

# 4. Training Loop
for step, batch in enumerate(dataloader):
    # Reset stats at the start of the step
    observer.reset_step()
    
    # Standard Forward/Backward
    loss = criterion(model(batch), target)
    loss.backward()
    optimizer.step()
    
    # --- Neural Observer Logic ---
    # 1. Capture the step data to the buffer
    buffer.add(step, observer.grad_norms, observer.act_norms, loss.item())
    
    # 2. Check for failure signals
    fail_reason = triggers.check(loss.item(), observer.grad_norms)
    if fail_reason:
        print(f" Crash Detected: {fail_reason}")
        # Dump the last 100 steps to disk for analysis
        buffer.save_json("crash_dump.json", trigger_reason=fail_reason)
        break
```

##  Experiments & Reproduction

This repository contains a reproduction script to force a Gradient Explosion on a ConvNet trained on MNIST to demonstrate the system's capabilities.

### 1. Run the Doomed Experiment

This script trains a ConvNet with an aggressively high learning rate (lr=100.0) to force a failure.

```bash
python -m experiments.train_mnist_failure
```

**Outcome:** The system will detect the explosion (usually around step ~350) and automatically save `mnist_explosion_dump.json`.

### 2. Visualize the Crash

Generate a "Failure Signature" plot from the crash dump.

```bash
python -m experiments.visualize_crash mnist_explosion_dump.json
```

**Outcome:** This generates `mnist_explosion_dump.png` (shown at the top of this README), revealing the per-layer breakdown of the failure.

##  System Architecture

The system is built on three decoupled components:

* **HookManager (The Eyes):** Registers forward/backward hooks to extract scalar norms from PyTorch layers.
* **RingBuffer (The Memory):** A `collections.deque` buffer that rotates out old steps to maintain constant memory usage.
* **TriggerEngine (The Brain):** Stateless logic that evaluates the current step against failure conditions (EMA spikes, thresholds).

##  Requirements

```text
torch
torchvision
matplotlib
numpy
```
