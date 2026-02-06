import json
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def visualize(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    print(f"Loading crash report: {json_path}")
    with open(json_path, 'r') as f:
        dump = json.load(f)

    history = dump['history']
    metadata = dump['metadata']
    print(f"Trigger Reason: {metadata['trigger_reason']}")
    
    steps = [h['step'] for h in history]
    loss = [h['loss'] for h in history]
    
    # 1. Identify all layers present in the dump
    all_keys = set()
    for h in history:
        all_keys.update(h['grad_norms'].keys())
    layer_names = sorted(list(all_keys))
    
    # 2. Extract gradient data (aligned with steps)
    # We fill missing steps with np.nan
    layer_grads = {name: [np.nan] * len(steps) for name in layer_names}
    for i, h in enumerate(history):
        norms = h['grad_norms']
        for name in layer_names:
            if name in norms:
                layer_grads[name][i] = norms[name]

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- PLOT 1: LOSS (Log Scale) ---
    # We use 'symlog' or 'log' to handle the massive explosion while keeping small values visible
    ax1.plot(steps, loss, color='red', linewidth=2, label='Loss')
    ax1.set_ylabel('Training Loss (Log Scale)')
    ax1.set_yscale('log')  # <--- THIS FIXES THE "ZERO" LOOK
    ax1.set_title(f"Crash Analysis: {metadata['trigger_reason']}")
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    # --- PLOT 2: GRADIENTS (Connected Lines) ---
    has_plotted = False
    for name, grads in layer_grads.items():
        # Clean data: Remove NaNs so matplotlib connects the lines
        valid_indices = [i for i, g in enumerate(grads) if not np.isnan(g)]
        
        if not valid_indices:
            continue
            
        valid_steps = [steps[i] for i in valid_indices]
        valid_grads = [grads[i] for i in valid_indices]
        
        # Only plot if the layer actually explodes or has activity
        if np.max(valid_grads) > 0.0: 
            # 'o-' draws markers AND lines
            ax2.plot(valid_steps, valid_grads, 'o-', markersize=4, linewidth=1.5, label=name, alpha=0.8)
            has_plotted = True
            
    if has_plotted:
        ax2.set_yscale('log')
        ax2.set_ylabel('Gradient Norm (L2)')
        ax2.set_xlabel('Training Step')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax2.text(0.5, 0.5, "No Gradient Data Found", ha='center', transform=ax2.transAxes)

    plt.tight_layout()
    
    output_img = json_path.replace('.json', '.png')
    plt.savefig(output_img)
    print(f" Visualization saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    target_file = "neural-observer\mnist_explosion_dump.json"
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    visualize(target_file)