from collections import deque
import json
import time

class RingBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, step, grad_norms, act_norms, loss=None):
        """
        Saves a snapshot of the training step.
        """
        record = {
            'step': step,
            'timestamp': time.time(),
            'loss': float(loss) if loss is not None else None,
            # We don't store capacity here to save space
            'grad_norms': grad_norms.copy(),
            'act_norms': act_norms.copy()
        }
        self.buffer.append(record)

    def to_list(self):
        return list(self.buffer)

    def save_json(self, filepath, trigger_reason="manual_dump"):
        """
        Dumps the buffer to disk with high-level metadata.
        """
        # STRUCTURE: Metadata at top, data below
        export_data = {
            "metadata": {
                "capacity": self.capacity,
                "trigger_reason": trigger_reason,
                "dump_timestamp": time.time(),
                "captured_steps": len(self.buffer)
            },
            "history": list(self.buffer)
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Crash dump saved to {filepath} (Reason: {trigger_reason})")
        except Exception as e:
            print(f"Failed to save crash dump: {e}")