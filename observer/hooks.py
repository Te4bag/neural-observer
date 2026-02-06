import torch
import torch.nn as nn

class HookManager:
    def __init__(self, verbose = False, log_every=1):
        self.verbose = False
        self.log_every = log_every
        self.step_count = 0

        # Current step storage
        self.grad_norms = {}
        self.act_norms = {}
        self.handles = []

    def _should_log(self):
        # Only log if we match the frequency
        return (self.step_count % self.log_every) == 0

    def _get_act_hook(self, name):
        def hook(model, input, output):
            if not self._should_log():
                return
            # Extract tensor from possible tuple outputs
            tensor = output[0] if isinstance(input, tuple) else output

            if isinstance(tensor, torch.Tensor):
                # .item() ensures we store a scalar, not a GPU tensor
                norm = tensor.norm().detach().cpu().item()
                # Stores latest activation norm per step (Overwrites if layer is called twice in one step)
                self.act_norms[name] = norm
                if self.verbose: print(f"[Activation] {name}: {norm:.4f}")
        return hook
    
    def _get_grad_hook(self, name):
        def hook(module, grad_input, grad_output):
            if not self._should_log():
                return
            
            # SAFETY CRITICAL: Handle empty/None gradients
            if not grad_output:
                return
            
            g = grad_output[0]
            if g is None or not torch.is_tensor(g):
                return
            
            norm = g.norm().detach().cpu().item()
            self.grad_norms[name] = norm

            if self.verbose: print(f"[Gradient] {name}: {norm:.4f}")

        return hook
    
    def attach(self, model):
        print(f"Attaching hooks to {model.__class__.__name__}...")
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Embedding)):
                # Forward
                self.handles.append(module.register_forward_hook(self._get_act_hook(name)))
                # Backward
                self.handles.append(module.register_full_backward_hook(self._get_grad_hook(name)))
                count +=1
        print(f"Instrumented {count} layers.")

    def reset_step(self):
        """Call this at the START of each training step, before forward()"""
        self.act_norms.clear()
        self.grad_norms.clear()
        self.step_count += 1

    def close(self):
        for h in self.handles:
            h.remove()
        print("Hooks detached.")