import math

class TriggerEngine:
    def __init__(self, spike_factor=2.5, grad_threshold=10.0, ema_alpha=0.1):
        """
        spike_factor: How many times larger than EMA counts as a spike? (e.g. 2.5x)
        grad_threshold: Max allowed L2 norm for any layer gradient.
        ema_alpha: Smoothing factor for loss history (0.1 = keeps ~10 steps memory).
        """
        self.spike_factor = spike_factor
        self.grad_threshold = grad_threshold
        self.ema_alpha = ema_alpha
        self.ema_loss = None

    def check(self, loss, grad_norms):
        """
        Returns a string reason if triggered, else None.
        """
        # 1. Check for NaNs or Infs (Immediate Fail)
        if loss is None or not math.isfinite(loss):
            return "loss_is_nan_or_inf"

        # 2. Check Gradient Explosion
        # We look for the *worst* layer in this step
        if grad_norms:
            max_norm = max(grad_norms.values())
            if max_norm > self.grad_threshold:
                return f"grad_explosion (max_norm={max_norm:.2f} > {self.grad_threshold})"

        # 3. Check Loss Spike
        # We need a history (EMA) to define what a "spike" is
        trigger = None
        if self.ema_loss is not None:
            # Add a small epsilon (1e-6) to handle near-zero loss stability
            threshold = (abs(self.ema_loss) + 1e-6) * self.spike_factor
            # If loss jumps > 2.5x the average, trigger
            if loss > threshold:
                trigger = f"loss_spike (current={loss:.4f}, avg={self.ema_loss:.4f})"
        
        # Update Internal State (EMA)
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = (self.ema_alpha * loss) + ((1 - self.ema_alpha) * self.ema_loss)

        return trigger