import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from observer import HookManager, RingBuffer, TriggerEngine
import os

def run_experiment():
    # 1. Setup Data (MNIST)
    print("Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Define a Real Model (ConvNet)
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Attach Neural Observer
    print("Attaching Observer...")
    observer = HookManager(log_every=5) # Log every 5 steps to save space
    observer.attach(model)
    
    # 4. Setup Training with FATAL param
    # A Learning Rate of 100.0 is guaranteed to cause an explosion
    optimizer = optim.SGD(model.parameters(), lr=100.0) 
    criterion = nn.CrossEntropyLoss()
    
    buffer = RingBuffer(capacity=100)
    # Triggers: 10x spike or Grad Norm > 500
    triggers = TriggerEngine(spike_factor=10.0, grad_threshold=500.0) 

    print("\n--- Starting 'Doomed' Training Run ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for step, (data, target) in enumerate(train_loader):
        observer.reset_step()
        data, target = data.to(device), target.to(device)
        
        # Forward
        output = model(data)
        loss = criterion(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Observer Logic ---
        # 1. Capture
        buffer.add(step, observer.grad_norms, observer.act_norms, loss.item())
        
        # 2. Check (Skip first 10 steps warmup)
        if step > 10:
            fail_reason = triggers.check(loss.item(), observer.grad_norms)
            
            if fail_reason:
                print(f"\n CRASH DETECTED at Step {step}!")
                print(f"Reason: {fail_reason}")
                
                filename = "mnist_explosion_dump.json"
                buffer.save_json(filename, trigger_reason=fail_reason)
                print(f"Artifact saved to {filename}")
                break
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {loss.item():.4f} (Model holding together...)")

if __name__ == "__main__":
    run_experiment()