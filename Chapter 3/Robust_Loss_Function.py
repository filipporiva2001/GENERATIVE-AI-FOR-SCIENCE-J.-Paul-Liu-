#1.1 Robust Loss Functions
##Scientific data often contains outliers. Huber loss is less sensitive to outliers than Mean Squared Error (MSE).


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------------------------
# Huber loss (PyTorch)
# ---------------------------
def huber_loss(pred, target, delta=1.0):
    """Huber loss: less sensitive to outliers than MSE"""
    error = pred - target
    is_small = torch.abs(error) <= delta
    squared = 0.5 * error**2
    linear = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small, squared, linear).mean()

# ---------------------------
# Part 1: Loss curves vs error
# ---------------------------
delta = 1.0
e = np.linspace(-5, 5, 400)

mse_curve = 0.5 * e**2
mae_curve = np.abs(e)
huber_curve = np.where(
    np.abs(e) <= delta,
    0.5 * e**2,
    delta * (np.abs(e) - 0.5 * delta)
)

plt.figure()
plt.plot(e, mse_curve, label="MSE (0.5 * e^2)")
plt.plot(e, mae_curve, label="MAE (|e|)")
plt.plot(e, huber_curve, label=f"Huber (δ={delta})")
plt.xlabel("Error (e)")
plt.ylabel("Loss")
plt.title("Comparison of MSE, MAE, and Huber Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Part 2: Numeric comparison on noisy data with outliers
# ---------------------------
torch.manual_seed(42)
np.random.seed(42)

predictions = torch.randn(100)
targets = torch.randn(100)

# Add some outliers
targets[::10] += torch.randn(10) * 5  # every 10th point is an outlier

mse_loss = nn.MSELoss()(predictions, targets)
mae_loss = nn.L1Loss()(predictions, targets)
huber = huber_loss(predictions, targets, delta=delta)

print(f"MSE Loss:   {mse_loss.item():.4f}")
print(f"MAE Loss:   {mae_loss.item():.4f}")
print(f"Huber Loss: {huber.item():.4f}")
print("\n✓ Huber loss is more robust to outliers than MSE,")
print("  and smoother than MAE around zero.")
