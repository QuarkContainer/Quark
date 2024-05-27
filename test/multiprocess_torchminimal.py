import ray
import math
import time
import torch

# ray.init(num_gpus=1)
ray.init()
dtype = torch.float

# @ray.remote(num_gpus=0.5)

# @ray.remote(num_gpus=0.5)
@ray.remote()
def train_model(model_id):
    # Set GPU device
    # import torch(registerFatbin)
    #device = torch.device("cuda:0")
    device = torch.device("cpu")

    # Instantiate your PyTorch model
    torch.manual_seed(0)

    # Create random input and output data
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # Randomly initialize weights
    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    print(f'a = {a}, b = {b}, c = {c}, d = {d}')
    learning_rate = 1e-6
    for t in range(3):
        # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        time.sleep(1)

    return a.item()

# Launch two training processes
model1_id = train_model.remote(1)
model2_id = train_model.remote(2)

# Retrieve trained models
model1 = ray.get(model1_id)
model2 = ray.get(model2_id)