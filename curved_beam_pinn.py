"""
Physics-Informed Neural Network (PINN) for a simply supported beam
under a uniformly distributed load (UDL), with a heuristic curvature
correction factor.

Important:
The governing equation implemented here is the Euler-Bernoulli straight-
beam equation. Curvature is represented only by the user-defined
empirical factor in q_eff; this is not a complete curved-beam formulation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Problem parameters
q = 1000.0       # UDL (N/m)
L = 4.0          # Beam length (m)
EI = 50000.0     # Flexural rigidity (N m^2)
kappa = 0.2      # Curvature parameter (1/m); 0 corresponds to no correction

N_col = 200
EPOCHS = 3000
lr = 1e-3
lam = 10.0

# Heuristic curvature adjustment used in this implementation.
curve_factor = 1.0 + 0.15 * kappa * L
q_eff = q * curve_factor


# 2. PINN model
class BeamPINN(nn.Module):
    """Map the beam coordinate x in [0, L] to predicted deflection w(x)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x / L)


# 3. Automatic differentiation
def grad(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def derivatives(model, x):
    w = model(x)
    dw = grad(w, x)
    d2w = grad(dw, x)
    d3w = grad(d2w, x)
    d4w = grad(d3w, x)
    return w, d2w, d4w


def main():
    # 4. Collocation and boundary points
    x_col = torch.linspace(0.0, L, N_col).view(-1, 1).requires_grad_(True)
    x0 = torch.tensor([[0.0]], requires_grad=True)
    xL = torch.tensor([[L]], requires_grad=True)

    # 5. Train
    model = BeamPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.8
    )
    losses = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        _, _, d4w_col = derivatives(model, x_col)
        physics_loss = torch.mean((d4w_col - q_eff / EI) ** 2)

        w0, d2w0, _ = derivatives(model, x0)
        wL, d2wL, _ = derivatives(model, xL)
        boundary_loss = (
            w0.square() + wL.square() + d2w0.square() + d2wL.square()
        ).mean()

        loss = physics_loss + lam * boundary_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:4d} | loss={loss.item():.6f} "
                f"| L_phys={physics_loss.item():.6f} "
                f"| L_BC={boundary_loss.item():.6f}"
            )

    # 6. Evaluate
    x_test = torch.linspace(0.0, L, 200).view(-1, 1)
    with torch.no_grad():
        w_pred = model(x_test).cpu().numpy().flatten()

    x_np = x_test.cpu().numpy().flatten()

    # Simply supported straight-beam analytical solution for the same
    # positive-deflection sign convention used in the PDE residual.
    w_analytical = (q / (24.0 * EI)) * x_np * (
        L**3 - 2.0 * L * x_np**2 + x_np**3
    )
    w_curvature_adjusted = w_analytical * curve_factor

    print(f"\nMax PINN deflection: {w_pred.max() * 1000:.4f} mm")
    print(
        "Max analytical (straight): "
        f"{w_analytical.max() * 1000:.4f} mm"
    )
    print(
        "Max curvature-adjusted analytical: "
        f"{w_curvature_adjusted.max() * 1000:.4f} mm"
    )
    print(
        "5qL^4/(384EI), straight beam: "
        f"{5 * q * L**4 / (384 * EI) * 1000:.4f} mm"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_np, w_pred * 1000, label="PINN", lw=2)
    axes[0].plot(
        x_np, w_analytical * 1000, label="Analytical (straight)", lw=1.5
    )
    axes[0].plot(
        x_np,
        w_curvature_adjusted * 1000,
        label="Curvature-adjusted estimate",
        lw=1.5,
        ls="--",
    )
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("Deflection w (mm)")
    axes[0].set_title("Deflection profile")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(losses, lw=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_title("Training loss convergence")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("pinn_curved_beam.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
