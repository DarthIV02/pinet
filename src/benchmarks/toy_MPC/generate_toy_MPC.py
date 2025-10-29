"""Generate toy MPC problem data (PyTorch version)."""

import pathlib
import cvxpy as cp
import numpy as np
import torch
from tqdm import tqdm

# Enable saving the dataset
SAVE_RESULTS = True

# ===================== PARAMETERS =========================
base_dim = 2  # Planning in 2D
T = 20        # Horizon
dimx = base_dim * (T + 1)
dimu = base_dim * T
alpha = 1.0   # Objective tradeoff

# Single instance
x0 = torch.tensor([0.0, 4.0])
xhat = torch.tensor([3.0, -12.0])

# Decision variable [x_0, ..., x_T, u_0, ..., u_{T-1}]
# Equality constraints
A_x = torch.diag(torch.ones(T + 1)) - torch.diag(torch.ones(T), diagonal=-1)
A_x = torch.kron(A_x, torch.eye(base_dim))
A_u = torch.diag(torch.ones(T), diagonal=-1)[:, :-1]
A_u = torch.kron(A_u, torch.eye(base_dim))
A = torch.cat((A_x, A_u), dim=1)

# Inequality constraints
boundx = 10
boundu = 1
lbx = -boundx * torch.ones((dimx, 1))
ubx = boundx * torch.ones((dimx, 1))
lbu = -boundu * torch.ones((dimu, 1))
ubu = boundu * torch.ones((dimu, 1))

# ===================== SINGLE SOLVE ======================
z = cp.Variable(dimx + dimu)
xinit = cp.Parameter(base_dim)
constraints = [
    A.numpy() @ z == np.hstack([xinit.value if xinit.value is not None else np.zeros(base_dim), np.zeros(dimx - base_dim)]),
    z[:dimx] >= lbx.numpy(),
    z[:dimx] <= ubx.numpy(),
    z[dimx:] >= lbu.numpy(),
    z[dimx:] <= ubu.numpy(),
]
objective = cp.Minimize(
    cp.sum_squares(z[:dimx] - np.tile(xhat.numpy(), T + 1)) + alpha * cp.sum_squares(z[dimx:])
)
problem = cp.Problem(objective, constraints)

xinit.value = x0.numpy()
problem.solve()

# ===================== DATASET GENERATION ======================
SEED = 42
NUM_EXAMPLES = 10000
torch.manual_seed(SEED)

# Random initial states
x0set = (2 * boundx) * torch.rand(NUM_EXAMPLES, base_dim) - boundx
objectives = torch.zeros(NUM_EXAMPLES)
Ystar = torch.zeros(NUM_EXAMPLES, dimx + dimu)

print(f"Solving {NUM_EXAMPLES} problem instances")
for idx in tqdm(range(NUM_EXAMPLES)):
    xinit.value = x0set[idx].numpy()
    problem.solve()
    objectives[idx] = problem.value
    Ystar[idx, :] = torch.tensor(z.value).reshape(-1)

# ===================== RESHAPE FOR DATASET ======================
As = A.reshape(1, A.shape[0], A.shape[1])
lbxs = lbx.reshape(1, -1, 1)
ubxs = ubx.reshape(1, -1, 1)
lbus = lbu.reshape(1, -1, 1)
ubus = ubu.reshape(1, -1, 1)
x0sets = x0set.reshape(NUM_EXAMPLES, base_dim, 1)
xhat = xhat.reshape(base_dim, 1)

# ===================== SAVE DATA ==========================
if SAVE_RESULTS:
    datasets_path = pathlib.Path(__file__).parent.resolve() / "datasets"
    datasets_path.mkdir(parents=True, exist_ok=True)
    filename = f"toy_MPC_seed{SEED}_examples{NUM_EXAMPLES}.npz"
    path = datasets_path / filename
    np.savez(
        path,
        As=As.numpy(),
        lbxs=lbxs.numpy(),
        ubxs=ubxs.numpy(),
        lbus=lbus.numpy(),
        ubus=ubus.numpy(),
        x0sets=x0sets.numpy(),
        xhat=xhat.numpy(),
        objectives=objectives.numpy(),
        Ystar=Ystar.numpy(),
        T=T,
        base_dim=base_dim,
        alpha=alpha,
    )