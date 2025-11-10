import numpy as np
import ot
from scipy.sparse import coo_matrix
import torch
import time

n = 50000
dim = 1  # Higher dimensional space
reg = 0.1
k = 100  # Increased - number of top flows to keep per row

rng = np.random.RandomState(42)
X_s = rng.randn(n, dim)
X_t = rng.randn(n, dim) + 2.0

a = ot.utils.unif(n)
b = ot.utils.unif(n)

print("Running Sinkhorn with geomloss...")
cost_sinkhorn, log = ot.bregman.empirical_sinkhorn2_geomloss(
    X_s, X_t, reg, a=a, b=b, scaling=0.95, log=True, backend="online"
)
print(f"Sinkhorn cost (with regularization): {cost_sinkhorn:.6f}")

# Extract sparse structure from lazy plan using PyKeOps argKmin
print(f"Extracting top-{k} flows from transport plan using PyKeOps...")

# Convert to torch tensors for PyKeOps operations
X_s_torch = torch.from_numpy(X_s).float().cuda()
X_t_torch = torch.from_numpy(X_t).float().cuda()
f_torch = torch.from_numpy(log['f']).float().cuda()
g_torch = torch.from_numpy(log['g']).float().cuda()
a_torch = torch.from_numpy(a).float().cuda()
b_torch = torch.from_numpy(b).float().cuda()

# Build the transport plan values using the dual potentials
# T = exp((f + g^T - C) / reg) * (a * b^T)
# We want to find the top-k largest values per row

from pykeops.torch import LazyTensor

# Create LazyTensors for efficient computation
x_i = LazyTensor(X_s_torch[:, None, :])  # (n, 1, d)
y_j = LazyTensor(X_t_torch[None, :, :])  # (1, m, d)
f_i = LazyTensor(f_torch[:, None, None])  # (n, 1, 1)
g_j = LazyTensor(g_torch[None, :, None])  # (1, m, 1)
a_i = LazyTensor(a_torch[:, None, None])  # (n, 1, 1)
b_j = LazyTensor(b_torch[None, :, None])  # (1, m, 1)

# Compute cost matrix C = ||x - y||^2 / 2
C_ij = ((x_i - y_j) ** 2).sum(-1) / 2  # (n, m, 1)

# Compute transport plan T = exp((f + g - C) / reg) * (a * b)
# For sqeuclidean metric, geomloss uses blur^2 = reg/2 (not reg!)
blur = torch.sqrt(torch.tensor(reg / 2, dtype=torch.float32)).cuda()
T_ij = ((f_i + g_j - C_ij) / (blur ** 2)).exp() * a_i * b_j  # (n, m, 1)

# Compute the pure transport cost of Sinkhorn's plan (without regularization)
print("\nComputing pure transport cost of Sinkhorn's solution...")
# Sum T_ij * C_ij over all pairs to get <γ, C>
transport_cost_sinkhorn = (T_ij * C_ij).sum(dim=1).sum().item()
print(f"  Sinkhorn's pure transport cost <γ,C>: {transport_cost_sinkhorn:.6f}")
print(f"  Sinkhorn's total cost (w/ reg): {cost_sinkhorn:.6f}")
print(f"  Regularization penalty: {cost_sinkhorn - transport_cost_sinkhorn:.6f}")

print("  Finding top-k flows per row...")
# For each row, find the k largest transport values
# argKmin with negative values gives us argKmax
indices_rows = (-T_ij).argKmin(k, dim=1).cpu().numpy().squeeze()  # (n, k)

print("  Finding top-k flows per column...")
# For each column, find the k largest transport values
indices_cols = (-T_ij).argKmin(k, dim=0).cpu().numpy().squeeze()  # (n, k) - k values for each of n columns

# Build sparse edge list
rows_sparse = []
cols_sparse = []

print("  Building edge list from row-wise top-k...")
for i in range(n):
    if i % 5000 == 0:
        print(f"    Processing row {i}/{n}...")
    for j in indices_rows[i]:
        rows_sparse.append(i)
        cols_sparse.append(int(j))

print("  Building edge list from column-wise top-k...")
for j in range(n):
    if j % 5000 == 0:
        print(f"    Processing column {j}/{n}...")
    for i in indices_cols[j]:  # Fixed: indices_cols[j] not indices_cols[:, j]
        rows_sparse.append(int(i))
        cols_sparse.append(j)

rows_sparse = np.array(rows_sparse)
cols_sparse = np.array(cols_sparse)

print(f"  Found {len(rows_sparse)} edges from top-k flows (before deduplication)")

# Remove duplicates and compute costs
print("  Removing duplicate edges...")
unique_edges = np.unique(np.column_stack([rows_sparse, cols_sparse]), axis=0)
rows_final = unique_edges[:, 0].astype(int)
cols_final = unique_edges[:, 1].astype(int)

print(f"  Computing costs for {len(rows_final)} unique edges...")
# Compute squared Euclidean distances for the sparse edges (divide by 2 for consistency)
data_final = np.sum((X_s[rows_final] - X_t[cols_final])**2, axis=1) / 2

C_sparse = coo_matrix((data_final, (rows_final, cols_final)), shape=(n, n))

print(f"Sparse graph has {len(data_final)} edges")

# Run sparse EMD
print("Running sparse EMD...")
print(f"  (This may take a while with {len(data_final):,} edges...)")
start_time = time.time()
G_sparse, log_sparse = ot.emd(a, b, C_sparse, numItermax=10000000, log=True)
emd_time = time.time() - start_time
cost_sparse = log_sparse["cost"]
print(f"Sparse EMD cost: {cost_sparse:.6f}")
print(f"EMD time: {emd_time:.2f} seconds")
print(f"EMD iterations: {log_sparse.get('numItermax', 'N/A')}")
print(f"EMD warning: {log_sparse.get('warning', 'None')}")

print(f"\n{'='*60}")
print("COST COMPARISON:")
print(f"{'='*60}")
print(f"Sinkhorn total (transport + reg): {cost_sinkhorn:.6f}")
print(f"  - Pure transport cost <γ,C>:    {transport_cost_sinkhorn:.6f}")
print(f"  - Regularization penalty:        {cost_sinkhorn - transport_cost_sinkhorn:.6f}")
print(f"\nSparse EMD (pure transport):       {cost_sparse:.6f}")
print(f"\nComparison:")
print(f"  Sinkhorn vs Sparse EMD (total):  {abs(cost_sinkhorn - cost_sparse):.6f} ({abs(cost_sinkhorn - cost_sparse) / cost_sinkhorn * 100:.2f}%)")
print(f"  Sinkhorn vs Sparse EMD (pure):   {abs(transport_cost_sinkhorn - cost_sparse):.6f} ({abs(transport_cost_sinkhorn - cost_sparse) / transport_cost_sinkhorn * 100:.2f}%)")
print(f"{'='*60}")

# Additional analysis
print(f"\nGraph statistics:")
print(f"  Total possible edges: {n * n:,}")
print(f"  Sparse edges: {len(data_final):,}")
print(f"  Sparsity: {len(data_final) / (n * n) * 100:.4f}%")
print(f"  Edges per row (avg): {len(data_final) / n:.1f}")
