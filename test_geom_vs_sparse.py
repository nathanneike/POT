import numpy as np
import ot
from scipy.sparse import coo_matrix

n = 10000
reg = 0.1
k = 20

rng = np.random.RandomState(42)
X_s = rng.randn(n, 2)
X_t = rng.randn(n, 2) + 2.0

a = ot.utils.unif(n)
b = ot.utils.unif(n)

print("Running Sinkhorn with geomloss...")
cost_sinkhorn, log = ot.bregman.empirical_sinkhorn2_geomloss(
    X_s, X_t, reg, a=a, b=b, scaling=0.9, log=True
)
G_sinkhorn = log["lazy_plan"][:]
print(f"Sinkhorn cost: {cost_sinkhorn:.6f}")

rows_sparse = []
cols_sparse = []
data_sparse = []

for i in range(n):
    row_vals = G_sinkhorn[i, :]
    top_k_cols = np.argpartition(row_vals, -k)[-k:]
    for j in top_k_cols:
        if row_vals[j] > 0:
            rows_sparse.append(i)
            cols_sparse.append(j)
            data_sparse.append(1.0)

for j in range(n):
    col_vals = G_sinkhorn[:, j]
    top_k_rows = np.argpartition(col_vals, -k)[-k:]
    for i in top_k_rows:
        if col_vals[i] > 0:
            rows_sparse.append(i)
            cols_sparse.append(j)
            data_sparse.append(1.0)

C = ot.dist(X_s, X_t)
rows_sparse = np.array(rows_sparse)
cols_sparse = np.array(cols_sparse)
unique_edges = np.unique(np.column_stack([rows_sparse, cols_sparse]), axis=0)
rows_final = unique_edges[:, 0]
cols_final = unique_edges[:, 1]
data_final = C[rows_final, cols_final]

C_sparse = coo_matrix((data_final, (rows_final, cols_final)), shape=(n, n))

print(f"Sparse graph has {len(data_final)} edges")

# Run sparse EMD
print("Running sparse EMD...")
G_sparse, log_sparse = ot.emd(a, b, C_sparse, numItermax=500000, log=True)
cost_sparse = log_sparse["cost"]
print(f"Sparse EMD cost: {cost_sparse:.6f}")

print(f"\nCost difference: {abs(cost_sinkhorn - cost_sparse):.6f}")
print(
    f"Relative difference: {abs(cost_sinkhorn - cost_sparse) / cost_sinkhorn * 100:.2f}%"
)
