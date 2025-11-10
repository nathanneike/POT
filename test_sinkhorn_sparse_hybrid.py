"""
Hybrid Geomloss + Sparse Exact EMD Solver

1. Generate clustered point clouds with natural geometric sparsity
2. Run GPU-accelerated Geomloss Sinkhorn on the full point clouds
3. Extract sparse support from Sinkhorn solution (top-k per row/column)
4. Solve exact sparse EMD on the extracted support

"""

import numpy as np
import torch
import ot
from scipy.sparse import coo_matrix
from pykeops.torch import LazyTensor
import time
import sys

n_points = 100000          # Number of points in each cloud
n_clusters = 200          # Number of clusters
k_extract = 100           # Top-k flows to extract per row/column
reg = 0.05                # Regularization parameter
dim = 2                   # Dimension of space
cluster_std = 0.5         # Standard deviation within clusters
separation = 3.0          # Separation between cluster centers
random_seed = 42          # Random seed
device = "cuda"           # Device to use
compute_dense_baseline = False  # Compute dense EMD baseline (slow for n>2000)
sinkhorn_scaling = 0.95   # Epsilon-scaling parameter (0.95-0.99)
sinkhorn_backend = "online"  # Geomloss backend: "online", "multiscale", "tensorized"
emd_max_iter = 10000000   # Max iterations for sparse EMD
num_threads = 1           # Number of threads for sparse EMD solver



def generate_clustered_point_clouds(
    n_points: int,
    n_clusters: int,
    dim: int,
    cluster_std: float,
    separation: float,
    random_seed: int
):
    """Generate two clustered point clouds with natural sparsity."""
    rng = np.random.RandomState(random_seed)
    points_per_cluster = n_points // n_clusters

    X_s_list = []
    X_t_list = []

    grid_size = int(np.ceil(np.sqrt(n_clusters)))

    for i in range(n_clusters):
        grid_x = (i % grid_size) * separation
        grid_y = (i // grid_size) * separation

        if dim == 2:
            center_s = np.array([grid_x, grid_y])
            center_t = np.array([grid_x + 0.5, grid_y + 0.3])
        else:
            center_s = np.zeros(dim)
            center_s[0] = grid_x
            center_s[1] = grid_y
            center_t = center_s + rng.randn(dim) * 0.5

        points_s = center_s + rng.randn(points_per_cluster, dim) * cluster_std
        points_t = center_t + rng.randn(points_per_cluster, dim) * cluster_std

        X_s_list.append(points_s)
        X_t_list.append(points_t)

    remainder = n_points - points_per_cluster * n_clusters
    if remainder > 0:
        center_s = np.zeros(dim)
        center_t = rng.randn(dim) * 0.5
        points_s = center_s + rng.randn(remainder, dim) * cluster_std
        points_t = center_t + rng.randn(remainder, dim) * cluster_std
        X_s_list.append(points_s)
        X_t_list.append(points_t)

    X_s = np.vstack(X_s_list).astype(np.float32)
    X_t = np.vstack(X_t_list).astype(np.float32)

    return X_s, X_t


def solve_geomloss_sinkhorn(
    X_s: np.ndarray,
    X_t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    reg: float,
    scaling: float = 0.95,
    backend: str = "online",
    device: str = "cuda"
):
    """Solve Sinkhorn OT using geomloss with GPU and lazy tensors."""
    print(f"\nRunning Geomloss Sinkhorn (reg={reg}, backend={backend})...")
    start = time.time()

    X_s_torch = torch.from_numpy(X_s).float().to(device)
    X_t_torch = torch.from_numpy(X_t).float().to(device)
    a_torch = torch.from_numpy(a).float().to(device)
    b_torch = torch.from_numpy(b).float().to(device)

    # Use geomloss
    total_cost, log = ot.bregman.empirical_sinkhorn2_geomloss(
        X_s_torch, X_t_torch, reg,
        a=a_torch, b=b_torch,
        verbose=False,
        scaling=scaling,
        log=True,
        backend=backend
    )

    elapsed = time.time() - start

    if isinstance(total_cost, torch.Tensor):
        total_cost_np = total_cost.cpu().detach().numpy()
    else:
        total_cost_np = total_cost

    f_torch = log['f']
    g_torch = log['g']

    if isinstance(f_torch, torch.Tensor):
        f_np = f_torch.cpu().detach().numpy()
    else:
        f_np = f_torch
    if isinstance(g_torch, torch.Tensor):
        g_np = g_torch.cpu().detach().numpy()
    else:
        g_np = g_torch

    print("  Computing pure transport cost (without regularization)...")

    x_i = LazyTensor(X_s_torch[:, None, :])
    y_j = LazyTensor(X_t_torch[None, :, :])
    f_i = LazyTensor(f_torch[:, None, None])
    g_j = LazyTensor(g_torch[None, :, None])
    a_i = LazyTensor(a_torch[:, None, None])
    b_j = LazyTensor(b_torch[None, :, None])

    # Cost matrix C = ||x - y||^2 / 2
    C_ij = ((x_i - y_j) ** 2).sum(-1) / 2

    # For sqeuclidean, geomloss uses blur^2 = reg/2
    blur_squared = torch.tensor(reg / 2, dtype=torch.float32, device=device)
    T_ij = ((f_i + g_j - C_ij) / blur_squared).exp() * a_i * b_j

    pure_cost = (T_ij * C_ij).sum(dim=1).sum().item()

    print(f"  Completed in {elapsed:.3f}s")
    print(f"    Total Sinkhorn cost (w/ reg): {float(total_cost_np):.6f}")
    print(f"    Pure transport cost <γ,C>:    {pure_cost:.6f}")
    print(f"    Regularization penalty:        {float(total_cost_np) - pure_cost:.6f}")

    return {
        'f': f_np,
        'g': g_np,
        'time': elapsed,
        'cost': pure_cost,
        'cost_with_reg': float(total_cost_np)
    }


def extract_topk_support(
    X_s: np.ndarray,
    X_t: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    reg: float,
    k: int,
    device: str = "cuda"
):
    """Extract top-k support from Sinkhorn solution using dual potentials."""
    print(f"\nExtracting top-{k} support from Sinkhorn solution...")
    total_start = time.time()
    n = X_s.shape[0]
    m = X_t.shape[0]

    print("  [TIMING] Transferring data to GPU...")
    transfer_start = time.time()
    X_s_torch = torch.from_numpy(X_s).float().to(device)
    X_t_torch = torch.from_numpy(X_t).float().to(device)
    f_torch = torch.from_numpy(f).float().to(device)
    g_torch = torch.from_numpy(g).float().to(device)
    transfer_time = time.time() - transfer_start
    print(f"    CPU→GPU transfer: {transfer_time:.3f}s")

    print("  [TIMING] Building lazy tensors...")
    lazy_start = time.time()
    x_i = LazyTensor(X_s_torch[:, None, :])
    y_j = LazyTensor(X_t_torch[None, :, :])
    f_i = LazyTensor(f_torch[:, None, None])
    g_j = LazyTensor(g_torch[None, :, None])

    # Cost matrix C = ||x - y||^2 / 2
    C_ij = ((x_i - y_j) ** 2).sum(-1) / 2

    T_ij = ((f_i + g_j - C_ij) / reg).exp()
    lazy_time = time.time() - lazy_start
    print(f"    Lazy tensor setup: {lazy_time:.3f}s")

    print(f"  [TIMING] Extracting top-{k} per row...")
    argkmin_start = time.time()
    indices_rows = (-T_ij).argKmin(k, dim=1).cpu().numpy().squeeze()  # (n, k)
    argkmin_row_time = time.time() - argkmin_start
    print(f"    argKmin (rows) + GPU→CPU: {argkmin_row_time:.3f}s")
    print(f"    Shape: {indices_rows.shape}, dtype: {indices_rows.dtype}")

    print(f"  [TIMING] Extracting top-{k} per column...")
    argkmin_start = time.time()
    indices_cols = (-T_ij).argKmin(k, dim=0).cpu().numpy().squeeze()  # (m, k)
    argkmin_col_time = time.time() - argkmin_start
    print(f"    argKmin (cols) + GPU→CPU: {argkmin_col_time:.3f}s")
    print(f"    Shape: {indices_cols.shape}, dtype: {indices_cols.dtype}")

    print(f"\n  [TIMING SUMMARY - Support Extraction]")
    print(f"    CPU→GPU transfer:     {transfer_time:.3f}s")
    print(f"    Lazy tensor setup:    {lazy_time:.3f}s")
    print(f"    argKmin rows (GPU):   {argkmin_row_time:.3f}s")
    print(f"    argKmin cols (GPU):   {argkmin_col_time:.3f}s")
    gpu_total = transfer_time + lazy_time + argkmin_row_time + argkmin_col_time
    print(f"    GPU operations total: {gpu_total:.3f}s")

    # Free GPU memory
    del T_ij
    torch.cuda.empty_cache()

    print("\n  [TIMING] Building sparse matrix in batches (ultra memory efficient)...")
    sparse_start = time.time()

    from scipy.sparse import csr_matrix

    # Build sparse matrices in batches and merge
    # Each batch creates a small sparse matrix, then we merge them
    print("  Processing row-wise edges in batches...")

    batch_size = 50000  # Process 50k rows at a time
    sparse_batches = []

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        if i_start % 200000 == 0:
            print(f"    Batch rows {i_start:,}/{n:,}...")

        # Build edges for this batch only
        batch_rows = []
        batch_cols = []
        batch_data = []

        for local_i, i in enumerate(range(i_start, i_end)):
            for j in indices_rows[i]:
                batch_rows.append(i)
                batch_cols.append(int(j))
                batch_data.append(1.0)

        # Create small sparse matrix for this batch
        batch_sparse = csr_matrix(
            (batch_data, (batch_rows, batch_cols)),
            shape=(n, m),
            dtype=np.float32
        )
        sparse_batches.append(batch_sparse)

        # Clean up
        del batch_rows, batch_cols, batch_data

    print(f"  Merging {len(sparse_batches)} row batches...")
    # Sum all batches (automatically deduplicates and merges)
    rows_sparse = sum(sparse_batches)
    del sparse_batches

    print("  Processing column-wise edges in batches...")
    sparse_batches = []

    for j_start in range(0, m, batch_size):
        j_end = min(j_start + batch_size, m)
        if j_start % 200000 == 0:
            print(f"    Batch cols {j_start:,}/{m:,}...")

        batch_rows = []
        batch_cols = []
        batch_data = []

        for local_j, j in enumerate(range(j_start, j_end)):
            for i in indices_cols[j]:
                batch_rows.append(int(i))
                batch_cols.append(j)
                batch_data.append(1.0)

        batch_sparse = csr_matrix(
            (batch_data, (batch_rows, batch_cols)),
            shape=(n, m),
            dtype=np.float32
        )
        sparse_batches.append(batch_sparse)
        del batch_rows, batch_cols, batch_data

    print(f"  Merging {len(sparse_batches)} col batches...")
    cols_sparse = sum(sparse_batches)
    del sparse_batches

    # Free memory
    del indices_rows, indices_cols

    print("  Combining row and col edges...")
    # Add the two sparse matrices (automatically deduplicates)
    combined_sparse = rows_sparse + cols_sparse
    del rows_sparse, cols_sparse

    # Convert to COO for final edge list
    print("  Converting to COO format...")
    temp_sparse = combined_sparse.tocoo()
    del combined_sparse

    rows_final = temp_sparse.row
    cols_final = temp_sparse.col
    del temp_sparse

    print(f"  Computing costs for {len(rows_final):,} unique edges...")
    # Compute costs in chunks to avoid memory issues
    chunk_size = 100000
    num_chunks = (len(rows_final) + chunk_size - 1) // chunk_size

    costs_list = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(rows_final))
        chunk_rows = rows_final[start_idx:end_idx]
        chunk_cols = cols_final[start_idx:end_idx]
        chunk_costs = np.sum((X_s[chunk_rows] - X_t[chunk_cols])**2, axis=1) / 2
        costs_list.append(chunk_costs)

    costs = np.concatenate(costs_list)

    C_sparse = coo_matrix((costs, (rows_final, cols_final)), shape=(n, m))

    sparse_time = time.time() - sparse_start
    total_time = time.time() - total_start

    print(f"    Sparse matrix construction: {sparse_time:.3f}s")
    print(f"  [TIMING] Total extraction time: {total_time:.3f}s")
    print(f"    → GPU work: {gpu_total:.3f}s ({gpu_total/total_time*100:.1f}%)")
    print(f"    → CPU work: {sparse_time:.3f}s ({sparse_time/total_time*100:.1f}%)")

    sparsity_pct = len(costs) / (n * m) * 100
    print(f"\n  Sparse support: {len(costs):,} edges ({sparsity_pct:.4f}%), avg {len(costs) / n:.1f} per row")

    return C_sparse


def solve_sparse_emd(a: np.ndarray, b: np.ndarray, C_sparse, max_iter: int, num_threads: int = 1):
    """Solve exact sparse EMD using network simplex with multi-threading."""
    n_edges = len(C_sparse.data)
    print(f"\n{'='*60}")
    print(f"SPARSE EMD SOLVER (threads={num_threads})")
    print(f"{'='*60}")
    print(f"Graph: {n_edges:,} edges, {C_sparse.shape[0]:,} × {C_sparse.shape[1]:,} nodes")

    start = time.time()
    _, log = ot.emd(a, b, C_sparse, numItermax=max_iter, numThreads=num_threads, log=True)
    elapsed = time.time() - start

    cost = log['cost']
    iters = log.get('numIterations', 0)

    print(f"Status: {'⚠ WARNING' if log.get('warning') else '✓ SUCCESS'}")
    print(f"Cost:       {cost:.8f}")
    print(f"Time:       {elapsed:.3f}s")
    print(f"Iterations: {iters:,}")
    print(f"Speed:      {iters/elapsed:.0f} iters/sec")
    if n_edges > 0:
        print(f"Throughput: {n_edges/elapsed:,.0f} edges/sec")
    print(f"{'='*60}\n")

    if log.get('warning'):
        print(f"⚠ Warning: {log['warning']}\n")

    log['time'] = elapsed
    return log


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        global n_points, n_clusters, k_extract, reg
        n_points = int(sys.argv[1])
        if len(sys.argv) > 2:
            n_clusters = int(sys.argv[2])
        if len(sys.argv) > 3:
            k_extract = int(sys.argv[3])
        if len(sys.argv) > 4:
            reg = float(sys.argv[4])

    _compute_dense = compute_dense_baseline and (n_points <= 2000)

    print(f"Config: {n_points} points, {n_clusters} clusters")
    print(f"        reg={reg}, k={k_extract}")
    print(f"        Solver: Geomloss (GPU + lazy tensors)")

    X_s, X_t = generate_clustered_point_clouds(
        n_points, n_clusters, dim, cluster_std, separation, random_seed
    )
    a = ot.utils.unif(n_points)
    b = ot.utils.unif(n_points)
    print(f"Generated {n_points} points in {n_clusters} clusters")

    dense_emd_cost = None
    dense_time = 0
    if _compute_dense:
        M_full = ot.dist(X_s, X_t, metric='sqeuclidean') / 2
        start = time.time()
        dense_emd_cost = ot.emd2(a, b, M_full, numItermax=10000000)
        dense_time = time.time() - start
        print(f"  Dense EMD: {dense_emd_cost:.6f} in {dense_time:.3f}s")

    sinkhorn_log = solve_geomloss_sinkhorn(
        X_s, X_t, a, b, reg,
        scaling=sinkhorn_scaling,
        backend=sinkhorn_backend,
        device=device
    )

    C_sparse = extract_topk_support(
        X_s, X_t,
        sinkhorn_log['f'], sinkhorn_log['g'],
        reg, k_extract, device
    )

    sparse_emd_log = solve_sparse_emd(a, b, C_sparse, emd_max_iter, num_threads)

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Problem size: {n_points:,} points, {n_clusters} clusters")
    print(f"Regularization: {reg}, Top-k: {k_extract}")
    print(f"Sparse edges: {len(C_sparse.data):,} ({len(C_sparse.data)/(n_points**2)*100:.3f}%)")

    if dense_emd_cost is not None:
        print(f"\nDense EMD (TRUE OPTIMUM): {dense_emd_cost:.8f} in {dense_time:.3f}s")

    print(f"\nSinkhorn cost:   {sinkhorn_log['cost']:.8f} in {sinkhorn_log['time']:.3f}s")
    print(f"Sparse EMD cost: {sparse_emd_log['cost']:.8f} in {sparse_emd_log['time']:.3f}s")

    if dense_emd_cost is not None:
        error = abs(sparse_emd_log['cost'] - dense_emd_cost) / dense_emd_cost * 100
        print(f"\nError vs dense EMD: {error:.4f}%")

    total_time = sinkhorn_log['time'] + sparse_emd_log['time']
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"  Sinkhorn: {sinkhorn_log['time']:.3f}s ({sinkhorn_log['time']/total_time*100:.1f}%)")
    print(f"  Sparse EMD: {sparse_emd_log['time']:.3f}s ({sparse_emd_log['time']/total_time*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
