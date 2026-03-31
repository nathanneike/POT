#include "ot/lp/EMD.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct SolveResult {
    int status;
    double cost;
    uint64_t n_flows;
    double avg_ms;
};

int parse_int(char** argv, int argc, int idx, int default_value) {
    if (idx >= argc) return default_value;
    return std::stoi(argv[idx]);
}

uint64_t parse_u64(char** argv, int argc, int idx, uint64_t default_value) {
    if (idx >= argc) return default_value;
    return static_cast<uint64_t>(std::stoull(argv[idx]));
}

void build_geometric_sparse_problem(
    int n,
    int row_topk,
    uint32_t seed,
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<uint64_t>& edge_sources,
    std::vector<uint64_t>& edge_targets,
    std::vector<double>& edge_costs
) {
    x.assign(n, 1.0 / n);
    y.assign(n, 1.0 / n);

    edge_sources.reserve(static_cast<size_t>(n) * row_topk);
    edge_targets.reserve(static_cast<size_t>(n) * row_topk);
    edge_costs.reserve(static_cast<size_t>(n) * row_topk);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coord_dist(0.0, 1.0);

    std::vector<double> src_x(n);
    std::vector<double> src_y(n);
    std::vector<double> tgt_x(n);
    std::vector<double> tgt_y(n);
    for (int i = 0; i < n; ++i) {
        src_x[i] = coord_dist(rng);
        src_y[i] = coord_dist(rng);
        tgt_x[i] = coord_dist(rng);
        tgt_y[i] = coord_dist(rng);
    }

    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(n);

    for (int src = 0; src < n; ++src) {
        candidates.clear();
        for (int tgt = 0; tgt < n; ++tgt) {
            const double dx = src_x[src] - tgt_x[tgt];
            const double dy = src_y[src] - tgt_y[tgt];
            const double dist2 = dx * dx + dy * dy;
            candidates.emplace_back(dist2, tgt);
        }
        const int keep = std::min(row_topk, n);
        if (keep < n) {
            std::nth_element(
                candidates.begin(),
                candidates.begin() + keep,
                candidates.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }
            );
            candidates.resize(keep);
        }

        bool has_diagonal = false;
        for (const auto& candidate : candidates) {
            if (candidate.second == src) {
                has_diagonal = true;
                break;
            }
        }
        if (!has_diagonal && !candidates.empty()) {
            const double dx = src_x[src] - tgt_x[src];
            const double dy = src_y[src] - tgt_y[src];
            candidates.back() = {dx * dx + dy * dy, src};
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
        for (const auto& candidate : candidates) {
            edge_sources.push_back(static_cast<uint64_t>(src));
            edge_targets.push_back(static_cast<uint64_t>(candidate.second));
            edge_costs.push_back(candidate.first);
        }
    }
}

SolveResult run_sparse_solver(
    int n,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<uint64_t>& edge_sources,
    const std::vector<uint64_t>& edge_targets,
    const std::vector<double>& edge_costs,
    int warmup,
    int runs,
    uint64_t max_iter,
    bool use_row_pricing
) {
    if (use_row_pricing) {
        setenv("POT_SPARSE_ROW_PRICING", "1", 1);
    } else {
        unsetenv("POT_SPARSE_ROW_PRICING");
    }

    std::vector<uint64_t> flow_sources(edge_sources.size());
    std::vector<uint64_t> flow_targets(edge_sources.size());
    std::vector<double> flow_values(edge_sources.size());
    std::vector<double> alpha(n, 0.0);
    std::vector<double> beta(n, 0.0);

    auto invoke = [&]() -> std::pair<int, std::pair<double, uint64_t>> {
        uint64_t n_flows = 0;
        double cost = 0.0;
        std::fill(alpha.begin(), alpha.end(), 0.0);
        std::fill(beta.begin(), beta.end(), 0.0);
        const int status = EMD_wrap_sparse(
            n,
            n,
            const_cast<double*>(x.data()),
            const_cast<double*>(y.data()),
            edge_sources.size(),
            const_cast<uint64_t*>(edge_sources.data()),
            const_cast<uint64_t*>(edge_targets.data()),
            const_cast<double*>(edge_costs.data()),
            flow_sources.data(),
            flow_targets.data(),
            flow_values.data(),
            &n_flows,
            alpha.data(),
            beta.data(),
            &cost,
            max_iter,
            nullptr,
            nullptr
        );
        return {status, {cost, n_flows}};
    };

    for (int i = 0; i < warmup; ++i) {
        (void)invoke();
    }

    int final_status = 0;
    double final_cost = 0.0;
    uint64_t final_n_flows = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) {
        const auto result = invoke();
        final_status = result.first;
        final_cost = result.second.first;
        final_n_flows = result.second.second;
    }
    auto t1 = std::chrono::steady_clock::now();

    SolveResult out;
    out.status = final_status;
    out.cost = final_cost;
    out.n_flows = final_n_flows;
    out.avg_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count() / runs;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const int n = parse_int(argv, argc, 1, 5000);
    const int row_topk = parse_int(argv, argc, 2, 16);
    const int warmup = parse_int(argv, argc, 3, 1);
    const int runs = parse_int(argv, argc, 4, 5);
    const uint64_t max_iter = parse_u64(argv, argc, 5, 100000);
    const uint32_t seed = static_cast<uint32_t>(parse_u64(argv, argc, 6, 0));

    std::vector<double> x;
    std::vector<double> y;
    std::vector<uint64_t> edge_sources;
    std::vector<uint64_t> edge_targets;
    std::vector<double> edge_costs;
    build_geometric_sparse_problem(n, row_topk, seed, x, y, edge_sources, edge_targets, edge_costs);

    const SolveResult baseline = run_sparse_solver(
        n, x, y, edge_sources, edge_targets, edge_costs, warmup, runs, max_iter, false
    );
    const SolveResult row = run_sparse_solver(
        n, x, y, edge_sources, edge_targets, edge_costs, warmup, runs, max_iter, true
    );

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Sparse row-pricing benchmark\n";
    std::cout << "n=" << n
              << " row_topk=" << row_topk
              << " pattern=geometric"
              << " edges=" << edge_sources.size()
              << " warmup=" << warmup
              << " runs=" << runs
              << " max_iter=" << max_iter
              << " seed=" << seed << "\n";
    std::cout << "baseline: status=" << baseline.status
              << " cost=" << baseline.cost
              << " flows=" << baseline.n_flows
              << " avg_ms=" << baseline.avg_ms << "\n";
    std::cout << "row     : status=" << row.status
              << " cost=" << row.cost
              << " flows=" << row.n_flows
              << " avg_ms=" << row.avg_ms << "\n";
    std::cout << "speedup=" << (baseline.avg_ms / row.avg_ms)
              << "x abs_cost_diff=" << std::abs(baseline.cost - row.cost)
              << " rel_cost_diff=" << (baseline.cost != 0.0 ? std::abs(baseline.cost - row.cost) / std::abs(baseline.cost) : 0.0)
              << "\n";

    if (baseline.status != row.status) {
        std::cerr << "status mismatch\n";
        return 2;
    }
    if (baseline.status == OPTIMAL && std::abs(baseline.cost - row.cost) > 1e-9) {
        std::cerr << "cost mismatch\n";
        return 3;
    }
    return 0;
}
