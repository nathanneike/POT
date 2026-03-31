#pragma once

#include <vector>

namespace lemon {

template <typename ArcsType, typename Cost>
struct SparsePricingView {
    using ArcVector = std::vector<ArcsType>;
    using IntVector = std::vector<int>;
    using CostVector = std::vector<Cost>;

    ArcVector row_ptr;
    IntVector row_source;
    ArcVector arc_id;
    IntVector target;
    CostVector cost;
    CostVector sign;
    ArcVector arc_pos;

    void clear() {
        row_ptr.clear();
        row_source.clear();
        arc_id.clear();
        target.clear();
        cost.clear();
        sign.clear();
        arc_pos.clear();
    }

    bool empty() const {
        return arc_id.empty();
    }

    template <typename GR, typename GetArcId, typename NodeIdFn, typename TargetVec,
              typename CostVec, typename StateVec>
    void build(const GR& graph,
               int n1,
               ArcsType arc_num,
               GetArcId get_arc_id,
               NodeIdFn node_id,
               const TargetVec& target_vec,
               const CostVec& cost_vec,
               const StateVec& state_vec) {
        clear();
        if (n1 <= 0 || arc_num <= 0) {
            return;
        }

        row_ptr.resize(n1 + 1, 0);
        row_source.resize(n1);
        arc_id.reserve(arc_num);
        target.reserve(arc_num);
        cost.reserve(arc_num);
        sign.reserve(arc_num);
        arc_pos.assign(arc_num, ArcsType(-1));

        ArcsType pos = 0;
        for (int src = 0; src < n1; ++src) {
            row_ptr[src] = pos;
            row_source[src] = node_id(src);

            typename GR::Arc a;
            graph.firstOut(a, GR::nodeFromId(src));
            for (; a != typename GR::Arc(-1); graph.nextOut(a)) {
                const ArcsType internal = get_arc_id(a);
                arc_id.push_back(internal);
                target.push_back(target_vec[internal]);
                cost.push_back(cost_vec[internal]);
                sign.push_back(static_cast<Cost>(state_vec[internal]));
                arc_pos[internal] = pos;
                ++pos;
            }
        }
        row_ptr[n1] = pos;
    }

    template <typename State>
    void updateArcState(ArcsType internal_arc, State state_value) {
        if (internal_arc < 0 || internal_arc >= static_cast<ArcsType>(arc_pos.size())) {
            return;
        }
        const ArcsType pos = arc_pos[internal_arc];
        if (pos < 0) {
            return;
        }
        sign[pos] = static_cast<Cost>(state_value);
    }
};

}  // namespace lemon
