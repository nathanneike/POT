# Network Simplex Algorithm for Optimal Transport

This document explains the network simplex algorithm as implemented in `ot/lp/network_simplex_simple.h` for solving the sparse Earth Mover's Distance (EMD) problem.

## Overview

The network simplex algorithm solves the minimum-cost flow problem on a bipartite graph. For optimal transport, we have:
- **Source nodes** (supply nodes): Distribution `a` with total mass to send
- **Sink nodes** (demand nodes): Distribution `b` with total mass to receive  
- **Arcs/Edges**: Sparse set of allowed transportation routes with costs
- **Objective**: Find flow that satisfies supply/demand at minimum total cost

### How It Works: The Pivot Operation

The network simplex algorithm **maintains a feasible spanning tree structure** and moves from one spanning tree structure to another until it finds an optimal structure. 

At each iteration (called a **pivot**), the algorithm performs three steps:

1. **Select entering arc**: Choose a nontree arc violating its optimality condition
2. **Create and augment cycle**: Add this arc to the spanning tree, creating a cycle. Send the maximum possible flow around this cycle until at least one arc reaches its lower or upper bound
3. **Drop leaving arc**: Remove an arc whose flow has reached its bound, giving a new spanning tree structure

## Algorithm Structure

```
algorithm network_simplex:
begin
    1. Determine an initial feasible tree structure (T, L, U)
    2. Let x be the flow and π be the node potentials associated with this tree
    3. while some nontree arc violates the optimality conditions do
       begin
           a. Select an entering arc (k, l) violating its optimality condition
           b. Add arc (k, l) to the tree and determine the leaving arc (p, q)
           c. Perform a tree update and update the solutions x and π
       end
end
```

## Key Components

### Initial Setup

Before the main iteration loop, we need to:
1. Construct an initial feasible spanning tree
2. Compute flows for tree arcs
3. Compute node potentials

#### **Textbook Method: Artificial Arcs (Reference: Ahuja et al., p. 415)**

**Connectedness Assumption:** For every node j ∈ N - {1}, the network contains arcs (1,j) and (j,1) with sufficiently large costs and infinite capacities.

**Initial Tree Construction:**
```
For each node j ≠ 1:
    If b(j) ≥ 0:  // Supply node
        Include arc (1,j) in T with flow x_1j = b(j)
    If b(j) < 0:  // Demand node  
        Include arc (j,1) in T with flow x_j1 = -b(j)

Set L = remaining arcs
Set U = ∅ (empty)
```

**Node Potentials:** Computed from tree arc optimality conditions:
```
c_ij - π(i) + π(j) = 0  for all (i,j) ∈ T
Set π(1) = 0 (root potential)
```

This creates a **star tree** rooted at node 1 where:
- Every other node connects directly to the root
- Flows satisfy supply/demand constraints
- Potentials satisfy tree arc optimality
- All artificial arcs may have high cost (might be removed during initial pivots)

**Result:** Feasible basic solution to start the network simplex algorithm.

---

#### **Code Implementation**

**Initial Tree Construction** (`init()` in `network_simplex_simple.h`):
```cpp
// Set up artificial root node
_root = _node_num;
_parent[_root] = -1;
_pred[_root] = -1;
_thread[_root] = 0;
_pi[_root] = 0;

// For EQ supply constraints (sum = 0):
for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
    _parent[u] = _root;              // All nodes → root
    _pred[u] = e;                    // Artificial arc index
    _thread[u] = u + 1;              // Sequential threading
    _rev_thread[u + 1] = u;
    _succ_num[u] = 1;                // Leaf nodes
    _last_succ[u] = u;
    _state[e] = STATE_TREE;
    
    if (_supply[u] >= 0) {
        // Supply node: send to root
        _forward[u] = true;
        _pi[u] = 0;                  // Potential = 0 (cost = 0)
        _source[e] = u;
        _target[e] = _root;
        _flow[e] = _supply[u];       // Flow = supply (direct!)
        _cost[e] = 0;
    } else {
        // Demand node: receive from root
        _forward[u] = false;
        _pi[u] = ART_COST;           // High potential (high cost)
        _source[e] = _root;
        _target[e] = u;
        _flow[e] = -_supply[u];      // Flow = demand
        _cost[e] = ART_COST;
    }
}
```

**Key Properties of Star Tree Initialization:**

The code implements **exactly the same approach** as Ahuja et al. (p. 415) for the balanced case.

After this initialization, the **heuristic initial pivots** run (`initialPivots()`), followed by the main iteration loop.

#### **Initial Heuristic Pivots** (`initialPivots()`)

**Purpose:** Replace expensive artificial arcs with real problem arcs before starting main loop

**Strategy:**
1. Select promising real arcs (low-cost connections between supply/demand)
2. Try to pivot them into the tree
3. This removes artificial arcs early, improving starting solution

**Code:**
```cpp
bool initialPivots() {
    // Identify supply and demand nodes
    for each node u:
        if (_supply[u] > 0) add to supply_nodes
        if (_supply[u] < 0) add to demand_nodes
    
    // Build list of promising arcs to try
    if (single source and single sink):
        // Find path from sink to source (reverse search)
        arc_vector = arcs on this path
    else:
        // For each demand node, find cheapest incoming arc
        // (or for supply nodes, find cheapest outgoing arc)
        arc_vector = these minimum-cost arcs
    
    // Try to pivot each promising arc into the tree
    for each arc in arc_vector:
        if (arc violates optimality condition):
            in_arc = arc
            findJoinNode()
            findLeavingArc()
            changeFlow()
            updateTreeStructure()
            updatePotential()
}
```

**Effect:** Often eliminates most/all artificial arcs, giving much better starting point for main iteration.

**Note:** Flows and potentials are updated with each pivot using the same incremental methods as the main loop.

---

### 2. Optimality Testing and Entering Arc Selection

**Purpose**: Check if current solution is optimal; if not, find arc to add to tree

#### **Optimality Conditions**

This is the **fundamental theorem** that makes the network simplex algorithm work. A spanning tree solution is optimal if and only if:

**Definition**: For arc (i,j), the **reduced cost** is:
```
c̄ᵢⱼ = cᵢⱼ - π(i) + π(j)
```
where `π(i)` are node potentials (dual variables).

**Optimal spanning tree conditions**:

For a feasible spanning tree structure (T, L, U) where:
- **T** = tree arcs (basic variables)
- **L** = lower bound arcs (flow at lower bound, typically 0)
- **U** = upper bound arcs (flow at capacity)

The solution is **optimal** if there exist node potentials π such that:

**(a) c̄ᵢⱼ = 0 for all (i,j) ∈ T** (tree arcs have zero reduced cost)

**(b) c̄ᵢⱼ ≥ 0 for all (i,j) ∈ L** (arcs at lower bound can't decrease profitably)

**(c) c̄ᵢⱼ ≤ 0 for all (i,j) ∈ U** (arcs at upper bound can't increase profitably)

1. **Guides improvement**: Any arc violating its condition can enter the basis to reduce cost
2. **Dual feasibility**: These conditions ensure the dual problem is feasible (strong duality)


#### **Block Search Pivot Rule** (`BlockSearchPivotRule::findEnteringArc()`)

There are plenty of different feasible pivor rules with different advantages and disadvantages (this one here isn't easily parallelizable, we might switch to Dantzig since in theory this leads to less iteration since we get large decreases in the obejctive function)

The implementation uses **block search** for efficiency: instead of scanning all arcs every iteration, it examines arcs in blocks and stops early if a good violation is found.

**Key parameters**:
- `_block_size = max(√m, 10)` where m = number of arcs
- `_next_arc`: Starting position for next search (wraps around)

**Algorithm**:
```cpp
bool findEnteringArc() {
    Cost min = 0;  // Track most negative reduced cost
    ArcsType cnt = _block_size;
    
    // Scan from _next_arc to end, then wrap to beginning
    for (e = _next_arc; e != _search_arc_num; ++e) {
        c = _state[e] * (_cost[e] + _pi[_source[e]] - _pi[_target[e]]);
        if (c < min) {
            min = c;
            _in_arc = e;
        }
        if (--cnt == 0) {  // Examined full block
            // Check if violation is "good enough"
            a = max(|π(i)|, |π(j)|, |c_ij|);  
            if (min < -EPSILON*a) goto search_end;  // Accept this arc
            cnt = _block_size;  // Continue to next block
        }
    }
    // Continue from beginning to _next_arc
    for (e = 0; e != _next_arc; ++e) { /* same logic */ }
    
    if (min >= -EPSILON*a) return false; 
    
search_end:
    _next_arc = e;  
    return true;
}
```

- **Early termination**: If a block contains a good violation, pivot immediately rather than scan all arcs

### 3. Leaving Arc Determination

**Purpose**: Maintain tree structure by removing one arc when entering arc is added

**What happens**:
1. Adding entering arc (k,l) creates a **cycle** in the tree
2. Augment flow around this cycle to improve objective
3. One arc in the cycle hits its bound (0 or capacity) → this is the **leaving arc**

**Implementation** (`findLeavingArc()` lines 1115-1160):

The cycle consists of two paths from the entering arc's endpoints to their common ancestor (join node). The algorithm determines which direction to send flow and finds the arc with minimum residual capacity.

```cpp
bool findLeavingArc() {
    // Set direction based on entering arc state
    if (_state[in_arc] == STATE_LOWER) {
        first  = _source[in_arc];
        second = _target[in_arc];
    } else {
        first  = _target[in_arc];
        second = _source[in_arc];
    }
    
    delta = INF;
    char result = 0;
    
    // Path from first node to join
    for (int u = first; u != join; u = _parent[u]) {
        e = _pred[u];
        d = _forward[u] ? _flow[e] : INF;  
        if (d < delta) {
            delta = d;
            u_out = u;
            result = 1;
        }
    }
    
    // Path from second node to join
    for (int u = second; u != join; u = _parent[u]) {
        e = _pred[u];
        d = _forward[u] ? INF : _flow[e];  
        if (d <= delta) {
            delta = d;
            u_out = u;
            result = 2;
        }
    }
    
    // Set u_in and v_in for tree structure update
    if (result == 1) {
        u_in = first;
        v_in = second;
    } else {
        u_in = second;
        v_in = first;
    }
    return result != 0;
}
```

**Key insight**: 
- The `_forward` flag indicates if the arc points toward the parent
- Flow increases on one path, decreases on the other (depending on cycle direction)
- Leaving arc is the first to hit zero flow (its bound)

### 4. Tree Update and Solution Update

**Purpose**: Pivot operation - swap entering and leaving arcs, update flow and potentials

**Implementation** (`NetworkSimplexSparse::changeFlow()` and `updatePotential()`):

**Code location**: Lines ~1250-1315

**Flow update** (`changeFlow()` - incremental update around cycle only):

```cpp
// Augment flow by delta around the cycle
if (delta > 0) {
    Value val = _state[in_arc] * delta;
    _flow[in_arc] += val;
    
    // Update flows along path from source to join node
    for (int u = _source[in_arc]; u != join; u = _parent[u]) {
        _flow[_pred[u]] += _forward[u] ? -val : val;
    }
    
    // Update flows along path from target to join node
    for (int u = _target[in_arc]; u != join; u = _parent[u]) {
        _flow[_pred[u]] += _forward[u] ? val : -val;
    }
}

// Update arc states
if (change) {
    _state[in_arc] = STATE_TREE;  // Entering arc joins tree
    _state[_pred[u_out]] = (_flow[_pred[u_out]] == 0) ? STATE_LOWER : STATE_UPPER;
}
```

**Why incremental:** Only arcs on the cycle change flow. All other tree arcs maintain their existing flows.

- Entering arc becomes a tree arc
- Leaving arc becomes a nontree arc (at its bound)

**Potential update**:

**Textbook procedure (Ahuja et al., Figure 11.10):**
```
procedure update-potentials;
begin
    if q ∈ T₂ then y := q else y := p;
    if k ∈ T₁ then change := -c_kl else change := c_kl;
    π(y) := π(y) + change;
    z := thread(y);
    while depth(z) > depth(y) do
    begin
        π(z) := π(z) + change;
        z := thread(z);
    end;
end;
```

Where:
- Arc (k,l) enters, arc (p,q) leaves
- T₁, T₂ are the two subtrees created by removing (p,q)
- y is the root of the subtree being moved
- `change` is the uniform shift applied to all nodes in that subtree

**Code implementation** (`updatePotential()` lines 1306-1315):

The code implements the same idea as the textbook procedure - apply a uniform shift to all nodes in the moved subtree:

```cpp
void updatePotential() {
    // Calculate the uniform shift 
    Cost sigma = _forward[u_in] ?
        _pi[v_in] - _pi[u_in] - _cost[_pred[u_in]] :  // Forward arc
        _pi[v_in] - _pi[u_in] + _cost[_pred[u_in]];   // Reverse arc
    
    // Apply uniform shift to all nodes in subtree
    int end = _thread[_last_succ[u_in]];  // First node after subtree
    for (int u = u_in; u != end; u = _thread[u]) {
        _pi[u] += sigma;  
    }
}
```

**Why the optimization works:**

When a subtree is reconnected to a new parent, all nodes experience the **same potential shift** σ:

1. **Tree arcs within subtree:** Potential differences are preserved
   ```
   Before: cost[i,j] = π(i) - π(j)
   After:  cost[i,j] = (π(i)+σ) - (π(j)+σ) = π(i) - π(j)  ✓
   ```

2. **New boundary arc:** σ is chosen so this arc satisfies optimality
   ```
   σ ensures: cost[pred[u_in]] = π(v_in) - π(u_in)_new
   ```

**Key data structures used**:
- **`_pred[u]`**: Arc index connecting node u to its parent in the tree
- **`_forward[u]`**: Boolean - true if tree arc goes from source→target, false if target→source
- **`_last_succ[u]`**: Last node in u's subtree (in thread order)
- **`_thread[u]`**: Next node in depth-first traversal - enables O(n) subtree iteration without recursion

**Tree structure update** (`updateTreeStructure()`):
- Update `_parent`, `_pred`, `_thread`, `_rev_thread` arrays
- Update `_succ_num` and `_last_succ` for affected nodes
- Maintain thread ordering for efficient tree traversal
- Uses **stem-and-blossom** technique: reverses parent-child relationships along the path 

### 5. Termination

**Condition**: All nontree arcs satisfy optimality (reduced cost ≥ 0)

**Implementation**:
```cpp
if (findEnteringArc() == false) {
    // No entering arc found → optimal solution
    return OPTIMAL;
}
```

**Output**:
- Optimal flow on each arc
- Node potentials (dual solution)
- Total transportation cost

## References

1. **Network Flows: Theory, Algorithms, and Applications** - Ahuja, Magnanti, Orlin 
   - Chapter 11: MINIMUM COST FLOWS: NETWORK SIMPLEX ALGORITHMS

2. **LEMON Graph Library** - Implementation reference
   - https://lemon.cs.elte.hu/ (Very Old website)
   - High-performance C++ network simplex implementation

3. **Original EMD Paper**: Rubner, Tomasi, Guibas (2000)
   - "The Earth Mover's Distance as a Metric for Image Retrieval"

