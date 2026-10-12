# 02 · Aggregation

> ▶ **Watch first:** [GNN Aggregation Explained: Sum vs Mean vs Max (and GCN)](https://www.youtube.com/watch?v=TevxIaGq7Cs). The video covers the intuition; this module is the runnable code behind it.

```
aggregation/
├── gnn.py      ← the GNN. Start here.
└── story/      ← optional: runs the models on the six friends and prints the walkthrough below
```

**[`gnn.py`](gnn.py)** is the whole lesson in code: `NeighbourhoodAggregator`, `AggregatingSAGEConv`, `HandGCNConv` and
`IdentityGCNConv`, each explained in its docstring.

**[`story/`](story/)** is there if you want to see the models in action. It holds `AggregationStory`, with one short
method per section below. Run it with:

```bash
uv run python -m modules.aggregation.story      # or: just aggregation
```

&nbsp;

---

## The idea

In the *gather* step of message passing, a node turns a pile of messages from its neighbours into one vector. The
function that does this is called the **aggregator**. In PyG it's a single argument, `MessagePassing(aggr=...)`, and
choosing it is a real design decision, because each aggregator throws some information away.

The aggregator must not depend on the order of the messages, because friends have no natural order. So we can't just
glue the messages into one long list. The usual choices are:

| aggregator | what it computes | remembers | forgets |
| --- | --- | --- | --- |
| **sum** | adds every message together | how many friends there are, and how strong each signal is | nothing, but it grows without limit as the number of friends rises |
| **mean** | the average message | the overall mix of what friends are like | how many friends there are |
| **max** | the largest value for each feature | the strongest single signal | how many friends sent it, and everything below the peak |

&nbsp;

---

## Walkthrough

### 1. Order doesn't matter

We shuffle the order of the graph's edges and run `NeighbourhoodAggregator` with each aggregator. Nancy gets exactly
the same result every time. If we glue her messages into one long list instead, then *Joe, Bill, Mei* and
*Mei, Bill, Joe* give different lists. That's why GNNs use order-blind aggregators.

&nbsp;

### 2. Same messages, different answers

`AggregatingSAGEConv(aggr)` is PyG's `SAGEConv` reduced to its aggregator: it ignores Nancy's own vector and passes
the aggregate straight through. Here's what Nancy hears from Joe, Bill and Mei:

| aggregator | music | outdoors | food |
| --- | ---: | ---: | ---: |
| sum | 1.400 | 0.700 | 0.900 |
| mean | 0.467 | 0.233 | 0.300 |
| max | 0.900 | 0.500 | 0.900 |

Max says Nancy's friends include someone who loves music *and* someone who loves food. It doesn't record that those
are two different people.

&nbsp;

### 3. Sum grows with the crowd

Omar's taste arrives from 1, then 10, then 1000 identical fans:

| fans | sum | mean |
| ---: | --- | --- |
| 1 | `[0.5, 0, 0.5]` | `[0.5, 0, 0.5]` |
| 10 | `[5, 0, 5]` | `[0.5, 0, 0.5]` |
| 1000 | `[500, 0, 500]` | `[0.5, 0, 0.5]` |

In real graphs, a few very popular nodes can have huge sums that swamp the next layer.

&nbsp;

### 4. Mean can't tell 1 friend from 2 identical friends

A friend who hears from Mei, and one who hears from Mei and her twin, both average to
`[0.5, 0.5, 0]`. Only sum (`[1, 1, 0]`) notices the twin.

&nbsp;

### 5. Max can't tell 1 food lover from 3

Bill plus two non-foodies, and three Bills, both give a max food score of
0.9. Sum tells them apart: 0.9 vs 2.7.

> This is why the [GIN paper](https://arxiv.org/abs/1810.00826) argues for **sum**: it is the only one of the three
> that can tell all these neighbourhoods apart. Mean and max are more stable, though, so they are often good choices
> in practice. PyG also lets you combine several aggregators, for example `aggr=["sum", "mean", "max"]`.

&nbsp;

### 6. GCN normalisation: a middle ground

[`GCNConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html) first
adds a self-loop to every node, so each node also hears its own message. It then gives each message from *j* to *i* the
weight:

```
w_ij = 1 / sqrt(deg_i · deg_j)      (degrees include the self-loop)
```

`HandGCNConv` in [`gnn.py`](gnn.py) writes this out in three lines: add self-loops, count degrees, multiply. A message
from a well-connected friend gets turned down, so popular nodes can't drown out everyone else. Unlike a plain mean,
though, the weights still depend on how many friends each node has. Nancy's degree is 4: Joe, Bill, Mei and herself.

| from | degree | weight into Nancy |
| --- | ---: | ---: |
| Joe | 2 | 1/√(2·4) = **0.354** |
| Bill | 3 | 1/√(3·4) = **0.289** |
| Mei | 3 | 1/√(3·4) = **0.289** |
| herself | 4 | 1/√(4·4) = **0.250** |

Joe's message counts the most because Joe has only one friend, so Nancy is all he has. The script checks that
`HandGCNConv` gives the same weights as PyG's own `gcn_norm`, and that its output matches `GCNConv`'s.

&nbsp;

---

## Takeaways

- Aggregators must ignore the order of the messages.
- **Sum** keeps the count but can blow up; **mean** is stable but forgets the count; **max** keeps only the strongest
  signal.
- GCN weighting shrinks messages from well-connected nodes, giving a balance between sum and mean.

&nbsp;

---

Previous: [01 · Message passing](../message_passing/)
