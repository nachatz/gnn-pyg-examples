# 01 · Message passing

> ▶ **Watch first:** [Graph Neural Networks (GNNs) Explained Visually](https://www.youtube.com/watch?v=segXtOfZbG8). The video covers the intuition; this module is the runnable code behind it.

```
message_passing/
├── gnn.py      ← the GNN. Start here.
└── story/      ← optional: runs the models on the six friends and prints the walkthrough below
```

**[`gnn.py`](gnn.py)** is the whole lesson in code: `MeanMessagePassing`, `FrozenSAGEConv`, `FoodieGNN` and
`DeepMeanGNN`, each explained in its docstring.

**[`story/`](story/)** is there if you want to see the models in action. It holds `MessagePassingStory`, with one short
method per section below, and `FoodieTrainer`, the training loop for `FoodieGNN`. Run it with:

```bash
uv run python -m modules.message_passing.story      # or: just message-passing
```

&#160;

______________________________________________________________________

## The idea

A graph neural network (GNN) learns from connections. One GNN layer does three things for every node at the same time:

1. **Send.** Each node sends its feature vector to its neighbours.
2. **Gather.** Each node combines what it received into one vector. Here we take the **mean**.
3. **Update.** Each node mixes that vector with its own to get its new representation.

In PyG you write a layer like this by subclassing `MessagePassing`. `message()` is the *send* step, the `aggr="mean"`
argument is the *gather* step, and `update()` is the *update* step. `MeanMessagePassing` in [`gnn.py`](gnn.py) is
exactly that, in about ten lines:

```
new_i = ½ · x_i + ½ · mean(x_j for every friend j of i)
```

A real GNN learns weight matrices instead of the fixed ½:

```
new_i = W_self · x_i  +  W_neigh · mean(x_j for every friend j)  +  b
```

That's PyG's [`SAGEConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html).
If you set both weights to ½ · identity and the bias to zero (`FrozenSAGEConv`), it computes the same thing as
`MeanMessagePassing`.

&#160;

______________________________________________________________________

## Walkthrough

### 1. One round by hand, for Nancy

Nancy's friends are Joe, Bill and Mei. She takes the mean of their vectors:

| from | music | outdoors | food |
| --- | ---: | ---: | ---: |
| Joe | 0.900 | 0.100 | 0.000 |
| Bill | 0.000 | 0.100 | 0.900 |
| Mei | 0.500 | 0.500 | 0.000 |
| **mean** | **0.467** | **0.233** | **0.300** |

Then she blends it with her own vector, `[0.1, 0.9, 0.0]`:

```
½ · [0.1, 0.9, 0.0] + ½ · [0.467, 0.233, 0.300] = [0.283, 0.567, 0.150]
```

Nancy has picked up some music and food from her friends, but she's still mostly about the outdoors.

&#160;

### 2. The same round with PyG layers

Both `MeanMessagePassing` and the frozen `SAGEConv` give Nancy `[0.283, 0.567, 0.150]`, exactly matching the hand
calculation. A GNN layer is just this send-gather-update step, done for every node at once.

&#160;

### 3. Two layers = two hops

Joe and Bill aren't friends. After one layer Joe has heard only from Nancy, whose food score is 0, so his food score
stays at **0.000**.

After the first layer, Nancy's food score is 0.150, because she heard from Bill. In the second layer, Joe hears from
the updated Nancy, so his food score rises to **0.075**.

> **Rule of thumb:** with *k* layers, each node can see up to *k* hops away. That set of nodes is its *receptive
> field*.

&#160;

### 4. Training a tiny GNN

We label the friends with a food score of at least 0.5 as foodies: Bill, Priya and Omar. `FoodieGNN` learns its weights
from those labels:

```
SAGEConv(3 → 8) → ReLU → SAGEConv(8 → 2) → softmax over {not a foodie, foodie}
```

The loss falls from 0.73 to 0.004 in 60 epochs, and all six friends end up classified correctly. The hyperparameters
live in `FoodieSettings` in [`common/config.py`](../../common/config.py), and you can override them with environment
variables, for example `FOODIE_EPOCHS=10`.

**Caveat:** the model trains on all six friends, so this shows that it can fit the labels, not that it generalises to
friends it hasn't seen. On real graphs you hide some labels and check the predictions on those nodes.

&#160;

### 5. Over-smoothing

If 2 layers are good, are 30 better? Not here. `DeepMeanGNN(depth=30)` averages each friend with their neighbours 30
times, which washes out the differences. The story tracks the largest distance between any two friends' vectors:

| layers | largest distance |
| ---: | ---: |
| 0 | 1.2728 |
| 1 | 0.7382 |
| 5 | 0.2459 |
| 10 | 0.0629 |
| 30 | 0.0003 |

After 30 layers everyone sits at about `[0.267, 0.417, 0.317]`. Joe the music lover and Bill the foodie end up with nearly identical vectors.
This is **over-smoothing**, and it's why most GNNs use only 2 or 3 layers. Common fixes include residual connections,
normalisation layers, and `JumpingKnowledge` (combining the outputs of every layer).

&#160;

______________________________________________________________________

## Takeaways

- A GNN layer is three steps: send, gather, update. In PyG these are `message()`, `aggr` and `update()`.
- Stacking *k* layers lets information travel *k* hops.
- With learned weights, the same mechanism becomes a trainable model.
- Too many layers make every node look nearly the same.

&#160;

______________________________________________________________________

Next: [02 · Aggregation](../aggregation/), which compares the ways a node can gather its messages.
