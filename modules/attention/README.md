# 03 · Attention

> ▶ **Watch first:** [Graph Attention Networks (GAT) Explained: Why Transformers Are GNNs](https://www.youtube.com/watch?v=0-lj4OqHfDk). The video covers the intuition; this module is the runnable code behind it.

```
attention/
├── gnn.py      ← the GNN. Start here.
└── story/      ← optional: runs the models on the six friends and prints the walkthrough below
```

**[`gnn.py`](gnn.py)** is the whole lesson in code: `TasteAttention` and `HandTasteAttention`, each explained in its
docstring.

**[`story/`](story/)** is there if you want to see the models in action. It holds `AttentionStory`, with one short
method per section below. Run it with:

```bash
uv run python -m modules.attention.story      # or: just attention
```

&#160;

______________________________________________________________________

## The idea

GCN weights every message with a fixed number that depends only on how many friends each node has. **Graph
attention** (GAT) learns the weights instead: for every message it computes a score, and a softmax turns the scores
arriving at a node into **shares** that add up to 1. The node hears the share-weighted sum of its neighbours' vectors,
plus its own through a self-loop.

```
score_ij = LeakyReLU(att_src · W x_j + att_dst · W x_i)
share_ij = softmax of score_ij over everyone i hears from
```

Normally `W`, `att_src` and `att_dst` are learned. To make attention readable, `TasteAttention` fixes them so that
each score asks one question: *how much is this message about one taste?* `HandTasteAttention` does the same
arithmetic by hand, and the story checks the two agree.

&#160;

______________________________________________________________________

## Walkthrough

### 1. Fixed weights

Before attention, here are GCN's weights into Nancy from [02 · Aggregation](../aggregation/). They only count
friends, so Joe, who has just one friend, weighs most whatever anyone says:

| message from | GCN weight into Nancy |
| --- | ---: |
| Joe | 0.354 |
| Bill | 0.289 |
| Mei | 0.289 |
| herself | 0.250 |

&#160;

### 2. Attention asks a question: food

`TasteAttention(Taste.FOOD)` sets `W` to the identity, zeroes `att_dst`, and puts `att_src` at 2.0 on the food
feature only, so each score is 2 × how much the sender cares about food. Bill is the only one of Nancy's friends who
does:

| message from | attention share into Nancy |
| --- | ---: |
| Bill | 0.668 |
| Joe | 0.111 |
| Mei | 0.111 |
| herself | 0.111 |

`HandTasteAttention` gets the same shares (score → LeakyReLU → softmax), and Nancy's new vector leans toward food:
`[0.166, 0.233, 0.602]`.

&#160;

### 3. A different question: music

Point the question at music instead, and Nancy's attention moves from Bill to Joe:

| message from | attention share into Nancy |
| --- | ---: |
| Joe | 0.551 |
| Mei | 0.247 |
| herself | 0.111 |
| Bill | 0.091 |

Same friends, same messages, different shares. That is the whole point of attention: what a node listens to depends
on what the messages say.

&#160;

### 4. Multi-head attention

`GATConv(3, 16, heads=4)` runs four attention heads side by side, each free to ask its own question.

| GATConv(3, 16, heads=4) | output shape | attention shape |
| --- | --- | --- |
| concat=True | [6, 64] | [18, 4] |
| concat=False | [6, 16] | [18, 4] |

The attention has 18 rows, one per message (12 friendship messages + 6 self-loops), and one column per head.
`concat=True` joins the heads into 4 × 16 = 64 features; `concat=False` averages them back to 16.

&#160;

### 5. Everyone hears everyone

On a complete graph every friend hears every other friend: 30 messages for 6 friends. That is exactly the attention
pattern of a **Transformer**, where every token attends to every token. Transformers score with dot products
(query · key), which is the style of PyG's `TransformerConv`:

| heads=4 on the complete graph | output shape | attention shape |
| --- | --- | --- |
| GATConv | [6, 64] | [36, 4] |
| TransformerConv | [6, 64] | [30, 4] |

GATConv adds 6 self-loops to the 30 messages; TransformerConv keeps each friend's own vector through a separate root
weight instead.

&#160;

______________________________________________________________________

## Takeaways

- GCN's weights are fixed by the graph; attention's shares are computed from what each message says.
- A node's attention shares always add up to 1, because they come from a softmax.
- Multiple heads let a layer ask several questions at once.
- A Transformer is attention on a complete graph.

&#160;

______________________________________________________________________

Previous: [02 · Aggregation](../aggregation/)
