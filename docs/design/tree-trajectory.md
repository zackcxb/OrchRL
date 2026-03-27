### Tree Trajectory Storage

#### Reasons for Tree Trajectory

In Agentic RL, a trajectory is traditionally viewed as a sequence alternating between LLM calls (assistant) and environment responses (user/tool call response):

prompt -> assistant -> user/tool -> assistant -> user/tool -> assistant ...

We are proposing to store trajectories as prefix trees (trie) instead of sequences for the following reasons:

1. **Parallel Sampling Methods**: For group-based advantage algorithms like GRPO, parralel samples may share a common prompt. In Multi-agent RL, some algorithms such as Stronger-MAS (https://arxiv.org/abs/2510.11062) uses tree sampling, which samples in parallel at each step in the agent loop. Tree trajectory storage naturally fits this parallel sampling method. Even for non group-based advantage like PPO, the user may sample outputs with the same prompt multiple times to get robust results.

2. **Context Compression**: Normally, all previous context is used as input in the next LLM call. Therefore, a sequence neatly represents the aggregated context. However, for multi-turn agent loop with very long contexts, the agent may compress the context to fit within the model's context window. In this case, previous context cannot be reused and one must start a new sequence whenever compression occurs. If the compressed context still shares some prefix with the previous context, we can reuse the shared prefix using tree trajectory.

3. **Retokenization Drift** (https://vllm.ai/blog/agent-lightning) may affect the reuse of shared prefix. Ideally, token drift should be eliminated. But once it occurs, the effect is similar to context compression.

4. **Training Efficiency**: New training algorithms such as Dynamic Tree Attention (https://arxiv.org/abs/2602.00482) handles tree-structured inputs efficiently, reducing repeated computations when samples in a batch share common prefixes. This benefit can only be reaped if the input batch is organized as a prefix tree.

Therefore, we propose to store trajectories as prefix trees (aka tries) instead of sequences. A new branch is created when:

1. Two trajectories start to diverge
2. Context compression or retokenization drift occurs


#### Node Fields

Apart from the usual fields of a prefix tree node, each node stores the following information:
```
role: str, "user" or "assistant". 
episode_ids: list[int], the episode ids that this node belongs to. If n episodes share this node as prefix, then the list length is n.
rewards: list[float], the rewards of the corresponding episode, same length as episode_ids. 
content: str, for "user", this is the user input or tool call return. For "assistant", this is the LLM output.
token_ids: list[int], the token ids of the content, can be directly consumed by the trainer. We store the token IDs to prevent token drift.
step: int, the step index of this node in the trajectory.
```
It should be noted that a mask field is not required if we use Dynamic Tree Attention.

A graphic representation is as below. 

<img width="2080" height="907" alt="屏幕截图 2026-03-27 105118" src="https://github.com/user-attachments/assets/aa18c7f1-9165-4022-9a60-41197af40e0d" />

