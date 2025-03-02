# Goals of the active learning module

This module contributes to maximize the probability of **finding good function approximation and good performing lipid within certain number of experiments, and the search space is within 220K predefined lipids**. Largely, this aligns with the idea of pool-based active learning.

Also we do have an ensemble of models, and there are specific active learning strategies for this, such as [this](https://modal-python.readthedocs.io/en/stable/content/examples/ensemble_regression.html).

## The difference

We have some unique settings and it is worth noting our goals and the difference to usual active learning

- We need to balance the exploration vs. exploitation, <-> while traditional active learning usually solely max the information gain (a.k.a. exploration) when collecting next round of data labels. This is because we have actually two goals during our experiments, after certain rounds of exps, we want to (1) have a good model that predicts the future mTP of 220K lipids well, and (2) find one of the best lipids and **have verified it in vitro during experiments**. Considering this goal particularly,
- We know the search space are not of independent examples, one lipid can be quite similar to another one. So instead of just maximizing the entropy examplewise, we should particularly think about the **entropy of all selected examples for the next round as a whole. In other words, also increase the diversity of selections**. We can also look into literature for this consideration of group.

# Design of experiments

## Proposed strategy

When making the experiment plan, we can set something like a ratio between exploration and exploitation. We can probably do it in straight-forward way, setting explicitly 0.5 of the plan for exploitation and the other 0.5 for exploitation. Specifically, each time planning the 96 x 2 wells of two new experiments:

1. Using the first 96 wells to test the most promising lipids. The specific lipids can be selected by the ensemble models of five-fold training on all previous data, and may also by the headwise tailwise ranking we proposed before.

2. Using the second 96 wells to test the most informative lipids. These can be selected by established active learning frameworks, and core strategies such as ranking prediction entropies. Each round of selection should also take in consideration of the diversity, to guarantee sufficient subdomains of the search space are explored.

# Plan of development

1. Start from active learning frameworks, such as modAL
2. Do they only target optimizing the function approximation. If so, we should adding our own exploitation strategies as proposed above
3. For the exploration strategy, we should probably also develop customized ones taking in consideration the diversity in the search space. For example, if solely ranking by entropy, all the most informative examples may just be some similar structures in clusters of the search space, so need to consider the diversity, picking up different lipids form different sub domains in the space. This can use similar strategies like the headwise tailwise selection, or direct hierarchical search based on the embedding similarities.
4. Test on existing data and compare strategies

Both 2 and 3 can be our unique interesting contributions, and will be worth interesting sections for discussing the results.
