## Project 3

#### Improving exploration strategies of evolution strategy algorithms for policy optimization.   

Some of recent research on improving ES algorithms for RL focuses on improving exploration strategies of these algorithms. 
The approach is usually to modify the standard reward function R($\pi$) to be maximized by extra additive factor that measures how
far" from the recently obtained policies the next constructed policy is and favors these policies that are different
from the previously seen [2].   


The extra factor is usually of the form E00 [D(; 0)], where D is a notion of some distance metric
defined on the set of all policies, expectation is with respect a certain distribution over all policies
(usually implemented as a distribution over a fixed-size batch of recently constructed policies, for
instance a uniform distribution) and alpha is a renormalizer that controls the importance of exploration.
Such an apprach might be especially useful in the sparse reward setting, where especially at the be-
ginning of training the algorithm does not get any reward signal and thus this extra exploration term
might guide it to explore policies that are substantially different from the previously constructed.

* Experiment with this approach on the set of chosen reinforcement learning tasks. In particular,
propose different distance metrics D that can be used to measure dissimilarity of policies and verify them empirically.

* Can you provide any theoretical guarantees regarding your proposed exploration
strategies ? 
* Compare with state-of-the-art on-policy optimization methods such as PPO, TRPO and
with baseline ES algorithm from [5]. 
* For those more theory-oriented, can you propose in that context
some exploration strategies based on distributions with negative dependence property (in particular:
determinantal point processes (DPPs)) ?
