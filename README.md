# Iterative-Feature-eXclusion
A method for mitigating feature starvation in gradient boosted decision trees.

Code was written to be run in parallel on a cluster some changes might be necessary for regular use, though the IterativeFeatureExclusion class should work as is.

Requires: XGBoost, Scikit-learn

Warning: Running code may takea a long time if not on a distributed cluster.

Results from our experiments are available here through weights and biases:

Running IFX based on XGBoost with DART: https://api.wandb.ai/links/amielmeiseles-bgu/qto3g56z

Running IFX based on XGBoost with gbtree: https://api.wandb.ai/links/amielmeiseles-bgu/f9xw65fl
