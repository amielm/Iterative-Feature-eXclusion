# Iterative-Feature-eXclusion
A method for strategically removing features during XGBoost training to reduce overfitting

Code was written to be run in parallel on a cluster some changes might be necessary for regular use, though the IterativeFeatureExclusion class should work as is.

Requires: XGBoost, Scikit-learn

Warning: Running code may takea a long time if not on a distributed cluster.
