# Random Forest Exercise

Random forest classifier on the 2-D two moons dataset (`make_moons`, 1500 points,
noise 0.23, 50/50 train/test split). Seeds are fixed so the numbers are comparable
between runs.

## Scripts

- `exercise_rf.py` - trains one forest and plots the decision boundary.
  ```
  python exercise_rf.py --num_trees 32 --tree_depth 5
  ```
- `grid_search_rf.py` - grid search over number of trees (1-128) and max. depth
  (1-None) with 5-fold cross-validation on the training set. Plots heatmaps of the
  training and CV accuracy and marks two forests:
  - red: best CV accuracy
  - orange: shallowest forest (with at least `--min_trees` trees) that is within
    one standard error of the best one
  ```
  python grid_search_rf.py --seed 0 --min_trees 32
  ```

Plots go to `random_forest_plots/`.

## Results

| Forest | Trees | Depth | CV acc. | Train acc. | Test acc. |
|--------|-------|-------|---------|------------|-----------|
| Best | 128 | 8 | 94.1% | 98.8% | 95.3% |
| Shallowest within 1 SE | 32 | 5 | 93.6% | 96.8% | 94.5% |

(seed 0)

**Chosen: 32 trees, max. depth 5.**

## Observations

- Depth 1 means every tree is just one split (stump), so the boundary is one
  straight line -> underfitting, ~82%. More trees dont help there because all
  stumps pick almost the same split.
- Depth up to 3-4 still underfits a bit (~91%), the big jump comes at depth 5.
- Deep trees overfit: training accuracy goes to 100% but CV accuracy stays around 94%,
  the boundary gets small islands around the noisy points.
- More trees mostly make the boundary smoother / reduce variance, they dont really
  cause overfitting. After ~32 trees the difference is negligible (just slower).
- The best forest (128 trees, depth 8) is only ~0.5% better in CV, which is within
  the noise. Depth 5 is almost as good with a smaller train/test gap, so I went with that.
- Accuracy plateaus at ~95% because the moons overlap due to the noise.
- Boundaries are always boxy since every split only uses one feature.

