"""Grid search for the random forest exercise.

Tries different numbers of trees and tree depths with cross-validation on the
training set and plots the accuracies as heatmaps. Then picks the parameters
and checks them on the test set.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as sk_ensemble
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


def heatmap(ax, scores, num_trees, depths, title):
    """Draws a depth x trees accuracy matrix with the values written in the cells."""
    im = ax.imshow(scores, cmap="viridis", vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(num_trees)))
    ax.set_xticklabels(num_trees)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Max. tree depth")
    ax.set_title(title)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            ax.text(j, i, "{:.1%}".format(scores[i, j]), ha="center", va="center",
                    color="white" if scores[i, j] < 0.9 else "black", fontsize=8)
    return im


def main(save_fig: bool, result_dir: str, seed: int, cv_folds: int, min_trees: int):
    os.makedirs(result_dir, exist_ok=True)

    # same data as exercise_rf.py
    features, labels = make_moons(n_samples=1500, noise=0.23, random_state=seed)
    feat_train, feat_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.5, random_state=42
    )

    num_trees = [1, 2, 4, 8, 16, 32, 64, 128]
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, None]  # None = no limit

    forest = sk_ensemble.RandomForestClassifier(max_features=2, random_state=seed)
    search = GridSearchCV(
        forest,
        {"n_estimators": num_trees, "max_depth": depths},
        cv=cv_folds,
        return_train_score=True,
        n_jobs=-1,
    )
    print("Running grid search ({} combinations, {}-fold CV)...".format(
        len(num_trees) * len(depths), cv_folds))
    # only the training data is used here, test set is for the end
    search.fit(feat_train, labels_train)
    res = search.cv_results_

    # put results into matrices (rows = depth, cols = trees)
    train_scores = np.zeros((len(depths), len(num_trees)))
    cv_scores = np.zeros((len(depths), len(num_trees)))
    for k, p in enumerate(res["params"]):
        i = depths.index(p["max_depth"])
        j = num_trees.index(p["n_estimators"])
        train_scores[i, j] = res["mean_train_score"][k]
        cv_scores[i, j] = res["mean_test_score"][k]

    # the best one is usually very deep, but lots of combinations are basically
    # equally good. so take the shallowest one that is within 1 standard error
    # of the best (and has enough trees to be stable)
    best = search.best_index_
    se = res["std_test_score"][best] / np.sqrt(cv_folds)
    threshold = res["mean_test_score"][best] - se
    ok = [k for k, p in enumerate(res["params"])
          if res["mean_test_score"][k] >= threshold and p["n_estimators"] >= min_trees]
    chosen = min(ok, key=lambda k: (res["params"][k]["max_depth"] or 999,
                                    res["params"][k]["n_estimators"]))
    chosen_params = res["params"][chosen]

    print("Best CV accuracy: {:.2%} with {}".format(res["mean_test_score"][best], search.best_params_))
    print("1-SE threshold: {:.2%}".format(threshold))
    print("Chosen: {} (CV {:.2%})".format(chosen_params, res["mean_test_score"][chosen]))

    # check both on the test set
    for name, params in [("best", search.best_params_), ("chosen", chosen_params)]:
        f = sk_ensemble.RandomForestClassifier(max_features=2, random_state=seed, **params)
        f.fit(feat_train, labels_train)
        print("  {}: train {:.2%}, test {:.2%}".format(
            name,
            accuracy_score(labels_train, f.predict(feat_train)),
            accuracy_score(labels_test, f.predict(feat_test))))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    heatmap(axes[0], train_scores, num_trees, depths, "Training accuracy")
    im = heatmap(axes[1], cv_scores, num_trees, depths, "Cross-validation accuracy")
    fig.colorbar(im, ax=axes, label="Accuracy")

    # mark best (red) and chosen (orange) cell
    for p, col in [(search.best_params_, "red"), (chosen_params, "orange")]:
        x = num_trees.index(p["n_estimators"])
        y = depths.index(p["max_depth"])
        axes[1].add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=col, linewidth=3))

    if save_fig:
        t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fname = os.path.join(result_dir, "rf_grid_search_{}.png".format(t))
        fig.savefig(fname, bbox_inches="tight")
        print("Plot saved as " + fname)

    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description="Grid search for the random forest exercise")
    parser.add_argument("--save_fig", action=argparse.BooleanOptionalAction, default=True,
                        help="Save plot to result_dir (--no-save_fig to disable).")
    parser.add_argument("--result_dir", type=str,
                        default=os.path.normpath(os.path.join(script_dir, "random_forest_plots")),
                        help="Directory for results.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--min_trees", type=int, default=32,
                        help="Min. number of trees for the chosen forest.")

    args = parser.parse_args()
    main(args.save_fig, args.result_dir, args.seed, args.cv_folds, args.min_trees)
