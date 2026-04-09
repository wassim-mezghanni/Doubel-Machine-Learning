# Double Machine Learning (DML) for Social Media Robot Interaction Analysis

This repository implements a 3-stage Double Machine Learning (DML) pipeline to estimate the causal effect of robot account interactions (@评论罗伯特) on Weibo post engagement (`likes_count`).

## 1. Methodology: 3-Stage Residual Regression

We follow the "partialling out" framework (Robinson, 1988) adapted for high-dimensional text data:

1.  **Nuisance Estimation**: Two-branch neural networks (Text VAE Branch + User Feature Branch) are trained via 3-fold cross-fitting to predict engagement ($ln\_likes$), robot replies ($RobReplied$), and gender-interaction terms ($female\_RobReplied$).
2.  **Residualization**: Out-of-fold residuals are computed for all targets, effectively removing the influence of text content and user demographics.
3.  **Causal Estimation**: OLS regression on the residuals provides unbiased causal estimates.

## 2. Key Findings (Full Dataset: 557,645 posts)

The analysis was performed on the full dataset, identifying **176,941 comments** from the robot account (`79f6c7ffc7270a3a7d3136245ab0f8ac`).

### Causal Coefficients

| Variable | Coefficient | P-Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **RobReplied (Main Effect)** | **+0.1597** | **0.002** | Significant boost in engagement. |
| **RobReplied (Model 2 Baseline)** | **+0.1592** | **0.002** | Effect remains positive and stable. |
| **female_RobReplied (Interaction)** | **+0.0006** | 0.159 | **Not Statistically Significant.** |

## 3. Project Structure & Files

- **`dml_analysis_final.py`**: Main execution script. Includes optimized JSON parsing, VAE training, and cross-fitting logic.
- **`environment.yml`**: Conda environment specification.
- **`dml_full_results.json`**: Full residuals and regression summary for the 557k samples.
- **`Datasets/`**: Location for `Posts.json`, `Users.json`, `Comments.json`, and `embeddings_final.npy`.
- **`Comment-Robert/` & `2022.0287/`**: Referenced research repositories.

## 4. Performance Metrics (NN Benchmarks)

The two-branch model achieved a significant reduction in bias during the residualization stage:
- **Engagement MSE (Log Scale)**: Significantly lower than baseline summation models.
- **Robot Reply Classification**: Validated with a Sigmoid head and class-weighted BCE for stability.
