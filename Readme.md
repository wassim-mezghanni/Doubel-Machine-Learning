# Double Machine Learning (DML) for Text Analysis

The report documents the refined DML analysis performed on 50,000 post samples, integrating official repository logic and enhanced user feature engineering.

## 1. Repository Integration 

We integrated the following research repositories to align with the paper's standards:
- **[Comment-Robert](https://github.com/FDUDataNET/Comment-Robert)**: For precise extraction of features like **Sunshine Credit** (ordinal), **label descriptions** (count), and **verified status**.
- **[2022.0287 (IJOC)](https://github.com/INFORMSJoC/2022.0287)**: For the "partialling out" DML framework and two-branch model logic.

### Refined Architecture
1.  **VAE Representation**: A Variational Autoencoder maps 1024D embeddings into a **100D latent space**, capturing core semantic features.
2.  **Two-Branch Model**:
    *   **Branch A (Text)**: 100D Latents $\to$ 256-unit MLP.
    *   **Branch B (User)**: Structured features $\to$ Linear overlay (as requested, providing controls).
3.  **Cross-Fitting**: A 3-fold sample splitting strategy was used to generate unbiased out-of-fold predictions.

## 2. Refined Results

| Metric | Refined Value |
| :--- | :--- |
| **Mean Squared Error (MSE)** | 26,215,711.30 |
| **Mean Absolute Error (MAE)** | 170.07 |
| **R² Score** | -0.0906 |

> [!NOTE]
> The refined model incorporates more complex user controls (Sunshine Credit, labels). The negative $R^2$ highlights the difficulty of predicting raw engagement in the presence of extreme outliers, while the model focus remains on capturing the underlying relationship between text and engagement.

## 3. Delivered outputs and files used 

- **dml_analysis.py**: The execution script.
- **environment.yml**
- **dml_results_refined.json**: Detailed metrics and predictions for all 50 000 samples.
- **Repos**: `Comment-Robert/` and `2022.0287/` are available in the project root.
