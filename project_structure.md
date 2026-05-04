# Double Machine Learning (DML) Project Structure

This document provides a comprehensive overview of the `Doubel-Machine-Learning` project directory structure and its contents.

## Root Directory (`c:\Users\ge27tuv\Projects\Doubel-Machine-Learning`)

*   **`.git/`**: Git repository folder.
*   **`.gitignore`**: Specifies intentionally untracked files to ignore.
*   **`environment.yml`**: Conda/Python environment configuration file detailing dependencies.
*   **`2022.0287/`**: Subdirectory (purpose unknown, possibly related to a specific run or older data).
*   **`Comment-Robert/`**: Subdirectory (purpose unknown, possibly containing raw data or specific scripts for the robot).
*   **`experiments/`**: Subdirectory, currently empty.
*   **`notebooks/`**: Subdirectory intended for Jupyter notebooks for exploration and prototyping.
*   **`reports/`**: Subdirectory intended for generated reports or figures.

---

## 1. `Datasets/`
Contains all the raw data, processed JSONs, and generated embeddings.

*   **`Posts.json`** (608 MB): Contains the raw post data, including `content`, `likes_count`, `comments_count`, `created_at`, `ip_location`, and `user` information.
*   **`Comments.json`** (1.4 GB): Contains all comments on the posts. Used to identify which posts the robot (`@评论罗伯特`) replied to.
*   **`Users.json`** (262 MB): Contains user profile data, including `gender`, `birthday`, `verified` status, follower/friend counts, and credit scores.
*   **`embeddings_final.npy`** (2.2 GB): The full 1024-dimensional embeddings for all posts (N = 557,645), generated from a language model (e.g., BERT/RoBERTa).
*   **`embeddings_part_0.npy`**: Partial embedding file from an earlier processing stage.
*   **`id_mapping_final.csv`**: Maps the continuous array indices to the actual post `_id`s in `Posts.json`. Crucial for alignment.
*   **`id_mapping_part_0.csv`**: Partial mapping file from an earlier processing stage.

---

## 2. `models/`
Stores the trained Variational Autoencoder (VAE) models and the extracted lower-dimensional latent representations.

*   **`optimal_vae_model.pt`**: PyTorch state dictionary for the trained VAE model with a 150-dimensional latent space.
*   **`optimal_vae_latents.npy`** (318 MB): The extracted 150-dimensional latent representations for the valid subset of posts (N = 531,276).
*   **`vae_model_100.pt`**: PyTorch state dictionary for the newly trained VAE model with a 100-dimensional latent space (used for robustness checks).
*   **`vae_latents_100.npy`** (212 MB): The extracted 100-dimensional latent representations for the valid subset of posts (N = 531,276).

---

## 3. `results/`
Contains all generated metrics, intermediate CSVs, and final JSON results from the various DML stages.

*   **`clean_comment_metrics.csv`** (40 MB): Contains counts of "clean comments" (excluding the robot and the original poster) and unique commenters for each post.
*   **`sentiment_results.csv`** (36 MB): Contains the calculated sentiment scores: `post_sentiment`, `avg_clean_sentiment`, and `robot_comment_sentiment`.
*   **`dml_full_results.json`** (55 MB): The initial, comprehensive results from the first round of DML analysis.
*   **`dml_reestimation_results_v2.json`**: Results from the re-estimation phase (incorporating subgroup analyses and verified/time interactions).
*   **`vae_grid_search_losses.json`**: The out-of-sample losses for the VAE grid search across different latent dimensions.
*   **`vae_loss_vs_dimension.png`**: Plot visualizing the VAE grid search results.
*   **`robustness/`**: Subdirectory to store the comprehensive robustness check results:
    *   *Will contain*: `case1_dim150_fold10.json`, `case2_dim100_fold5.json`, `descriptive_statistics.csv`, and `correlation_matrix.csv`.

---

## 4. `scripts/`
Contains the core Python scripts executing the pipeline.

### Feature Extraction & Preparation
*   **`comment_measures.py`**: Extracts the "clean comments" and "unique commenters" metrics from `Comments.json` and saves them to `results/clean_comment_metrics.csv`.
*   **`sentiment_analysis.py`**: Uses the `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` model to calculate sentiment scores for posts, clean comments, and robot comments. Saves to `results/sentiment_results.csv`.
*   **`test_sentiment.py`**: A small test script for the sentiment model.
*   **`visualize_comments.py`**: Script to visualize the comment metrics.

### VAE Training
*   **`vae_grid_search.py`**: Performs a grid search over latent dimensions (25 to 200) to find the optimal VAE architecture. Saves `optimal_vae_model.pt` and `optimal_vae_latents.npy`.
*   **`train_vae_100.py`**: Specifically trains the 100-dimensional VAE used for Case 2 of the robustness checks.

### DML Analysis
*   **`dml_analysis_final.py`**: An earlier iteration of the DML analysis script.
*   **`dml_reestimation_v2.py`**: The refined DML script that includes subgroup analyses (16%, 34%, 50%) and interactions with `verified` status and `time`.
*   **`dml_sentiment_regressions.py`**: An intermediate script that attempted to integrate sentiment scores into the DML framework.
*   **`dml_robustness.py`**: The comprehensive final script that re-runs *all* models (m1-m8, subgroups, sentiment as DV) under two different VAE/fold configurations for robustness checking.
*   **`descriptive_stats.py`**: Calculates summary statistics and correlations for all variables used in the analysis.
