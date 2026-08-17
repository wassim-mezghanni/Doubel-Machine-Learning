# Double Machine Learning for Human-LLM Social Interaction Analysis

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Double Machine Learning](https://img.shields.io/badge/Methodology-Double%20Machine%20Learning%20(DML)-darkgreen.svg)](https://academic.oup.com/ectj/article/21/1/C1/5056456)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/16921462)

An end-to-end econometric and causal machine learning pipeline investigating the causal effects of social media LLM agent replies (specifically Weibo's `@评论罗伯特` / `@CommentR`) on user engagement, sentiment dynamics, and demographic/gender disparities.

---

## 📖 Overview

As conversational AI and social bots increasingly participate in public online spaces, understanding their causal impact on human engagement and communication patterns is critical. This project implements a **Double Machine Learning (DML)** framework based on Robinson's partially linear model to identify the causal effect of LLM agent replies (`RobReplied`), user gender (`female`), and their interactions on post-level outcomes.

### Key Highlights
- **High-Dimensional Text Representation**: Encodes large-scale post contents into 1024-dimensional LLM embeddings (RoBERTa), compressed into lower-dimensional latent representations (150-dim and 100-dim) using **Variational Autoencoders (VAE)** with grid search optimization.
- **Dual-Branch Deep Residualization**: Employs dual-branch PyTorch neural networks combining text latent embeddings and structured user metadata (e.g., follower counts, credit scores, account age, location fixed effects) with $K$-fold cross-fitting.
- **Engagement & Sentiment Outcomes**: Measures effects on multiple log-transformed metrics:
  - $\ln(1 + \text{likes})$
  - $\ln(1 + \text{comments})$
  - $\ln(1 + \text{clean\_comments})$ (excluding bot and poster self-replies)
  - $\ln(1 + \text{unique\_commenters})$
  - Net sentiment scores via `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`.
- **Heterogeneity & Moderation Analyses**: Evaluates moderating roles of user verification (`verified`), temporal dynamics (`time`), regional Sex Ratio at Birth (`SRB`), mainland residency, poster age, and LLM comment sentiment polarity.
- **Extensive Robustness Checks**: Validates findings across multiple VAE latent dimensions (150 vs. 100) and cross-fitting fold configurations (10-fold vs. 5-fold).

---

##  Project Structure

```text
Doubel-Machine-Learning/
├── Datasets/                     # Raw and processed datasets & embeddings
│   ├── Posts.json                # Post-level metadata and contents (N = 557,645)
│   ├── Comments.json             # Comment-level records and bot replies
│   ├── Users.json                # User profiles and demographic indicators
│   ├── embeddings_final.npy      # 1024-dim post text embeddings
│   └── id_mapping_final.csv      # Alignment mapping between post IDs and array indices
├── models/                       # Trained VAE architectures & extracted latent vectors
│   ├── optimal_vae_model.pt      # Optimal VAE checkpoint (150-dim latent space)
│   ├── optimal_vae_latents.npy   # Extracted 150-dim latents (N = 531,276)
│   ├── vae_model_100.pt          # Robustness VAE checkpoint (100-dim latent space)
│   └── vae_latents_100.npy       # Extracted 100-dim latents (N = 531,276)
├── results/                      # Generated metrics, regression outputs & visualizations
│   ├── clean_comment_metrics.csv # Clean comment and unique commenter counts
│   ├── sentiment_results.csv     # Sentiment scores for posts, clean comments, and bot
│   ├── dml_reestimation_results_v2.json  # Main DML estimation results
│   ├── vae_loss_vs_dimension.png # VAE grid search loss plot
│   └── robustness/               # Comprehensive robustness check outputs
│       ├── case1_dim150_fold10.json
│       ├── case2_dim100_fold5.json
│       ├── descriptive_statistics.csv
│       └── correlation_matrix.csv
├── scripts/                      # Core Python execution pipeline
│   ├── comment_measures.py       # Computes clean comments & unique commenter metrics
│   ├── sentiment_analysis.py     # Batch sentiment scoring via Erlangshen-RoBERTa
│   ├── test_sentiment.py         # Sentiment model test script
│   ├── visualize_comments.py     # Comment distribution visualization
│   ├── vae_grid_search.py        # VAE latent dimension search (25 to 200)
│   ├── train_vae_100.py          # 100-dim VAE training for robustness
│   ├── dml_reestimation_v2.py    # Refined DML estimation with subgroups & interactions
│   ├── dml_robustness.py         # Automated full robustness suite (Case 1 & Case 2)
│   └── descriptive_stats.py      # Summary statistics and correlation matrix generator
├── notebooks/                    # Jupyter notebooks for analysis & exploration
│   ├── clean_comments_analysis.ipynb
│   └── dml_diffusion_analysis.ipynb
├── environment.yml               # Conda environment specification
└── project_structure.md          # Detailed directory breakdown
```

---

## Environment Setup

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate dml_env
```

### Core Dependencies
- **Python**: 3.9
- **Deep Learning**: PyTorch (`torch`, `torchvision`), `tqdm`
- **Data & Causal Econometrics**: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `openpyxl`, `matplotlib`
- **NLP / Transformers**: Hugging Face `transformers` (for sentiment scoring with Erlangshen-RoBERTa)

---

##  Step-by-Step Pipeline

### 1. Data Preparation & Feature Extraction
Compute cleaned comment measures (removing bot comments and poster self-replies) and extract sentiment scores:
```bash
# Extract clean comment metrics and unique commenter counts
python scripts/comment_measures.py

# Calculate sentiment scores for posts, clean comments, and bot replies
python scripts/sentiment_analysis.py
```

### 2. VAE Representation Learning
Reduce the 1024-dimensional text embeddings into a compact latent space:
```bash
# Perform grid search over latent dimensions (dim=25..200)
python scripts/vae_grid_search.py

# Train the 100-dim VAE model for robustness testing
python scripts/train_vae_100.py
```

### 3. Double Machine Learning (DML) Estimation
Execute the DML cross-fitting pipeline to residualize outcomes and treatments using the dual-branch neural network, followed by OLS estimation on the residuals:
```bash
# Run primary DML models including subgroups and moderator interactions
python scripts/dml_reestimation_v2.py
```

### 4. Robustness Checks & Descriptive Statistics
Run the complete automated robustness suite across latent dimensions and cross-fitting fold configurations:
```bash
# Execute robustness suite (Case 1: dim=150, 10 folds; Case 2: dim=100, 5 folds)
python scripts/dml_robustness.py

# Generate descriptive statistics and correlation matrix
python scripts/descriptive_stats.py
```

---

##  Model Specifications

The DML estimation evaluates the following econometric specifications across the four primary engagement DVs:

| Model ID | Specification / Focus | Key Regressors |
| :--- | :--- | :--- |
| **$m_1$** | Baseline Direct Effect | $\widetilde{\text{RobReplied}}$ |
| **$m_2$** | Gender Heterogeneity | $\widetilde{\text{RobReplied}},\, \widetilde{\text{Female} \times \text{RobReplied}}$ |
| **$m_3$** | Verification Interaction | $\widetilde{\text{Rob}},\, \text{Verified} \times \widetilde{\text{Rob}},\, \widetilde{\text{FemRob}},\, \text{Verified} \times \widetilde{\text{FemRob}}$ |
| **$m_4$** | Temporal Moderation | $\widetilde{\text{Rob}},\, \text{Time} \times \widetilde{\text{Rob}},\, \widetilde{\text{FemRob}},\, \text{Time} \times \widetilde{\text{FemRob}}$ |
| **$m_5$** | Sex Ratio at Birth (SRB) | $\widetilde{\text{Rob}},\, \text{SRB} \times \widetilde{\text{Rob}},\, \widetilde{\text{FemRob}},\, \text{SRB} \times \widetilde{\text{FemRob}}$ |
| **$m_6$** | Mainland Residency Moderation | $\widetilde{\text{Rob}},\, \text{Mainland} \times \widetilde{\text{Rob}},\, \widetilde{\text{FemRob}},\, \text{Mainland} \times \widetilde{\text{FemRob}}$ |
| **$m_7$** | Poster Age Interaction | $\widetilde{\text{Rob}},\, \text{Age} \times \widetilde{\text{Rob}},\, \widetilde{\text{FemRob}},\, \text{Age} \times \widetilde{\text{FemRob}}$ |
| **$m_8$** | Bot Comment Sentiment Moderation | $\widetilde{\text{Rob}},\, \text{BotSent} \times \widetilde{\text{Rob}},\, \widetilde{\text{FemRob}},\, \text{BotSent} \times \widetilde{\text{FemRob}}$ |
| **$m_{\text{sub}}$** | Subgroup Analyses | Split by temporal phases ($0\text{--}16\%$, $16\text{--}50\%$, $50\text{--}100\%$) |
| **$s_1\text{--}s_4$** | Sentiment as Dependent Variable | Effects on average clean comment sentiment |

*(Note: $\widetilde{X}$ denotes cross-fitted residuals obtained via Robinson's transformation).*

---

##  Outputs & Artifacts

- **Model Results**: Saved as structured JSON objects in [`results/`](file:///c:/Users/ge27tuv/Projects/Doubel-Machine-Learning/results/) containing estimated coefficients, standard errors, $p$-values, and $R^2$ statistics.
- **Robustness Check Logs**: Detailed comparisons stored in [`results/robustness/case1_dim150_fold10.json`](file:///c:/Users/ge27tuv/Projects/Doubel-Machine-Learning/results/robustness/case1_dim150_fold10.json) and [`results/robustness/case2_dim100_fold5.json`](file:///c:/Users/ge27tuv/Projects/Doubel-Machine-Learning/results/robustness/case2_dim100_fold5.json).
- **Summary Tables**: [`results/robustness/descriptive_statistics.csv`](file:///c:/Users/ge27tuv/Projects/Doubel-Machine-Learning/results/robustness/descriptive_statistics.csv) and [`results/robustness/correlation_matrix.csv`](file:///c:/Users/ge27tuv/Projects/Doubel-Machine-Learning/results/robustness/correlation_matrix.csv).

---

##  References & Data Source

- **Dataset**: CommentR Interaction Dataset available on [Zenodo (Record 16921462)](https://zenodo.org/records/16921462).
- **Double Machine Learning Framework**: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters*. The Econometrics Journal, 21(1), C1-C68.
- **Sentiment Model**: [`IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment).
