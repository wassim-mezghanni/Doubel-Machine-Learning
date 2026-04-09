# DML Diffusion Analysis: Summary Report

This table summarizes the causal effects of robot interactions and user gender across three stages of technology diffusion.

| Group                   | Target   | Variable                 |         Coef | Significance   |          R2 |
|:------------------------|:---------|:-------------------------|-------------:|:---------------|------------:|
| Early Adopters (0-16%)  | Likes    | Robot Reply              |    0.773144  | ns             | 2.23068e-05 |
| Early Adopters (0-16%)  | Likes    | Interaction (Female*Rob) |    0.042661  | ***            | 0.0131589   |
| Early Adopters (0-16%)  | Likes    | Female                   |   -0.396218  | ns             | 3.69158e-06 |
| Early Adopters (0-16%)  | Comments | Female                   |   -0.466564  | ns             | 5.84959e-07 |
| Early Adopters (0-16%)  | Comments | Interaction (Female*Rob) |    0.654732  | ***            | 0.353582    |
| Early Majority (16-50%) | Likes    | Robot Reply              |   13.0254    | *              | 3.42837e-05 |
| Early Majority (16-50%) | Likes    | Interaction (Female*Rob) |  -44.2587    | ***            | 0.921479    |
| Early Majority (16-50%) | Likes    | Female                   |   -5.11382   | ns             | 3.52574e-06 |
| Early Majority (16-50%) | Comments | Female                   |  -12.0223    | ns             | 3.55819e-06 |
| Early Majority (16-50%) | Comments | Interaction (Female*Rob) | -103.526     | ***            | 0.920748    |
| Late Majority (50-100%) | Likes    | Robot Reply              |    0.0981369 | ***            | 0.00262616  |
| Late Majority (50-100%) | Likes    | Interaction (Female*Rob) |   -0.0111038 | ***            | 0.0031202   |
| Late Majority (50-100%) | Likes    | Female                   |   -0.0378701 | ***            | 0.000413271 |
| Late Majority (50-100%) | Comments | Female                   |    1.57413   | ***            | 0.000198388 |
| Late Majority (50-100%) | Comments | Interaction (Female*Rob) |   -0.456003  | ***            | 0.000436744 |

*Note: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant.*