import pandas as pd

# load submissions
baseline = pd.read_csv('D:/Data/ISIC-2019/submissions/efficient-net-b1-baseline-v3.csv', index_col='image')

efn_b3 = pd.read_csv('D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v1-467.csv', index_col='image')

efn_b3_v1_augment = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v1-augment.csv', index_col='image')

efn_b3_v2_augment = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v2-augment-502.csv', index_col='image')

efn_b3_v3_augment = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v3-augment-508.csv', index_col='image')

efn_b3_v4_subset = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v4-augment-521.csv', index_col='image')

efn_b3_v5_subset = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v5-augment-493.csv', index_col='image')

efn_b4_v1_subset = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b4-baseline-v1-subset-504.csv', index_col='image')

efn_b4_v2_subset = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b4-baseline-v2-augment-502.csv', index_col='image')

efn_b4_v2_pseudo_450 = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b4-baseline-v3-pseudo-450-513.csv', index_col='image')

# combine submissions
# submission = efn_b3_v1_augment + efn_b3_v2_augment + efn_b3_v3_augment + efn_b3_v4_subset + efn_b4_v1_subset + efn_b4_v2_subset
# submission = submission / 6

submission = efn_b3_v3_augment + efn_b3_v4_subset + efn_b4_v2_pseudo_450
submission = submission / 3

# save new submission
submission.to_csv('D:/Data/ISIC-2019/submissions/ensemble-450-v1-augment.csv', index=True)
