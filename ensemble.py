import pandas as pd

# load submissions
# baseline = pd.read_csv('D:/Data/ISIC-2019/submissions/efficient-net-b1-baseline-v3.csv', index_col='image')
efn_b3 = pd.read_csv('D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v1.csv', index_col='image')

efn_b3_v1_augment = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v1-augment.csv', index_col='image')

efn_b3_v2_augment = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v2-augment.csv', index_col='image')

efn_b3_v3_augment = pd.read_csv(
    'D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v3-augment.csv', index_col='image')

# combine submissions
submission = efn_b3 + efn_b3_v1_augment + efn_b3_v2_augment + efn_b3_v3_augment
submission = submission / 4

# save new submission
submission.to_csv('D:/Data/ISIC-2019/submissions/ensemble-v3-augment-v123.csv', index=True)
