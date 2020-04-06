import pandas as pd

# load submissions
baseline = pd.read_csv('D:/Data/ISIC-2019/submissions/efficient-net-b1-baseline-v3.csv', index_col='image')
efn_b3 = pd.read_csv('D:/Data/ISIC-2019/submissions/efficient-net-b3-baseline-v1.csv', index_col='image')

# combine submissions
submission = baseline + efn_b3
submission = submission / 2

# save new submission
submission.to_csv('D:/Data/ISIC-2019/submissions/ensemble-v1.csv', index=True)
