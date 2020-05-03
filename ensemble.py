import pandas as pd

# load submissions
model_1 = pd.read_csv('D:/Data/ISIC-2019/submissions/final-efficient-net-b3-baseline-v1-augment-no-augment-450-LB-540.csv', index_col='image')
model_2 = pd.read_csv('D:/Data/ISIC-2019/submissions/final-efficient-net-b3-baseline-v3-augment-no-augment-450-LB-537.csv', index_col='image')
model_3 = pd.read_csv('D:/Data/ISIC-2019/submissions/final-efficient-net-b3-baseline-v4-augment-no-augment-450-LB-534.csv', index_col='image')
model_4 = pd.read_csv('D:/Data/ISIC-2019/submissions/final-efficient-net-b3-baseline-v5-augment-no-augment-450-LB-537.csv', index_col='image')
model_5 = pd.read_csv('D:/Data/ISIC-2019/submissions/final-efficient-net-b4-baseline-v1-augment-no-augment-450-LB-544.csv', index_col='image')
model_6 = pd.read_csv('D:/Data/ISIC-2019/submissions/final-efficient-net-b4-baseline-v10-pseudo-no-augment-450-LB-540.csv', index_col='image')

# combine submissions
submission = model_1 + model_2 + model_3 + model_4 + model_5 + model_6
submission = submission / 2

# save new submission
submission.to_csv('D:/Data/ISIC-2019/submissions/final-ensemble-B4.csv', index=True)
