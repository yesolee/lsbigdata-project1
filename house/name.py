import pandas as pd

# https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/
# edit?gid=158309216#gid=158309216
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2")
df.head()

# 랜덤하게 2명을 뽑아서 보여주는 코드
import numpy as np

np.random.seed(20240730)
np.random.choice(df['이름'],2,replace=False)

