import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins
from plotly.subplots import make_subplots


penguins = load_penguins()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color= "species"
).show()

fig_subplot = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Adeli","Gentoo","Chinstrap")
)
 
fig_subplot.add_trace(
    {
        'type' : 'scatter',
        'mode' : 'markers',
        'x': penguins.query('species=="Adelie"')["bill_length_mm"],
        'y':penguins.query('species=="Adelie"')["bill_depth_mm"],
        'name': "Adelie"
    },
    row=1, col=1
)

fig_subplot.add_trace(
    {
        'type' : 'scatter',
        'mode' : 'markers',
        'x': penguins.query('species=="Gentoo"')["bill_length_mm"],
        'y':penguins.query('species=="Gentoo"')["bill_depth_mm"],
        'name': "Gentoo"
    },
    row=1, col=2
)

fig_subplot.add_trace(
    {
        'type' : 'scatter',
        'mode' : 'markers',
        'x': penguins.query('species=="Chinstrap"')["bill_length_mm"],
        'y':penguins.query('species=="Chinstrap"')["bill_depth_mm"],
        'name': "Chinstrap"
    },
    row=1, col=3
)

fig_subplot.update_layout(
    title=dict(text="펭귄종별 부리길이 vs 깊이",
               x=0.5)
)


# 함수 도움말 보는법: 함수이름 뒤에 ? 붙여서 실행
# make_subplots?

import plotly.express as px
from palmerpenguins import load_penguins

# 펭귄 데이터 로드
penguins = load_penguins()

# scatter plot 생성 (서브플롯 자동 생성)
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    facet_col="species"  # 이 옵션으로 서브플롯 생성
)

# 그래프 보여주기
fig.show()
