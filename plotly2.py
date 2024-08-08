# 128p
import plotly.express as px
import pandas as pd
from palmerpenguins import load_penguins

# 데이터를 불러옵니다
penguins = load_penguins()

# 산포도를 생성합니다
fig = px.scatter(
    penguins, 
    x="bill_length_mm", 
    y="bill_depth_mm", 
    color="species",
    trendline="ols", # p.134
    size_max=20  # 점의 최대 크기 설정
)

# 레이아웃을 업데이트합니다
fig.update_layout(
    title=dict(
        text="팔머펭귄 종별 부리 길이 vs 깊이",
        font=dict(size=20, color='white')  # 제목 크기와 색상 설정
    ),
    paper_bgcolor="black", 
    plot_bgcolor="black",
    font=dict(color='white'),
    xaxis=dict(
        title="부리 길이 (mm)",  # x축 레이블
        gridcolor='rgba(255, 255, 255, 0.2)'
    ),
    yaxis=dict(
        title="부리 깊이 (mm)",  # y축 레이블
        gridcolor='rgba(255, 255, 255, 0.2)'
    ),
    legend=dict(
        title=dict(
            text="종",
            font=dict(color='white')
        ),
        traceorder="normal",  # 범례 항목의 순서
        orientation="v",     # 범례의 방향
        title_font_size='16px'  # 범례 제목의 폰트 크기 조정
    )
)

# 점의 크기와 투명도를 조절합니다
fig.update_traces(
    marker=dict(
        size=12,      # 점의 크기 조정
        opacity=0.6   # 점의 투명도 조정
    )
)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

penguins = penguins.dropna()
x=penguins[["bill_length_mm"]]
y=penguins["bill_depth_mm"]

model.fit(x,y)
linear_fit = model.predict(x)

fig.add_trace(
    go.Scatter(
        mode="lines",
        x=penguins["bill_length_mm"],
        y=linear_fit,
        name="선형회귀직선",
        line=dict(dash="dot",color="yellow"),
    )
)

fig.show()

model.coef_
model.intercept_

# 심슨's 패러독스
# lurking variable 잠복변수를 주의해야 한다. 트렌드 자체가 변할 수 있기 때문에

# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환 => 문자를 숫자(True, False)로 변환환
penguins_dummies = pd.get_dummies(penguins, columns=['species'], drop_first=True)
penguins_dummies.iloc[:,-3:]

x=penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y=penguins_dummies["bill_depth_mm"]

model = LinearRegression()
model.fit(x,y)

model.coef_
model.intercept_

y = mode.coef_[0] * bill_length + mode.coef_[1] * species_Chinstrap + mode.coef_[2] * species_Gentoo + model.intercept
# 인덱스 1번 행의 펭귄: 아델리 종의 y값 예측
# Adelie, 0, 0
# y = mode.coef_[0] * 39.5 + mode.coef_[1] * 0+ mode.coef_[2] * 0+ model.intercept
# chinstrap이라면 40.5, 1,0
# y = mode.coef_[0] * 40.5 + mode.coef_[1] * 1+ mode.coef_[2] * 0+ model.intercept

regline_y = model.predict(x)
regline_y

import matplotlib.pyplot as plt
sns.scatterplot(x['bill_length_mm'], y, color="black")

sns.scatterplot(x['bill_length_mm'],regline_y, hue=penguins['species'])    

plt.show()
plt.clf()


























