# 11-1
import numpy as np

import json
geo = json.load(open('data/bigfile/SIG.geojson', encoding='UTF-8'))
geo['features'][0]['properties']
geo['features'][0]['geometry']

import pandas as pd
df_pop = pd.read_csv('data/Population_SIG.csv')
df_pop.head()
df_pop.info()
df_pop['code'] = df_pop['code'].astype(str)

# !pip install folium
# !pip install -U numpy
import folium
my_map = folium.Map(location=[35.95,127.7],zoom_start=8, tiles='cartodbpositron')

bins = list(df_pop['pop'].quantile([0,0.2,0.4,0.6,0.8,1]))
bins

my_map = folium.Map(location=[35.95,127.7],zoom_start=8, tiles='cartodbpositron')
folium.Choropleth(
    geo_data = geo,
    data= df_pop,
    columns=('code','pop'),
    key_on='feature.properties.SIG_CD',
    fill_color = 'YlGnBu',
    fill_opacity=1,
    line_opacity=0.5,
    bins = bins)\
    .add_to(my_map)
my_map

my_map.save('my_map.html')

import webbrowser
webbrowser.open_new('my_map.html')


# 11-2
geo_seoul = json.load(open('data/EMD_Seoul.geojson', encoding='UTF-8'))
geo_seoul['features'][0]['properties']
geo_seoul['features'][0]['geometry']

foreigner = pd.read_csv('data/Foreigner_EMD_Seoul.csv')
foreigner.head()
foreigner.info()

foreigner['code'] = foreigner['code'].astype(str)
foreigner.info()

bins = list(foreigner['pop'].quantile([0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1]))
bins

map_seoul = folium.Map(location=[37.56,127],
                        zoom_start=12,
                        tiles='cartodbpositron')

folium.Choropleth(
    geo_data=geo_seoul,
    data= foreigner,
    columns = ('code', 'pop'),
    key_on = 'feature.properties.ADM_DR_CD',
    fill_color= 'Blues',
    nan_fill_color = 'white',
    fill_opacity = 1,
    line_opacity = 0.5,
    bins = bins)\
    .add_to(map_seoul)
map_seoul    
    
geo_seoul_sig = json.load(open('data/SIG_Seoul.geojson', encoding='UTF-8'))    
folium.Choropleth(geo_data=geo_seoul_sig,
                  fill_opacity=0,
                  line_weight=4)\
      .add_to(map_seoul)
map_seoul
    
map_seoul.save('map_seoul.html')
webbrowser.open_new('map_seoul.html')

### houseprice

house = pd.read_csv('data/houseprice-with-lonlat.csv')
house=house.rename(columns={'Unnamed: 0':'Index'})

map_house = folium.Map(location=[house['Latitude'].mean(),house['Longitude'].mean()],zoom_start=13,tiles="cartodb positron")

house_locations = house[['Index','Longitude','Latitude']]
house_locations

for i, x in house_locations.iterrows():
    folium.Marker([x["Latitude"], x["Longitude"]], popup=x["Index"]).add_to(map_house)
    
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')

########################################

import json
import matplotlib.pyplot as plt
    
geo_seoul = json.load(open("data/bigfile/SIG_Seoul.geojson"))

type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul['features']
len(geo_seoul['features'])
len(geo_seoul['features'][0])
type(geo_seoul['features'][0])
geo_seoul['features'][0].keys() #딕셔너리 안에 딕셔너리
geo_seoul['features'][0]['properties']
geo_seoul['features'][1]['properties']
geo_seoul['features'][2]['properties']

type(geo_seoul['features'][0]['geometry']) # 여전히 딕셔너리

df = pd.read_csv('data/Population_SIG.csv')
df
# df.query('region == )
len(geo_seoul['features'][0]['geometry']['coordinates'][0][0])
# 지도 그리기 함수 만들기
def draw_seoul(num):
    gu_name=geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_array=np.array(coordinate_list[0][0])
    x= coordinate_array[:,0]
    y= coordinate_array[:,1]

    # plt.plot(x[::5],y[::5]) 
    # 점의 갯수가 많아지면 성능이 떨어짐
    # 점이 너무 적으면 모양이 둥글어지는 등 해상도가 떨어짐
    plt.plot(x,y)
    plt.rcParams.update({'font.family':'Malgun Gothic'})
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None
    
draw_seoul(11)

# 서울시 전체 지도를 그려보자!
# 1) 데이터 프레임 만들기
## gu_name | x | y
## ===============
## 종로구  |126| 36
## 종로구  |126| 37
## 중구    |120| 30
## plt.plot(x,y, hue="gu_name")


seoul=pd.DataFrame(columns=["gu_name","x","y"])
seoul

for i in range( len(geo_seoul['features']) ):
    gu_name= geo_seoul['features'][i]['properties']['SIG_KOR_NM']
    coordinate_list = geo_seoul['features'][i]['geometry']['coordinates']
    coordinate_array=np.array(coordinate_list[0][0])
    x= coordinate_array[:,0]
    y= coordinate_array[:,1]
    new_row = pd.DataFrame({
        "gu_name": [gu_name]*len(x),
        "x": x,
        "y": y
    })
    
    seoul = pd.concat([seoul, new_row], ignore_index=True)

seoul

import seaborn as sns
sns.lineplot(data=seoul, x='x',y='y', hue='gu_name')
plt.legend(prop={'size': 2})

plt.show()
plt.clf()

###########

# 구 이름 만들기
# 방법1
gu_name= []
for i in range(25):
    gu_name.append(geo_seoul['features'][i]['properties']['SIG_KOR_NM'])
gu_name

# 방법2 
gu_name2 = [geo_seoul['features'][i]['properties']['SIG_KOR_NM'] for i in range(25)]
gu_name2

# x, y 판다스 데이터 프레임
def make_seouldf(num):
    gu_name=geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list = geo_seoul['features'][num]['geometry']['coordinates']
    coordinate_array=np.array(coordinate_list[0][0])
    x= coordinate_array[:,0]
    y= coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x":x, "y":y})
    

make_seouldf(12)

seoul_df = pd.DataFrame()
for i in range(25):
    seoul_df = pd.concat([seoul_df, make_seouldf(i)],ignore_index=True) 
seoul_df['is_gangnam'] = np.where(seoul_df['gu_name'] == "강남구","강남","안강남")
seoul_df
## 그림 그리기
# seoul_df.plot(kind='scatter', x='x', y='y', style='o',s=1)

sns.scatterplot(data=seoul_df, x='x', y='y', hue='is_gangnam', palette={'안강남':'grey','강남':'red'}, s=1, legend=False)
plt.show()
plt.clf()

seoul_df['is_gangnam'].unique()

import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open('data/bigfile/SIG_seoul.geojson', encoding="UTF-8"))
geo_seoul['features'][0]['properties']

df_pop = pd.read_csv('data/Population_SIG.csv')
# 서울시 중 동별 인구수 조회
df_seoulpop = df_pop.iloc[1:26]
# df_pop['code']<20000
 
df_seoulpop['code'] = df_seoulpop['code'].astype(str)
df_seoulpop.info()

# 패키지 설치하기
# !pip install folium
import folium
seoul_df['x'].mean()
seoul_df['y'].mean()

map_sig = folium.Map(location=[37.55180997129064,126.97315486480478 ],
                    zoom_start=12, tiles='cartodbpositron')

geo_seoul['features'][0]['properties']['SIG_CD']
# 코로플릿
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns=('code', 'pop'),
    key_on = 'feature.properties.SIG_CD'
).add_to(map_sig)

# quantile 0.5면 중앙값
bins= list(df_pop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

# 코로플릿 with bins
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns=('code', 'pop'),
    bins = bins,
    fill_color='viridis',
    key_on = 'feature.properties.SIG_CD'
).add_to(map_sig)


make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744,126.983800], popup='종로구').add_to(map_sig)

map_sig.save('map_seoul2.html')

import webbrowser
webbrowser.open_new('map_seoul2.html')

df_seoulpop.sort_values('pop', ascending=False)

#####################
#houseprice로 마커 표시하기
# from folium.plugins import MarkerCluster 쓰는법 찾아보기
import pandas as pd
df = pd.read_csv('data/houseprice-with-lonlat.csv')
df.columns
df[['Longitude', 'Latitude']].mean()

map_house = folium.Map(location=[42.034482,-93.642897 ],
                    zoom_start=13, tiles='cartodbpositron')
Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']

# zip을 쓰면 좀 더 깔끔하게 된다.
for i in range(len(Longitude)):
    folium.CircleMarker([Latitude[i], Longitude[i]],
                        popup=f"Price: ${Price[i]}",
                        radius=3, # 집의 면적으로 표현해보기
                        color='skyblue', 
                        fill_color='skyblue',
                        fill=True, 
                        fill_opacity=0.6 ).add_to(map_house)

map_house.save('map_house.html')
webbrowser.open_new('map_house.html')










