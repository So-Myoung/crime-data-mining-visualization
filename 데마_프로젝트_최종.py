#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pylab as plt
import plotly.express as px
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import font_manager, rc
import folium
from folium.plugins import MarkerCluster
import json
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point


# In[2]:


import warnings
import platform

# seaborn style 설정
sns.set(style='whitegrid')
sns.set_palette('pastel')
warnings.filterwarnings('ignore')

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

if platform.system() == 'Windows':
# 윈도우인 경우
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

mpl.rcParams['axes.unicode_minus'] = False 


# ### 전국 연도별 범죄 발생 및 검거 건수

# In[3]:


df = pd.read_csv('[범죄(상세죄종)] 연도별 범죄 발생 건수와 검거 건수(2012_) _ 총합 (2021년).csv', encoding='UTF-8' )
df


# In[4]:


x = df['날짜']
total = df["발생(건)"]
total2 = df["검거(건)"]

plt.bar(x,total, color = 'b', alpha = 0.35, label='발생')
plt.plot(x,total2, color = 'r', alpha = 0.9, label='검거')

plt.xlabel('연도별')
plt.ylabel('발생(건)')
plt.title('전국 연도별 범죄 발생 및 검거 건수')

plt.xticks(x)
plt.legend()
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])


# ### 신고 접수 대비 출동 건수

# In[5]:


df1 = pd.read_excel('stat_160901.xls')
df2 = df1.T
df2 = df2.filter([0,2,3])
df2 = df2.T

x = df2.columns[1:]
y = df2.iloc[0][1:]
y1 = df2.iloc[1][1:]

y = list(y.values)
y1 = list(y1.values)

plt.bar(x,y,color = '#17becf',label='신고 접수 건수')
plt.plot(x,y1,color ='r',label = '총 출동 건수')
plt.legend()
plt.xlabel('연도별')
plt.ylabel('접수건수')
plt.title('신고 접수 대비 출동 건수')


# ### 전국 범죄건수

# In[6]:


df = pd.read_csv('./2021 전국 지역별 범죄율.csv')
df


# In[7]:


df = df.drop(index=0, axis=0)
crime = df['발생(건)']
label = df['지역']

plt.rcParams['figure.figsize']=[7,7]
plt.title('2021 전국 범죄건수', size=15)
plt.pie(crime, labels=label, shadow=True, autopct='%.1f%%', 
        startangle = 90, textprops = {'fontsize':9},
        explode=[0,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
        wedgeprops = {'width':0.7,'edgecolor':'w','linewidth':3})
plt.show()


# ### 전국 인구대비 범죄율

# In[8]:


df = pd.read_csv('./2021 전국 지역별 범죄율.csv')
df['범죄율'] = (df['발생(건)'] / df['주민등록인구(명)'])*100
new_df = df[['날짜','지역','범죄율']]
new_df


# In[9]:


# 전국 범죄율 plot 차트로 표현
crime_rate = new_df['범죄율']
label = new_df['지역']
plt.plot(label,crime_rate,
         color='black', # color : 색깔 지정
         alpha=0.6,
         linestyle='--',
         marker='o')
#plt.bar(label,crime_rate,color='#9990BA')
plt.xticks(size=10, rotation=90)
plt.title('2021 인구대비 전국 범죄율', size=15)


# In[10]:


# 전국 범죄율 bar 차트로 표현
plt.bar(label,crime_rate)
plt.xticks(size=10,rotation=90)
plt.title('2021 인구대비 전국 범죄율', size=15)
plt.show()


# ### 경기도 북부 인구대비 범죄율

# In[11]:


df2 = pd.read_csv('./2021 전국 범죄 발생 건수 및 종류.csv', encoding='cp949')
gyeonggi_population = pd.read_csv('./2021 경기도 시군별 세대 및 인구.csv', encoding='cp949')


# In[12]:


pop_north = [44,52,48,54,51,49,50,53]
north_pop_df = gyeonggi_population.iloc[pop_north,:2]
north_pop_df.columns = ['지역','인구수']
north_pop_df.set_index('지역', inplace=True, drop=True)

north_pop = gyeonggi_population.iloc[pop_north,1]
north_pop = north_pop.str.replace(',', '')
north_pop = north_pop.astype(int)
north_pop = list(north_pop)

print('경기도 북부 인구수 data frame 형태로 출력')
north_pop_df


# In[13]:


north_crimerate_df = df2.iloc[[3,12,13],[0,11,15,18,19,27,32,34,36]]
north_crimerate_df.columns = ['죄종','고양시','구리시','남양주시','동두천시','양주시','의정부시','파주시','포천시']
north_crimerate_df.set_index('죄종', inplace=True, drop=True)

print('경기도 북부 범죄건수 data frame 형태로 출력')
north_crimerate_df


# In[14]:


d = '#AED4FF'
colors = [d,d,'silver',d,d,'#ff9999',d,d]
crimerate_north = [11,15,18,19,27,32,34,36]
north_label = ['고양시','구리시','남양주시','동두천시','양주시','의정부시','파주시','포천시']

north_violent = df2.iloc[3,crimerate_north].astype(int)
north_theft = df2.iloc[12,crimerate_north].astype(int)
north_violence = df2.iloc[13,crimerate_north].astype(int)

north_crimerate=[]

for i in range(0,8):
    north_result = (north_violent[i] + north_theft[i] + north_violence[i]) / north_pop[i] * 100
    north_crimerate.append(north_result)
    
plt.figure(figsize=(10,10))
plt.bar(north_label,north_crimerate,
        width=0.4,
        color=colors
       ) 
plt.title('2021 경기도 북부 인구대비 범죄율(4대 범죄)', size=20)

dic = { y:x for x, y in zip(north_label, north_crimerate) }

plt.text(dic[min(north_crimerate)],min(north_crimerate),
         '최저' + str(round(min(north_crimerate),3)) + '%',
         horizontalalignment='center',
         verticalalignment='bottom')

plt.text(dic[max(north_crimerate)],max(north_crimerate),
         '최고' + str(round(max(north_crimerate),3)) + '%',
         horizontalalignment='center',
         verticalalignment='bottom')


# In[15]:


north_df = pd.DataFrame({'지역':north_label,
                         '범죄율':north_crimerate})

#인구 대비 범죄율 내림차순 정렬
north_df = north_df.sort_values('범죄율', ascending = False)

rank=[]
for i in range(0,len(north_df.index)):
    rank.append(str(i+1)+"위")

north_df.index = rank
north_df


# ### 경기도 남부 인구대비 범죄율

# In[16]:


pop_south = [39,30,31,32,29,19,15,6,28,36,24,38,34,11,37,33,27,35,23]
south_pop_df = gyeonggi_population.iloc[pop_south,:2]
south_pop_df.columns = ['지역','인구수']
south_pop_df.set_index('지역', inplace=True, drop=True)

south_pop = gyeonggi_population.iloc[pop_south,1]
south_pop = south_pop.str.replace(',', '')
south_pop = south_pop.astype(int)
south_pop = list(south_pop)

print('경기도 남부 인구수 data frame 형태로 출력')
south_pop_df


# In[17]:


south_crimerate_df = df2.iloc[[3,12,13],[0,12,13,14,16,17,20,21,22,23,25,26,28,29,30,31,33,35,37,38]]
south_crimerate_df.columns = ['죄종','과천시','광명시','광주시','군포시','김포시','부천시','성남시',
                              '수원시','시흥시','안성시','안양시','여주시','오산시','용인시','의왕시',
                              '이천시','평택시','하남시','화성시']
south_crimerate_df.set_index('죄종', inplace=True, drop=True)

print('경기도 남부 범죄건수 data frame 형태로 출력')
south_crimerate_df


# In[18]:


d = '#AED4FF'
colors = [d,d,d,d,d,d,d,d,d,d,d,d,d,d,'silver',d,'#ff9999',d,d]
crimerate_south = [12,13,14,16,17,20,21,22,23,25,26,28,29,30,31,33,35,37,38]
south_label = ['과천시','광명시','광주시','군포시','김포시','부천시','성남시','수원시',
   '시흥시','안성시','안양시','여주시','오산시','용인시','의왕시',
   '이천시','평택시','하남시','화성시']

south_violent = df2.iloc[3,crimerate_south].astype(int)
south_theft = df2.iloc[12,crimerate_south].astype(int)
south_violence = df2.iloc[13,crimerate_south].astype(int)

south_crimerate=[]

for i in range(0,19):
    south_result = (south_violent[i] + south_theft[i] + south_violence[i]) / south_pop[i] * 100
    south_crimerate.append(south_result)
    
plt.figure(figsize=(15,10))
plt.bar(south_label,south_crimerate,
        width=0.4,
        color = colors
       ) 
plt.title('2021 경기도 남부 인구대비 범죄율(4대 범죄)', size=20)

dic = { y:x for x, y in zip(south_label, south_crimerate) }

plt.text(dic[min(south_crimerate)],min(south_crimerate),
         '최저' + str(round(min(south_crimerate),3)) + '%',
         horizontalalignment='center',
         verticalalignment='bottom')

plt.text(dic[max(south_crimerate)],max(south_crimerate),
         '최고' + str(round(max(south_crimerate),3)) + '%',
         horizontalalignment='center',
         verticalalignment='bottom')


# In[19]:


south_df = pd.DataFrame({'지역':south_label,
                         '범죄율':south_crimerate})

#인구 대비 범죄율 내림차순 정렬
south_df = south_df.sort_values('범죄율', ascending = False)

rank=[]
for i in range(0,len(south_df.index)):
    rank.append(str(i+1)+"위")

south_df.index = rank
south_df


# ## 경기도 4대 범죄율 1위 지역

# In[20]:


crime_rate = []
crime_rate.append(max(north_crimerate))
crime_rate.append(max(south_crimerate))

plt.figure(figsize=(3,5))
plt.bar(['의정부시','평택시'],crime_rate,width=0.3, color=['#AED4FF','#ff9999'])
plt.title('2021 경기도 인구대비 범죄율(4대 범죄)', size=15)

dic = { y:x for x, y in zip(['의정부시','평택시'], crime_rate) }

plt.text(dic[max(crime_rate)],max(crime_rate),
         str(round(max(crime_rate),3)) + '%',
         horizontalalignment='center',
         verticalalignment='bottom')


# In[21]:


crimerate = pd.DataFrame({'지역':north_label+south_label,
                             '범죄율':north_crimerate + south_crimerate})

#인구 대비 범죄율 내림차순 정렬
crimerate = crimerate.sort_values('범죄율', ascending = False)

rank=[]
for i in range(0,27):
    rank.append(str(i+1)+"위")

crimerate.index = rank
crimerate


# ### 경기도 발생장소별 4대 범죄 건수

# In[22]:


df3 = pd.read_csv('2021 경기도 범죄 발생장소.csv', encoding='cp949')

crimescene_violent = df3.iloc[2,3:].astype(int)
crimescene_theft = df3.iloc[11,3:].astype(int)
crimescene_violence = df3.iloc[12,3:].astype(int)
crimescene_label = df3.iloc[0,3:]
crimescene_sum=[]


for i in range(0,len(crimescene_label)):
    result = crimescene_violent[i] + crimescene_theft[i] + crimescene_violence[i]
    crimescene_sum.append(result)

#plt.title('2021 경기도 발생장소별 4대 범죄 건수 조사', size=15)    
#plt.pie(crimescene_sum, labels=piechart_label, shadow=True, startangle = 90, textprops = {'fontsize':9})
#plt.show()


# In[23]:


plt.figure(figsize=(15,10))
plt.bar(crimescene_label,crimescene_sum,color='#9990BA',alpha=0.8)
plt.xticks(size=10,rotation=90)
plt.title('2021 경기도 발생장소 별 범죄 건수(4대 범죄)', size=20)

dic = { y:x for x, y in zip(crimescene_label,crimescene_sum) }

plt.text(dic[max(crimescene_sum)],max(crimescene_sum),
         str(round(max(crimescene_sum),3)) + '건',
         horizontalalignment='center',
         verticalalignment='bottom')


# In[24]:


crimescene = pd.DataFrame({'범죄 발생장소':crimescene_label,
                             '건수':crimescene_sum})

#인구 대비 범죄율 내림차순 정렬
crimescene = crimescene.sort_values('건수', ascending = False)

rank=[]
for i in range(0,len(crimescene_label)):
    rank.append(str(i+1)+"위")

crimescene.index = rank
crimescene


# In[25]:


crimescene.set_index('범죄 발생장소', inplace=True, drop=True)
crimescene_num = crimescene['건수']
etc_sum = 0

for i in range(6,len(crimescene_label)):
    etc_sum += crimescene_num[i].astype(int)

crimescene = crimescene.iloc[:6,:]    
crimescene_num[3] = crimescene_num[3].astype(int) + etc_sum
crimescene


# In[26]:


plt.rcParams['figure.figsize']=[4.5,4.5]
plt.title('2021 경기도 발생장소 별 범죄 건수(4대 범죄)', size=15)    
plt.pie(crimescene['건수'], 
        labels=crimescene.index, 
        shadow=True, 
        startangle = 90, 
        autopct='%.1f%%', 
        textprops = {'fontsize':9},
        explode=[0.05,0.05,0.05,0.05,0.05,0.05], 
        colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0','#AED4FF','#02E1D9'],
        wedgeprops = {'width':0.7,'edgecolor':'w','linewidth':3})
plt.show()


# ### 평택 세대수 현황

# In[27]:


number_df = pd.read_csv('읍·면·동별_세대_및_인구_20221121182116.csv', encoding = 'cp949')


# In[28]:


number_df = pd.DataFrame({'동':number_df.iloc[3:,0],
                          '세대수':number_df.iloc[3:,1]})

number = number_df['세대수'].astype(int)
number[22] = number[22]+number[23]
number_df['세대수'] = number

number_label = number_df['동'].astype(str)
number_label.loc[22] = '비전동'
number_df['동'] = number_label

number_df = number_df.drop(index=23, axis=0)
number_df


# In[29]:


d = '#8fd9b6'
colors = [d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,'#ff9999',d,d]
plt.figure(figsize=(6,4))

plt.bar(number_df['동'],number_df['세대수'],color=colors)
plt.xticks(size=10,rotation=90)
plt.xlabel('동')
plt.ylabel('세대수')
plt.title('2021 경기도 평택시 세대수', size=15) 

current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])


# In[30]:


number_df

number_df = number_df.sort_values('세대수', ascending = False)

rank=[]
for i in range(0,len(number_df.index)):
    rank.append(str(i+1)+"위")

number_df.index = rank
number_df


# In[31]:


pol = pd.read_csv('경기도지구대(파출소)현황.csv', encoding='cp949')
pol1 = pol.filter(['시군명','WGS84위도','WGS84경도'])
# data1 = data1[:100]
pol2 = pol1[pol1['시군명']=='평택시']
pol2 = pol2.reset_index(drop=True)
pol2


# ### 비전동 비상벨, 경찰서 위치

# In[32]:


df3 = pd.read_csv('12_04_09_E_안전비상벨위치정보.csv',encoding='cp949')
geo_data = json.load(open('./LSMD_ADM_SECT_UMD_41.json', encoding='UTF8'))
map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)
pt_111 = df3[df3['소재지지번주소'].str.contains('비전')]
pt_112 = df3[df3['소재지도로명주소'].str.contains('비전')]
pt_119 = pd.concat([pt_111,pt_112])


# In[33]:


map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)

for i in range(0,len(pt_119)):
    lat = pt_119.iloc[i, 7]
    lng = pt_119.iloc[i, 8]
    marker= folium.Marker([lat,lng],icon=folium.Icon(color='red')).add_to(map_osm)
    
for i in range(0,len(pol2.index)):
    police_lat = pol2.loc[i, 'WGS84위도']
    police_lng = pol2.loc[i, 'WGS84경도']    
    folium.Marker([police_lat, police_lng], 
                  icon=folium.Icon(color='darkblue')).add_to(map_osm)
    
folium.Choropleth(geo_data=geo_data,fill_opacity=0).add_to(map_osm)

map_osm


# In[34]:


df = gpd.read_file('./TL_SCCO_EMD.shp',encoding='euc-kr')
set = df[df['EMD_CD'].str.contains('41220118')]
pt_110 = pd.read_csv('./12_04_09_E_안전비상벨위치정보.csv',encoding='cp949')
pt_111 = pt_110[pt_110['소재지지번주소'].str.contains('비전')]
pt_112 = pt_110[pt_110['소재지도로명주소'].str.contains('비전')]

pol = pd.read_csv('경기도지구대(파출소)현황.csv', encoding='cp949')
pol1 = pol.filter(['시군명','WGS84위도','WGS84경도'])
# data1 = data1[:100]
pol2 = pol1[pol1['시군명']=='평택시']
pol2['WGS84경도'] = pol2['WGS84경도'].astype(float)
pol2['WGS84위도'] = pol2['WGS84위도'].astype(float)
pol2['geometry'] = pol2.apply(lambda row : Point([row['WGS84경도'], row['WGS84위도']]), axis=1)
pol2 = gpd.GeoDataFrame(pol2, geometry='geometry')
pol2.crs = {'init':'epsg:4326'}
pol2 = pol2.to_crs({'init':'epsg:5179'})

pt_119 = pd.concat([pt_111,pt_112])
pt_119.sort_values('WGS84위도')
print(pt_119)

pt_119['WGS84경도'] = pt_119['WGS84경도'].astype(float)
pt_119['WGS84위도'] = pt_119['WGS84위도'].astype(float)
pt_119['geometry'] = pt_119.apply(lambda row : Point([row['WGS84경도'], row['WGS84위도']]), axis=1)
pt_119 = gpd.GeoDataFrame(pt_119, geometry='geometry')
pt_119.crs = {'init':'epsg:4326'}
pt_119 = pt_119.to_crs({'init':'epsg:5179'})

ax = set.plot(column="EMD_KOR_NM", figsize=(8,8))
pt_119.plot(ax=ax, marker='v', color='#ff6961', label='안심 비상벨')
pol2.plot(ax=ax, marker='s', color='darkblue', label='경찰서')

ax.set_title("평택시 비전동", fontsize=20)
ax.set_axis_off()
plt.legend()
plt.show()


# In[35]:


df = gpd.read_file('./TL_SCCO_EMD.shp',encoding='euc-kr')
set1 = df[df['EMD_KOR_NM'].str.contains('반구')]
pt_110 = pd.read_csv('./12_04_09_E_안전비상벨위치정보_울산.csv',encoding='cp949')
pt_119 = pt_110[pt_110['소재지지번주소'].str.contains('반구')]
pt_119.sort_values('WGS84위도')
print(pt_119)

pt_119['WGS84경도'] = pt_119['WGS84경도'].astype(float)
pt_119['WGS84위도'] = pt_119['WGS84위도'].astype(float)
pt_119['geometry'] = pt_119.apply(lambda row : Point([row['WGS84경도'], row['WGS84위도']]), axis=1)
pt_119 = gpd.GeoDataFrame(pt_119, geometry='geometry')
pt_119.crs = {'init':'epsg:4326'}
pt_119 = pt_119.to_crs({'init':'epsg:5179'})

ax = set1.plot(column="EMD_KOR_NM", figsize=(8,8), alpha=0.8)
pt_119.plot(ax=ax, marker='v', color='#ff6961', label='안심 비상벨')

ax.set_title("울산 중구", fontsize=20)
ax.set_axis_off()
plt.legend()
plt.show()


# ### 비전동 비상벨 안전 범위 및 가상 비상벨 설치

# In[36]:


import folium
from folium.plugins import MarkerCluster
import json
df3 = pd.read_csv('12_04_09_E_안전비상벨위치정보.csv',encoding='cp949')
geo_data = json.load(open('./LSMD_ADM_SECT_UMD_41.json', encoding='UTF8'))
map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)
pt_111 = df3[df3['소재지지번주소'].str.contains('비전')]
pt_112 = df3[df3['소재지도로명주소'].str.contains('비전')]
pt_119 = pd.concat([pt_111,pt_112])

for i in range(0,len(pt_119)):
    lat = pt_119.iloc[i, 7]
    lng = pt_119.iloc[i, 8]
    marker= folium.Circle([lat,lng], radius = 200, color='gray').add_to(map_osm)

folium.Choropleth(geo_data=geo_data, fill_opacity = 0).add_to(map_osm)

map_osm


# In[37]:


new_location = ['평택성결교회','비전고등학교','비전5로상가','비전4로상가',
                '소사벌호반베르디움','비전휴먼시아어린이집놀이터','소사벌뜨레휴이곡마을']

new_lat = [37.001415, 37.0001999, 37.000220, 37.002579,
           37.007243, 37.003447, 37.009513]

new_lng = [127.102487, 127.105388, 127.115388, 127.114833, 
           127.111029, 127.095785, 127.103557]

new_emergency_alram = pd.DataFrame({'위치':new_location,
                                '위도':new_lat,
                                '경도':new_lng})


new_emergency_alram


# In[38]:


import folium
from folium.plugins import MarkerCluster
import json
df3 = pd.read_csv('12_04_09_E_안전비상벨위치정보.csv',encoding='cp949')
geo_data = json.load(open('./LSMD_ADM_SECT_UMD_41.json', encoding='UTF8'))
map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)
pt_111 = df3[df3['소재지지번주소'].str.contains('비전')]
pt_112 = df3[df3['소재지도로명주소'].str.contains('비전')]
pt_119 = pd.concat([pt_111,pt_112])

for i in range(0,len(pt_119)):
    lat = pt_119.iloc[i, 7]
    lng = pt_119.iloc[i, 8]
    marker= folium.Circle([lat,lng], radius = 200, color='gray').add_to(map_osm)

# 가상 안심 비상벨 설치    
for i in range(0,len(new_emergency_alram.index)):
    name = new_emergency_alram.loc[i, '위치']
    lat = new_emergency_alram.loc[i, '위도']
    lng = new_emergency_alram.loc[i, '경도']
    marker= folium.Circle([lat,lng], popup=name, radius = 200, color='#ff6961').add_to(map_osm)    
    
folium.Choropleth(geo_data=geo_data, fill_opacity=0).add_to(map_osm)

map_osm


# ### 비전동 경찰서 안전 범위 및 가상 경찰초소 설치

# In[39]:


pol = pd.read_csv('경기도지구대(파출소)현황.csv', encoding='cp949')
pol1 = pol.filter(['시군명','WGS84위도','WGS84경도'])
# data1 = data1[:100]
pol2 = pol1[pol1['시군명']=='평택시']
pol2 = pol2.reset_index(drop=True)


# In[40]:


map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)
    
for i in range(0,len(pol2.index)):
    police_lat = pol2.loc[i, 'WGS84위도']
    police_lng = pol2.loc[i, 'WGS84경도']    
    marker= folium.Circle([police_lat,police_lng], radius = 600, color='gray').add_to(map_osm)
    
folium.Choropleth(geo_data=geo_data,fill_opacity=0).add_to(map_osm)
map_osm


# In[46]:


new_location = ['경찰초소1','경찰초소2','경찰초소3',
                '경찰초소4','경찰초소5']

new_lat = [37.002019, 37.007421, 36.999277,
           36.992891, 37.002034]

new_lng = [127.09207, 127.105195, 127.113599,
           127.113410, 127.100899]

new_pol = pd.DataFrame({'경찰초소 명':new_location,
                                '위도':new_lat,
                                '경도':new_lng})

new_pol


# In[42]:


map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)
    
for i in range(0,len(pol2.index)):
    police_lat = pol2.loc[i, 'WGS84위도']
    police_lng = pol2.loc[i, 'WGS84경도']    
    marker= folium.Circle([police_lat,police_lng], radius = 600, color='gray').add_to(map_osm)

# 가상 경찰초소 설치
for i in range(0,len(new_pol.index)):
    name = new_pol.loc[i, '위치']
    lat = new_pol.loc[i, '위도']
    lng = new_pol.loc[i, '경도']
    marker= folium.Circle([lat,lng], popup=name, radius = 600, color='#ff6961').add_to(map_osm)    
    
folium.Choropleth(geo_data=geo_data,fill_opacity=0).add_to(map_osm)
map_osm


# ### 실제 지도 및 지도에 표시

# In[43]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
df = gpd.read_file('./TL_SCCO_EMD.shp',encoding='euc-kr')
set = df[df['EMD_CD'].str.contains('41220118')]
pt_110 = pd.read_csv('./12_04_09_E_안전비상벨위치정보.csv',encoding='cp949')
pt_111 = pt_110[pt_110['소재지지번주소'].str.contains('비전')]
pt_112 = pt_110[pt_110['소재지도로명주소'].str.contains('비전')]

pol = pd.read_csv('경기도지구대(파출소)현황.csv', encoding='cp949')
pol1 = pol.filter(['시군명','WGS84위도','WGS84경도'])
# data1 = data1[:100]
pol2 = pol1[pol1['시군명']=='평택시']
pol2['WGS84경도'] = pol2['WGS84경도'].astype(float)
pol2['WGS84위도'] = pol2['WGS84위도'].astype(float)
pol2['geometry'] = pol2.apply(lambda row : Point([row['WGS84경도'], row['WGS84위도']]), axis=1)
pol2 = gpd.GeoDataFrame(pol2, geometry='geometry')
pol2.crs = {'init':'epsg:4326'}
pol2 = pol2.to_crs({'init':'epsg:5179'})

pt_119 = pd.concat([pt_111,pt_112])
pt_119.sort_values('WGS84위도')
print(pt_119)

pt_119['WGS84경도'] = pt_119['WGS84경도'].astype(float)
pt_119['WGS84위도'] = pt_119['WGS84위도'].astype(float)
pt_119['geometry'] = pt_119.apply(lambda row : Point([row['WGS84경도'], row['WGS84위도']]), axis=1)
pt_119 = gpd.GeoDataFrame(pt_119, geometry='geometry')
pt_119.crs = {'init':'epsg:4326'}
pt_119 = pt_119.to_crs({'init':'epsg:5179'})

new_emergency_alram['경도'] = new_emergency_alram['경도'].astype(float)
new_emergency_alram['위도'] = new_emergency_alram['위도'].astype(float)
new_emergency_alram['geometry'] = new_emergency_alram.apply(lambda row : Point([row['경도'], row['위도']]), axis=1)
new_emergency_alram = gpd.GeoDataFrame(new_emergency_alram, geometry='geometry')
new_emergency_alram.crs = {'init':'epsg:4326'}
new_emergency_alram = new_emergency_alram.to_crs({'init':'epsg:5179'})

new_pol['경도'] = new_pol['경도'].astype(float)
new_pol['위도'] = new_pol['위도'].astype(float)
new_pol['geometry'] = new_pol.apply(lambda row : Point([row['경도'], row['위도']]), axis=1)
new_pol= gpd.GeoDataFrame(new_pol, geometry='geometry')
new_pol.crs = {'init':'epsg:4326'}
new_pol = new_pol.to_crs({'init':'epsg:5179'})


ax = set.plot(column="EMD_KOR_NM", figsize=(8,8), alpha=0.8)
pt_119.plot(ax=ax, marker='v', color='#ff6961', label='기존 안심 비상벨')
new_emergency_alram.plot(ax=ax, marker='v', color='orange', label='신규 안심 비상벨')
pol2.plot(ax=ax, marker='s', color='darkblue', label='기존 경찰서')
new_pol.plot(ax=ax, marker='s', color='blue', label='신규 경찰초소')


ax.set_title("평택시 비전동", fontsize=20)
ax.set_axis_off()
plt.legend()
plt.show()


# In[44]:


pol = pd.read_csv('경기도지구대(파출소)현황.csv', encoding='cp949')
pol1 = pol.filter(['시군명','WGS84위도','WGS84경도'])
# data1 = data1[:100]
pol2 = pol1[pol1['시군명']=='평택시']
pol2 = pol2.reset_index(drop=True)


# In[45]:


import folium
from folium.plugins import MarkerCluster
import json
df3 = pd.read_csv('12_04_09_E_안전비상벨위치정보.csv',encoding='cp949')
geo_data = json.load(open('./LSMD_ADM_SECT_UMD_41.json', encoding='UTF8'))
map_osm = folium.Map(location=[36.993955883067, 127.10848409658], zoom_start=14.5)
pt_111 = df3[df3['소재지지번주소'].str.contains('비전')]
pt_112 = df3[df3['소재지도로명주소'].str.contains('비전')]
pt_119 = pd.concat([pt_111,pt_112])

for i in range(0,len(pt_119)):
    lat = pt_119.iloc[i, 7]
    lng = pt_119.iloc[i, 8]
    marker= folium.Marker([lat,lng],icon=folium.Icon(color='red')).add_to(map_osm)

# 가상 안심 비상벨 설치    
for i in range(0,len(new_emergency_alram.index)):
    name = new_emergency_alram.loc[i, '위치']
    lat = new_emergency_alram.loc[i, '위도']
    lng = new_emergency_alram.loc[i, '경도']
    marker= folium.Marker([lat,lng],icon=folium.Icon(color='orange')).add_to(map_osm) 
    
for i in range(0,len(pol2.index)):
    police_lat = pol2.loc[i, 'WGS84위도']
    police_lng = pol2.loc[i, 'WGS84경도']    
    folium.Marker([police_lat, police_lng], 
                  icon=folium.Icon(color='darkblue')).add_to(map_osm)
    
# 가상 경찰초소 설치
for i in range(0,len(new_pol.index)):
    name = new_pol.loc[i, '위치']
    lat = new_pol.loc[i, '위도']
    lng = new_pol.loc[i, '경도']
    folium.Marker([lat,lng], icon=folium.Icon(color='blue')).add_to(map_osm)

folium.Choropleth(geo_data=geo_data, fill_opacity=0).add_to(map_osm)

map_osm

