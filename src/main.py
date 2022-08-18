#!/usr/bin/env python
# coding: utf-8

# # CMPT459 Data Mining
# ## Course Project Milestone 1
# ### Group: metaverse -- Ze Ming Gong, Zhuo Ning, Yunlong Li

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #visualisation
from geopy.geocoders import Nominatim
# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
import datetime
from datetime import date
from scipy import stats

import geoplot as gplt
import geopandas as gpd
import geoplot.crs as gcrs
import imageio
import pathlib
import mapclassify as mc
import contextily as cx


# ## 1.1 Cleaning messy outcome labels

# In[2]:


cases_train = pd.read_csv('../data/cases_2021_train.csv')
cases_test = pd.read_csv('../data/cases_2021_test.csv')
location = pd.read_csv('../data/location_2021.csv')


# In[3]:


cases_train.groupby('outcome').size()


# In[4]:


cases_train.loc[cases_train.outcome == 'Discharged', 'outcome_group'] = "hospitalized"
cases_train.loc[cases_train.outcome == 'Discharged from hospital', 'outcome_group'] = "hospitalized"
cases_train.loc[cases_train.outcome == 'Hospitalized', 'outcome_group'] = "hospitalized"
cases_train.loc[cases_train.outcome == 'critical condition', 'outcome_group'] = "hospitalized"
cases_train.loc[cases_train.outcome == 'discharge', 'outcome_group'] = "hospitalized"
cases_train.loc[cases_train.outcome == 'discharged', 'outcome_group'] = "hospitalized"

cases_train.loc[cases_train.outcome == 'Alive', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'Receiving Treatment', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'Stable', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'Under treatment', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'recovering at home 03.03.2020', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'released from quarantine', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'stable', 'outcome_group'] = "nonhospitalized"
cases_train.loc[cases_train.outcome == 'stable condition', 'outcome_group'] = "nonhospitalized"

cases_train.loc[cases_train.outcome == 'Dead', 'outcome_group'] = "deceased"
cases_train.loc[cases_train.outcome == 'Death', 'outcome_group'] = "deceased"
cases_train.loc[cases_train.outcome == 'Deceased', 'outcome_group'] = "deceased"
cases_train.loc[cases_train.outcome == 'Died', 'outcome_group'] = "deceased"
cases_train.loc[cases_train.outcome == 'death', 'outcome_group'] = "deceased"
cases_train.loc[cases_train.outcome == 'died', 'outcome_group'] = "deceased"

cases_train.loc[cases_train.outcome == 'Recovered', 'outcome_group'] = "recovered"
cases_train.loc[cases_train.outcome == 'recovered', 'outcome_group'] = "recovered"

cases_train = cases_train.drop(columns=['outcome'])


# In[5]:


cases_train.groupby('outcome_group').size()


# ## 1.2 Outcome labels
# For the `cases_2021_train.csv` file, the type of data mining task is data cleaning, since the `outcome_group` labels are obtained by `.groupby()` method. The primary type is multi-class classification.
# 
# For the `cases_2021_test.csv` file, the type of data mining task is multi-class classification.

# ## 1.3 Exploratory Data Analysis (EDA)

# In[6]:


cases_train


# In[7]:


cases_train.isnull().sum(axis = 0)


# In[8]:


cases_test


# In[9]:


cases_test.isnull().sum(axis = 0)


# In[10]:


location


# In[11]:


location.isnull().sum(axis = 0)


# To draw historgram of age value, we decide to perform data cleaning of age in this section.

# In[12]:


cases_train = cases_train[cases_train['age'].notna()]
cases_test = cases_test[cases_test['age'].notna()]


# In[13]:


# Function that handle age column.
def handleAge(age):
    if ' months' in age:
        return round(float(age.replace(' months', ''))/12)
    positionOfTo = age.find('-')
    if positionOfTo >= 1:
        age1 = float(age[0:positionOfTo])
        age2 = age[positionOfTo+1:]
        if len(age2) <= 0:
            age2 = 0
        else:
            age2 = float(age2)
        age = (age1 + age2) / 2
    return(round(float(age)))


# In[14]:


cases_train['age'] = cases_train['age'].apply(handleAge)
cases_test['age'] = cases_test['age'].apply(handleAge)


# In[15]:


# historgram of training data's age
ax = cases_train['age'].hist()
ax.set_title('Convid Cases Age Histrogram - Train Set')
ax.set_xlabel('Age')
ax.set_ylabel('Count')


# In[16]:


# historgram of testing data's age
ax = cases_test['age'].hist()
ax.set_title('Convid Cases Age Histrogram - Test Set')
ax.set_xlabel('Age')
ax.set_ylabel('Count')


# In[17]:


ax = cases_train.groupby('sex',dropna=False).size().plot.pie(y='sex', figsize=(5, 5))
ax.set_title('Covid Cases in Sex Group - Train Set')


# In[18]:


ax = cases_test.groupby('sex',dropna=False).size().plot.pie(y='sex', figsize=(5, 5))
ax.set_title('Covid Cases in Sex Group - Test Set')


# In[19]:


# cases_train.groupby('country',dropna=False).size().sort_values().plot.pie(y='country')
cases_train_groupby_country = cases_train.groupby('country',dropna=False).size().sort_values(ascending=False)
cases_train_groupby_country_valueArray = cases_train_groupby_country.values
cases_train_groupby_country_countryArray = cases_train_groupby_country.index.tolist()

first2Value = cases_train_groupby_country_valueArray[0:2]
restValue = np.sum(cases_train_groupby_country_valueArray[2:])

cases_train_groupby_country_valueArray = np.append(first2Value, restValue)
cases_train_groupby_country_countryArray = np.append(cases_train_groupby_country_countryArray[0:2], 'other')

plt.pie(cases_train_groupby_country_valueArray)
plt.legend(loc=3, labels=cases_train_groupby_country_countryArray)
plt.title('Convid Cases in Country - Train')
plt.show() 


# In[20]:


# cases_train.groupby('country',dropna=False).size().sort_values().plot.pie(y='country')
cases_test_groupby_country = cases_test.groupby('country',dropna=False).size().sort_values(ascending=False)
cases_test_groupby_country_valueArray = cases_test_groupby_country.values
cases_test_groupby_country_countryArray = cases_test_groupby_country.index.tolist()

first2Value = cases_test_groupby_country_valueArray[0:2]
restValue = np.sum(cases_test_groupby_country_valueArray[2:])

cases_test_groupby_country_valueArray = np.append(first2Value, restValue)
cases_test_groupby_country_countryArray = np.append(cases_test_groupby_country_countryArray[0:2], 'other')

plt.pie(cases_test_groupby_country_valueArray)
plt.legend(loc=3, labels=cases_test_groupby_country_countryArray)
plt.title('Convid Cases in Country - Test')
plt.show() 


# In[21]:


ax = cases_train['outcome_group'].hist()
ax.set_title('Convid Cases Outcome Groups Histrogram')
ax.set_xlabel('Outcome')
ax.set_ylabel('Count')


# In[22]:


location['Confirmed_log'] = np.log(location['Confirmed'] + 1)
location['Deaths_log'] = np.log(location['Deaths'] + 1)
location['Active_log'] = np.log(location['Active'] + 1)
location['Recovered_log'] = np.log(location['Recovered'] + 1)


# In[23]:


ax = location['Confirmed_log'].hist()
ax.set_title('Confirmed Cases in Log Scale')
ax.set_xlabel('log(Confirmed Cases)')
ax.set_ylabel('Count')


# In[24]:


ax = location['Deaths_log'].hist()
ax.set_title('Death Cases in Log Scale')
ax.set_xlabel('log(Death Cases)')
ax.set_ylabel('Count')


# In[25]:


ax = location['Active_log'].hist()
ax.set_title('Active Cases in Log Scale')
ax.set_xlabel('log(Active Cases)')
ax.set_ylabel('Count')


# In[26]:


ax = location['Recovered_log'].hist()
ax.set_title('Recovered Cases in Log Scale')
ax.set_xlabel('log(Recovered Cases)')
ax.set_ylabel('Count')


# In[27]:


# Confirmed	Deaths	Recovered	Active	Incident_Rate	Case_Fatality_Ratio
loc_contry_grp = location.groupby('Country_Region')
country_cf = loc_contry_grp.sum('Confirmed').sort_values(by='Confirmed', ascending=False).head(10)
country_cf['Confirmed'].sort_values(ascending=True).plot.barh(title='Top 10 Countries with the Most Confirmed Cases', xlabel='Country')


# In[28]:


country_d = loc_contry_grp.sum('Deaths').sort_values(by='Deaths', ascending=False).head(10)
country_d['Deaths'].sort_values(ascending=True).plot.barh(title='Top 10 Countries with the Most Death Cases', xlabel='Country')


# In[29]:


country_a = loc_contry_grp.sum('Active').sort_values(by='Active', ascending=False).head(10)
country_a['Active'].sort_values(ascending=True).plot.barh(title='Top 10 Countries with the Most Active Cases', xlabel='Country')


# In[30]:


country_r = loc_contry_grp.sum('Recovered').sort_values(by='Recovered', ascending=False).head(10)
country_r['Recovered'].sort_values(ascending=True).plot.barh(title='Top 10 Countries with the Most Recovered Cases', xlabel='Country')


# In[31]:


country_i = loc_contry_grp.sum('Incident_Rate').sort_values(by='Incident_Rate', ascending=False).head(10)
country_i['Incident_Rate'].sort_values(ascending=True).plot.barh(title='Top 10 Countries with the Most Incident Rate', xlabel='Country')


# In[32]:


country_f = loc_contry_grp.sum('Case_Fatality_Ratio').sort_values(by='Case_Fatality_Ratio', ascending=False).head(10)
country_f['Case_Fatality_Ratio'].sort_values(ascending=True).plot.barh(title='Top 10 Countries with the Most Case Fatality Ratio', xlabel='Country')


# In[33]:


location_na = location.dropna(subset=['Lat', 'Long_'])
location_g = gpd.GeoDataFrame(location_na, geometry=gpd.points_from_xy(x=location_na.Long_, y=location_na.Lat))
location_g


# In[34]:


ax = location_g.plot(figsize=(16, 9),
    alpha=0.5,
    edgecolor='k',
    column='Confirmed_log',
    legend=True)

ax.set_title('Confirmed Cases World Map in Log Scale')


# In[35]:


plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 200

ax = gplt.pointplot(
    location_g, projection=gcrs.WebMercator(),
    hue='Confirmed_log', scale='Confirmed_log', limits=(1, 10),
    legend=True, legend_var='scale'
)
ax.set_title('Active Cases World Map in Log Scale')


# ## 1.4 Data cleaning and imputing missing values

# In[36]:


# Codes that used to check which columns contain NaN
nan_values = cases_train.isna()
nan_columns = nan_values.any()
columns_with_nan = cases_train.columns[nan_columns].tolist()
print(columns_with_nan)


# In[37]:


# Codes that used to check which columns contain NaN
nan_values = cases_test.isna()
nan_columns = nan_values.any()
columns_with_nan = cases_test.columns[nan_columns].tolist()
print(columns_with_nan)


# In[38]:


# Replace NaN in sex column by 'unknown'
cases_train['sex'] = cases_train['sex'].replace(np.nan, 'unknown')
cases_test['sex'] = cases_test['sex'].replace(np.nan, 'unknown')


# In[39]:


# With lait land long, we can use geopy to find the country and provience name.
def handleCountry(lait,long, country):
    if pd.notna(country):
        return country
    else:
        if lait > 90 or lait < -90 or long > 180 or long < -180:
            return country
        geolocator = Nominatim(user_agent="geoapiExercises")
        loc = geolocator.reverse(str(lait)+','+str(long), language='en', timeout=None)
        if loc is None:
            return country
        address = loc.raw['address']
        Country = address.get('country', '')
        return Country
def handleProvince(long, lait, province):
    if pd.notna(province):
        return province
    else:
        if lait > 90 or lait < -90 or long > 180 or long < -180:
            return province
        geolocator = Nominatim(user_agent="geoapiExercises")
        loc = geolocator.reverse(str(lait)+","+str(long), language='en', timeout=None)
        if loc is None:
            return province
        address = loc.raw['address']
        Province = address.get('state', '')
        return Province


# To handle GeocoderServiceError:
# 
# Download https://letsencrypt.org/certs/lets-encrypt-r3.pem
# 
# rename file .pem to .cer
# 
# install

# In[40]:


try:
    cases_train['country'] = cases_train.apply(lambda row : handleCountry(row['latitude'], row['longitude'], row['country']), axis = 1)
    cases_train['province'] = cases_train.apply(lambda row: handleProvince(row['latitude'], row['longitude'], row['province']), axis = 1)
    cases_test['country'] = cases_test.apply(lambda row : handleCountry(row['latitude'], row['longitude'], row['country']), axis = 1)
    cases_test['province'] = cases_test.apply(lambda row: handleProvince(row['latitude'], row['longitude'], row['province']), axis = 1)
except:
    print("Unable to reach geopy server, just remove rows with NaN in country and provience instead.")


# For rows still have country or province as NaN, replace NaN with 'unknown'
cases_train[['country', 'province']] = cases_train[['country','province']].fillna(value='unknown')
cases_test[['country', 'province']] = cases_test[['country','province']].fillna(value='unknown')


# In[41]:


allDate = cases_train['date_confirmation'].dropna().tolist() + cases_test['date_confirmation'].dropna().tolist()


# In[42]:


# Use the average date to replace NaN
# May cause dataset skewed
totalDays = 0
initialDay = date(1, 1, 1)
for i in allDate:
    day = int(i[0:2])
    month = int(i[3:5])
    year = int(i[6:10])
    currDate = date(year, month, day)
    totalDays = totalDays + (currDate - initialDay).days
avgDays = round(totalDays/len(allDate))
avgDate = initialDay + datetime.timedelta(days = avgDays)
avgDateInStr = str(avgDate.day) + '.' + str(avgDate.month) + '.' + str(avgDate.year)
cases_train['date_confirmation'] = cases_train['date_confirmation'].fillna(avgDateInStr)
cases_test['date_confirmation'] = cases_test['date_confirmation'].fillna(avgDateInStr)


# In[43]:


# Replace NaN in additional_information and source with empty string
cases_train['additional_information'] = cases_train['additional_information'].fillna('')
cases_test['additional_information'] = cases_test['additional_information'].fillna('')
cases_train['source'] = cases_train['source'].fillna('')
cases_test['source'] = cases_test['source'].fillna('')


# In[44]:


# Remove rows with NaN in Lat and Long_
location = location[location['Lat'].notna()]
location = location[location['Long_'].notna()]


# In[45]:


# Geopy can not really help find the province state with lait and long
# Replace NaN in Province_State with unknown
# location['Province_State'] = location.apply(lambda row: handleProvince(row['Lat'], row['Long_'], row['Province_State']), axis = 1)
location['Province_State'] = location['Province_State'].fillna('Unknown')


# In[46]:


# Use average recover ratio to calculate number of recovered and active
# MAY CAUSE DATASET SKEWED
def handleRecovered(Confirmed, Deaths, Recovered, Active, MeanRecoverRatio):
    if pd.notna(Recovered):
        return Recovered
    else:
        if pd.notna(Active):
            return (Confirmed-Deaths)-Active
        else:
            return (Confirmed-Deaths)*MeanRecoverRatio
        return Country
def handleActive(Confirmed, Deaths, Recovered, Active, MeanRecoverRatio):
    if pd.notna(Active):
        return Active
    else:
        if pd.notna(Recovered):
            return (Confirmed-Deaths)-Recovered
        else:
            return (Confirmed-Deaths)*(1-MeanRecoverRatio)
        return Country


# In[47]:


recoveredAndActive = location.get(['Recovered','Active'])
recoveredAndActive = recoveredAndActive[recoveredAndActive['Recovered'].notna()]
recoveredAndActive = recoveredAndActive[recoveredAndActive['Active'].notna()]
recoveredAndActive['RecoverRatio'] = recoveredAndActive['Recovered']/(recoveredAndActive['Recovered']+recoveredAndActive['Active'])
MeanRecoverRatio = recoveredAndActive['RecoverRatio'].mean()


location['Recovered'] = location.apply(lambda row: handleRecovered(row['Confirmed'], row['Deaths'], row['Recovered'], row['Active'],MeanRecoverRatio), axis = 1)
location['Active'] = location.apply(lambda row: handleActive(row['Confirmed'], row['Deaths'], row['Recovered'], row['Active'],MeanRecoverRatio), axis = 1)

location['Case_Fatality_Ratio'] = location['Case_Fatality_Ratio'].fillna(0)
location['Incident_Rate'] = location['Incident_Rate'].fillna(0)


# In[48]:


nan_values = location.isna()
nan_columns = nan_values.any()
columns_with_nan = location.columns[nan_columns].tolist()
print(columns_with_nan)


# ## 1.5 Dealing with outliers
# 
# In previous steps, we have dropped rows in cases_train, cases_test and location with wrong latitude and longitude. Thus, we do not have to handle outlaiers in latitude and longitude. 
# Moreover, by checking the box plot of age, we can say that the age values are reasonable. Age in train and test sets does not include particularly unreasonable values.
# 
# However, by checking the box plots, we can see there are significent outliers in 'Confirmed', 'Recovered', 'Active'

# In[49]:


cases_train.select_dtypes(include=np.number)


# In[50]:


cases_test.select_dtypes(include=np.number)


# In[51]:


location.select_dtypes(include=np.number)


# In[52]:


boxplot1 = cases_train.boxplot(column=['age'])
boxplot2 = cases_test.boxplot(column=['age'])
boxplot3 = location.boxplot(column=['Lat', 'Long_'])
boxplot4 = location.boxplot(column=['Confirmed'])
boxplot5 = location.boxplot(column=['Deaths'])
boxplot6 = location.boxplot(column=['Recovered'])
boxplot7 = location.boxplot(column=['Active'])
boxplot8 = location.boxplot(column=['Incident_Rate'])


# In[53]:


boxplot9 = location.boxplot(column=['Case_Fatality_Ratio'])


# In[54]:


outlierColumns = location[['Case_Fatality_Ratio']]


# In[55]:


filteredLocation = outlierColumns[(np.abs(stats.zscore(outlierColumns)) < 2).all(axis=1)]
# Reference: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe


# In[56]:


location = pd.concat([location, filteredLocation], axis=1, join="inner")
location = location.loc[:,~location.columns.duplicated()]


# ## 1.6 Joining the cases and location dataset

# In[57]:


location['Country_Region'] = location['Country_Region'].replace(['US'],'United States')
location['Country_Region'] = location['Country_Region'].replace(['Korea, South'],'South Korea')
location['Country_Region'] = location['Country_Region'].replace(['Taiwan*'],'Taiwan')


# In[58]:


aggColumnsInLocation = location[['Province_State', 'Country_Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']]
meanColumnsInLocation = location[['Province_State', 'Country_Region', 'Incident_Rate', 'Case_Fatality_Ratio']]


# In[59]:


aggColumnsInLocation = aggColumnsInLocation.groupby(['Province_State', 'Country_Region']).sum()


# In[60]:


meanColumnsInLocation = meanColumnsInLocation.groupby(['Province_State', 'Country_Region']).mean()


# In[61]:


groupedLocation = pd.merge(aggColumnsInLocation, meanColumnsInLocation, on=['Province_State', 'Country_Region'], how='inner')
groupedLocation
groupedLocation.to_csv('../results/location_2021_processed.csv.csv')


# In[62]:


joined_case_train = pd.merge(cases_train, groupedLocation,  how='left', left_on=['country','province'], right_on = ['Country_Region','Province_State'])
joined_case_test = pd.merge(cases_test, groupedLocation,  how='left', left_on=['country','province'], right_on = ['Country_Region','Province_State'])


# In[63]:


joined_case_train


# In[64]:


joined_case_test
joined_case_test.to_csv('../results/cases_2021_test_processed.csv')


# In[65]:


joined_case_train = joined_case_train.dropna()
joined_case_test = joined_case_test.dropna()


# In[66]:


joined_case_train
joined_case_train.to_csv('../results/cases_2021_train_processed.csv')


# In[67]:


joined_case_test


# In[ ]:




