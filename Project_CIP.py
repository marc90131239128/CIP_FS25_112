# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:29:32 2025

@author: bur
"""
import numpy as np
import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import requests
import pandas as pd



csv_files = glob.glob(f"immoscout*.csv")

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    
# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)
#

# Convert string representation of dictionary to actual dictionary and create new columns
def extract_hauptangaben(x):
    if pd.isna(x):
        return {}
    try:
        return eval(x)
    except:
        return {}
#
# Extract all unique keys from Hauptangaben dictionaries
all_keys = set()
for d in merged_df['Hauptangaben'].apply(extract_hauptangaben):
    all_keys.update(d.keys())

# Create new columns for each key in Hauptangaben
for key in all_keys:
    merged_df[key] = merged_df['Hauptangaben'].apply(
        lambda x: extract_hauptangaben(x).get(key, None))
#
apartment_types = [
    'Wohnung', 
    'Maisonette / Duplex',
    'Dachwohnung',
    'Attikawohnung', 
    'Studio',
    'Loft',
    'Terrassenwohnung',
    'Mansarde'
]

# Filter dataframe to keep only specified apartment types
merged_df = merged_df[merged_df['Objekttyp:'].isin(apartment_types)]
#doppelte Einträge löschen (gibt es zwar nicht wahrscheinlich)
merged_df = merged_df.drop_duplicates()
#
#
#drop rows with NaN entries in specific rows
merged_df=merged_df.dropna(axis=0, subset=['City', 'Price (CHF/Month)','Rooms', 'Street', 'PLZ/City'])
merged_df.columns = merged_df.columns.str.replace(':', '', regex=True)
#clean datas with price
merged_df['Net Price']=np.where(pd.isna(merged_df['Net Price']),merged_df['Price (CHF/Month)'],merged_df['Net Price'])
merged_df['Additional Costs']=np.where(pd.isna(merged_df['Additional Costs']),'CHF 0',merged_df['Additional Costs'])

for column in ['Price (CHF/Month)', 'Net Price', 'Additional Costs']:
    merged_df[column]=merged_df[column].apply(lambda x: re.sub(r'\D', '', x))
    merged_df[column]=merged_df[column].astype(int)
    
#clean living space
merged_df['Living Space']=merged_df['Living Space'].apply(lambda x: x.replace("m2",'').strip() if not pd.isna(x) else x)
#preis pro m2 und leere einträge füllen
price_m2 = merged_df.dropna(subset=['Price (CHF/Month)', 'Living Space']).copy()
price_m2['Price per m2'] = price_m2['Price (CHF/Month)'] / price_m2['Living Space'].astype(int)
mean_price_m2 = price_m2.groupby('City')['Price per m2'].mean().reset_index()

def fill_missing_living_space(row):
    if pd.isna(row['Living Space']):
    # Finde den entsprechenden Durchschnittspreis pro m2 für die Stadt
        mean_price = mean_price_m2.loc[mean_price_m2['City'] == row['City'], 'Price per m2'].iat[0]
        return int(row['Price (CHF/Month)']/mean_price) # Hier price_m2 sollte der Wert für die Berechnung sein
    else:
        return row['Living Space']

merged_df['Living Space'] = merged_df.apply(fill_missing_living_space, axis=1)
merged_df['Living Space']=merged_df['Living Space'].astype(int)
#Etage: EG durch 0 und 100 durch 1
merged_df['Etage']=merged_df['Etage'].replace('EG',0).replace(100,1)
# search in text for the 
def search_etage(row):
    if pd.isnull(row['Etage']):
        text=row['Description']
        try:
            output=re.search('. Stock',text).start()
            try:
                etage=int(text[output-2:output]) #if Double-digit
            except:
                etage=int(text[output-1])
            return str(etage)
        except:
            try:
                output=re.search('. OG',text).start()
                try:
                    etage=int(text[output-2:output]) #if Double-digit
                except:
                    etage=int(text[output-1])
                return str(etage)
            except: pass
    else:
        return row['Etage']

merged_df['Etage']=merged_df.apply(search_etage,axis=1)


#Seperate PLZ and City
merged_df[['PLZ', 'City']] = merged_df['PLZ/City'].str.split(' ', expand=True,n=1)

#nur relevante Spalten verwenden
merged_df = merged_df[['City','PLZ' ,'Price (CHF/Month)', 'Net Price', 'Additional Costs', 'Rooms',
       'Living Space', 'Street', 'Description', 'Eigenschaften',
        'Listing URL', 'Etage','Objekttyp']]

#
def extract_eigenschaften(x):
    if isinstance(x, str):
        elements=x.split(",")
    else: elements=[]
    return elements

properties=set()
for d in merged_df['Eigenschaften'].apply(extract_eigenschaften):
    properties.update(d)
properties=set(el.strip() for el in properties)

merged_df['Eigenschaften']=merged_df['Eigenschaften'].replace(np.nan,'')

# check which properties is how often represent
df_properties=merged_df.copy()
for properti in properties:
    df_properties[properti]=df_properties['Eigenschaften'].apply(
        lambda x: properti in x)
    
#count all columns with true values for deciding which further predictor is relevant
true_count = df_properties[properties].sum()
true_percentages = (df_properties[properties].sum() / len(df_properties)) * 100

# Create a summary DataFrame
true_percentages_summary = pd.DataFrame({
    'True Count': true_count,
    'True Percentage': true_percentages
}) 
true_percentages_summary = true_percentages_summary.sort_values('True Percentage', ascending=False) 
#  
test=merged_df[pd.isna(merged_df['Net Price'])]

#versuche anhand der Spalte Eigenschaften das Propertie zu finden
properties_predictors=['Balkon / Terrasse','Lift','Minergie','Aussicht','Garage','Parkplatz','Kinderfreundlich','Swimmingpool','Haustiere erlaubt']
for properti in properties_predictors:
    merged_df[properti]=merged_df['Eigenschaften'].apply(
        lambda x: properti in x)
    
#versuche anhand der Spalte Description das Propertie zu finden, falls dies nicht in der spalte Description vorhanden ist
def aktualisiere_properti(merged_df, properti):
    def pruefe_properti(beschreibung, properti):
        if pd.isna(beschreibung):
            return False
        # Überprüft, ob "kein" oder "ohne" vor dem Eigenschaftsnamen stehen
        ausschlusswoerter = ["kein", "ohne","keinen"]
        for wort in ausschlusswoerter:
            if f"{wort} {properti.lower()}" in beschreibung.lower():
                return False

        # Überprüft, ob der Eigenschaftsname in der Beschreibung vorkommt
        return properti.lower() in beschreibung.lower()

    # Erstellt eine boolesche Maske, die angibt, wo die Eigenschaft aktualisiert werden soll
    maske = merged_df[properti] == False
    merged_df.loc[maske, properti] = merged_df.loc[maske, "Description"].apply(
        lambda x: pruefe_properti(x, properti))
    return merged_df

for properti in properties_predictors:
    merged_df = aktualisiere_properti(merged_df, properti)
#

# Wordcloud "Description" Might be useful to get a certain sentiment of different price classes luxury vs modest prices
if False:
    import re
    from wordcloud import STOPWORDS
    
    text = ' '.join(df['Description'].astype(str).tolist())
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    
    stopwords = set(STOPWORDS)
    text = ' '.join(word for word in text.split() if word not in stopwords)
    
    
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title("Apartment Descriptions Word Cloud")
    plt.savefig("word_cloud_description.png", dpi=300, bbox_inches="tight")
    plt.show()


#
#delete columns
df = merged_df.drop(columns=['Description', 'Eigenschaften'])




# # first graphs to detect outliers
outliers=[]
#Price and Rooms
plt.figure(figsize=(10, 6))
plt.scatter(df['Price (CHF/Month)'],df['Rooms'], color='skyblue',marker='o')
plt.title('Price to rooms')
plt.xlabel('Price (CHF/Month)')
plt.ylabel('Rooms')
plt.xticks(rotation=0)
plt.grid(axis='y')

outliers.extend(df[df['Rooms']>10].index.to_list()) #one advertisement has 27 Rooms, thats wrong

#Price and living space
plt.figure(figsize=(10, 6))
plt.scatter(df['Price (CHF/Month)'],df['Living Space'], color='skyblue',marker='o')
plt.title('Price to living space')
plt.xlabel('Price (CHF/Month)')
plt.ylabel('Living Space')
plt.xticks(rotation=0)
plt.grid(axis='y')

outliers.extend(df[df['Price (CHF/Month)']<500].index.to_list()) #only cellar and storeroom

#Net Price and additional Costs
plt.figure(figsize=(10, 6))
plt.scatter(df['Net Price'],df['Additional Costs'], color='skyblue',marker='o')
plt.title('Net Price to additional costs')
plt.xlabel('Net Price')
plt.ylabel('Additional Costs')
plt.xticks(rotation=0)
plt.grid(axis='y')

outliers.extend(df[df['Additional Costs']>df['Net Price']].index.to_list()) #wrong price indication -> may switch price?

#Rooms and Living Space
plt.figure(figsize=(10, 6))
plt.scatter(df['Rooms'],df['Living Space'], color='skyblue',marker='o')
plt.title('Rooms to living space')
plt.xlabel('Rooms')
plt.ylabel('Living Space')
plt.xticks(rotation=0)
plt.grid(axis='y')
# -> no outliers

#delete outliers
df=df[~df.index.isin(outliers)]

# add longitude und latitude to the dataframe
df['complete_address'] = df['Street'] + df['PLZ'] + ' ' + df['City'] 
def getgeocoordinates_publictransport(df):
    # Add the longitude and latitude to the dataframe
    #########Function to get coordinates from geo.admin.ch##########################
    def get_coordinates_geo_admin(address):
        base_url = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
        params = {
            "searchText": address,
            "type": "locations",
            "limit": 1  # Get only the best match
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                attrs = results[0].get("attrs", {})
                lat = attrs.get("lat")
                lon = attrs.get("lon")
                return lat, lon
        
        return None, None  # Return None if no match is found 
    
    
    
    # Apply function to the 'complete_address' column
    df["latitude"], df["longitude"] = zip(*df["complete_address"].apply(get_coordinates_geo_admin))
    # Are there any missing values in the latitude and longitude columns?
    missing=df[df['latitude'].isna() | df['longitude'].isna()].reset_index(drop=True)
    print(f"Werte mit Nan{len(missing)}") # 0 missing values
    
    ################# Get the Accessibility to public transport points#####################
    
    def get_public_transport_accessibility_wgs84(latitude, longitude):
        """
        Retrieves public transport accessibility information for a given location using the coordinates.
        """
        url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "layers": "all:ch.are.erreichbarkeit-oev",
            "tolerance": 0,
            "mapExtent": f"{longitude-0.01},{latitude-0.01},{longitude+0.01},{latitude+0.01}",
            "imageDisplay": "100,100,96",
            "returnGeometry": "false",
            "sr": "4326"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                # Extract relevant attributes from the first result
                attributes = data['results'][0]['attributes']
                return attributes
            else:
                return None
        else:
            return None
    
    # Load your DataFrame (assuming df already contains latitude and longitude)
    
    
    # Apply the function to the DataFrame
    df["public_transport_accessibility"] = df.apply(
        lambda row: get_public_transport_accessibility_wgs84(row["latitude"], row["longitude"]), axis=1
    )
    
    # Expand the dictionary column into separate columns
    df_accessibility = df["public_transport_accessibility"].apply(pd.Series)
    df = pd.concat([df, df_accessibility], axis=1).drop(columns=["public_transport_accessibility"])
    
    # How many missing values in oev_erreichb_ewap?
    print(df['oev_erreichb_ewap'].isna().sum())
    
    # Drop the column label as we don't need it
    df = df.drop(columns=['label'])
    df[['complete_address','latitude','longitude','oev_erreichb_ewap']].to_csv(f"coordinates_adresses.csv",index=False)

    return df
if False: getgeocoordinates_publictransport(df) #calculate only with new dataset because of time


df_geocoordinates=pd.read_csv(f"coordinates_adresses.csv")[['complete_address', 'latitude', 'longitude','oev_erreichb_ewap']]
df_geocoordinates=df_geocoordinates.drop_duplicates()
df=df.merge(df_geocoordinates)

# Delete the row where complete_address == "Bahnhofstrasse 2, Raron,3018 Bern" as we cannot identify where the right address is as it is already deletd on Immoscout

df = df[df["complete_address"] != "Bahnhofstrasse 2, Raron,3018 Bern"]  



# Steuerdaten hinzufügen
df_steuern_personen=pd.read_excel(f"Steuerdaten_2021.xlsx",sheet_name="121",skiprows=2)
df_steuern=pd.read_excel(f"Steuerdaten_2021.xlsx",sheet_name="122",skiprows=2)

df_median_einkommen=df_steuern.iloc[:,:4]
df_steuern_personen=df_steuern_personen.iloc[:, 4:13].replace('- ',np.nan)
df_steuern=df_steuern.iloc[:, 4:13].replace('- ',np.nan)

#spezfisches Einkommen pro Person
steuern_pro_p=df_steuern.div(df_steuern_personen)
weighted_median=[] #median berechnen
for i in range(len(df_steuern)):
    weighted_median.append(round(np.median(np.repeat(steuern_pro_p.loc[i].dropna(), df_steuern_personen.loc[i].dropna())),1))
df_median_einkommen['median_einkommen']=pd.Series(weighted_median)    

df_gemeindeliste=pd.read_csv(f"Gemeindeliste.csv",sep=";")
steuerdaten=pd.merge(df_median_einkommen[['Gemeinde ID','median_einkommen']],df_gemeindeliste[['BFS-Nr','PLZ']],left_on=['Gemeinde ID'],right_on=['BFS-Nr'])
df['PLZ']=df['PLZ'].astype(int)
steuerdaten=steuerdaten[['median_einkommen','PLZ']].drop_duplicates(subset=['PLZ'])

df=pd.merge(df,steuerdaten,left_on=['PLZ'],right_on=['PLZ'],how='left')

# remove rows with missing values in median_einkommen
df=df[df['median_einkommen'].notna()]

#replace wrong city names
df['City'] = df['City'].replace('Bern-Bümpliz', 'Bern')
df['City'] = df['City'].replace('Hinterkappelen', 'Bern')
df['City'] = df['City'].replace('Littau', 'Luzern')
df['City'] = df['City'].replace('Kriens', 'Luzern')
df['City'] = df['City'].replace('Winkeln', 'St.Gallen')
df['City'] = df['City'].replace('St. Gallen', 'St.Gallen')
df['City'] = df['City'].replace('Zurich', 'Zürich')
df['City'] = df['City'].replace('Schwamendingen', 'Zürich')
df['City'] = df['City'].replace('Winterthur Seen', 'Winterthur')
df['City'] = df['City'].replace('Dübendorf', 'Zürich')


#
############ Exploratory Data Analysis ######################



from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for all visualizations
plt.style.use('seaborn')
sns.set_palette("Set2")

# Create figure for price distribution
plt.figure(figsize=(15, 6))

# Price distribution by city
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='City', y='Price (CHF/Month)', color=sns.color_palette("Set2")[0])
plt.xticks(rotation=45)
plt.title('Rental Price Distribution by City')
plt.ylabel('Monthly Rent (CHF)')

# Overall price distribution
plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Price (CHF/Month)', bins=50, color=sns.color_palette("Set2")[0])
plt.title('Overall Rental Price Distribution')
plt.xlabel('Monthly Rent (CHF)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig("price_distribution.png", dpi=300, bbox_inches="tight")
plt.show()



###########################################################
# Create figure for living space analysis
plt.figure(figsize=(15, 6))

# Living space distribution by city
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='City', y='Living Space', color=sns.color_palette("Set2")[0])
plt.xticks(rotation=45)
plt.title('Living Space Distribution by City')
plt.ylabel('Living Space (m²)')

# Room count distribution
plt.subplot(1, 2, 2)
room_counts = df['Rooms'].value_counts().sort_index()
sns.barplot(x=room_counts.index, y=room_counts.values, color=sns.color_palette("Set2")[0])
plt.title('Distribution of Room Numbers')
plt.xlabel('Number of Rooms')
plt.ylabel('Count')
plt.subplots_adjust(wspace=0.4) 

plt.tight_layout()
plt.savefig("livingspace_rooms.png", dpi=300, bbox_inches="tight")
plt.show()
###############################################
# Calculate percentage of properties with each amenity
amenities = ['Balkon / Terrasse', 'Lift', 'Minergie', 'Aussicht',
             'Garage', 'Parkplatz', 'Kinderfreundlich', 
             'Swimmingpool', 'Haustiere erlaubt']

amenity_percentages = df[amenities].mean() * 100
amenity_percentages = amenity_percentages.sort_values(ascending = False)
print(amenity_percentages.index)



# Create bar plot for amenities
plt.figure(figsize=(12, 6))
sns.barplot(x=amenity_percentages.values, y=amenity_percentages.index,color=sns.color_palette("Set2")[0])
plt.title('Percentage of Properties with Different Amenities')
plt.xlabel('Percentage of Properties')
plt.ylabel( ' ')
plt.tight_layout()
plt.savefig("percentage_properties_with_amenities.png", dpi=300, bbox_inches="tight")
plt.show()




### 4.5 Geographic Distribution of Rental Properties
#
import geopandas as gpd

from io import StringIO
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Get Switzerland's geometry from OSM
url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
response = requests.get(url)
countries = gpd.read_file(StringIO(response.text))
switzerland = countries[countries.ADMIN == 'Switzerland']

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Plot Switzerland with a nice background color
switzerland.plot(ax=ax, color='#f2f2f2', edgecolor='#a6a6a6')

# Create custom colormap for prices
colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
n_bins = 100
cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

# Normalize prices for better color distribution
price_min = df['Price (CHF/Month)'].min()
price_max = df['Price (CHF/Month)'].max()
norm = plt.Normalize(vmin=price_min, vmax=price_max)

# Plot rental properties
scatter = ax.scatter(df['longitude'], df['latitude'],
                    c=df['Price (CHF/Month)'],
                    cmap=cmap,
                    norm=norm,
                    s=100,
                    alpha=0.6,
                    zorder=2)

# Define city positions manually to avoid overlaps
city_positions = {
    'Zurich': (-80, 100),      # Higher up position for Zurich
    'Basel': (-40, 40),      # Upper left
    'Bern': (-50, -50),      # Lower left
    'Winterthur': (40, -50), # Lower right
    'St.Gallen': (70, 20),   # Right
    'Aarau': (-30, 50),      # Left
    'Luzern': (20, -60)      # Bottom
}

# Calculate statistics and add city labels
for city in df['City'].unique():
    city_data = df[df['City'] == city]
    avg_lat = city_data['latitude'].mean()
    avg_lon = city_data['longitude'].mean()
    mean_price = city_data['Price (CHF/Month)'].mean()
    median_price = city_data['Price (CHF/Month)'].median()
    count = len(city_data)
    
    # Add city marker
    ax.plot(avg_lon, avg_lat, 'ko', markersize=8, zorder=3)
    
    # Get predefined position for this city
    x_offset, y_offset = city_positions.get(city, (20, 20))
    
    # Create annotation text
    stats_text = f"{city}\n" \
                 f"Mean: {mean_price:,.0f} CHF\n" \
                 f"Median: {median_price:,.0f} CHF\n" \
                 f"Listings: {count}"
    
    # Special handling for Zurich to ensure it moves up
    if city == 'Zürich':
        y_offset = 100  # Much higher value
        x_offset = 90  # Move more to the right
        connection_style = 'arc3,rad=-0.3'  # Different curve for the arrow
    else:
        connection_style = 'arc3,rad=0.2'



    # Create annotation with arrow
    ax.annotate(
        stats_text,
        (avg_lon, avg_lat),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            alpha=0.8,
            pad=1,
            boxstyle='round,pad=0.5'
        ),
        fontsize=9,
        ha='center',
        va='center',
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.2',
            color='black',
            alpha=0.6
        ),
        zorder=4
    )

# Add title
plt.title('Geographic Distribution of Rental Properties in Switzerland', pad=20, fontsize=12)

# Add overall summary statistics
overall_stats = (
    f"Overall Statistics:\n"
    f"Total Properties: {len(df):,}\n"
    f"Mean Rent: {df['Price (CHF/Month)'].mean():,.0f} CHF\n"
    f"Median Rent: {df['Price (CHF/Month)'].median():,.0f} CHF\n"
    f"Price Range: {df['Price (CHF/Month)'].min():,.0f} - {df['Price (CHF/Month)'].max():,.0f} CHF"
)
plt.text(
    0.02, 0.02, overall_stats,
    transform=ax.transAxes,
    bbox=dict(
        facecolor='white',
        edgecolor='black',
        alpha=0.8,
        pad=1,
        boxstyle='round,pad=0.5'
    ),
    fontsize=9,
    verticalalignment='bottom',
    zorder=4
)

# Set axis labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Focus the view on Switzerland
plt.xlim([5.9559, 10.4921])
plt.ylim([45.8183, 47.8084])

# Add gridlines for better geographic reference
ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

plt.tight_layout()
plt.savefig("map", dpi=300, bbox_inches="tight")
plt.show()
#

# Correlation Matrix
# Select numeric columns for correlation
numeric_columns = ['Price (CHF/Month)', 'Net Price', 'Additional Costs', 'Rooms', 
                  'Living Space', 'Balkon / Terrasse', 'Lift', 'Minergie', 
                  'Aussicht', 'Garage', 'Parkplatz', 'Kinderfreundlich', 
                  'Swimmingpool', 'Haustiere erlaubt', 'oev_erreichb_ewap', 'median_einkommen']

# Create a copy of the DataFrame with only numeric columns
df_numeric = df[numeric_columns].copy()


# Create correlation matrix
matrix = df_numeric.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(matrix, cmap="Greens", annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("geographic_distribution.png", dpi=300, bbox_inches="tight")
plt.show()



########## Modelling ##########

#make a csv file of the final df
df.to_csv(f"df_for_modelling.csv",index=False)

# Copy the dataframe df to a new dataframe df_modelling
df_modelling = pd.read_csv(f"df_for_modelling.csv")

# Prepare features for modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import warnings
warnings.filterwarnings('ignore')




# Select features
continuous_features = [
    'Rooms', 'Living Space', 'oev_erreichb_ewap', 'median_einkommen'
]
categorical_features = ['City', 'Objekttyp']
binary_features = [
    'Balkon / Terrasse', 'Lift', 'Minergie', 'Aussicht', 'Garage',
    'Parkplatz', 'Kinderfreundlich', 'Swimmingpool', 'Haustiere erlaubt'
]

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features + binary_features),
        ('cat', categorical_transformer, categorical_features)
        #('binary', 'passthrough', binary_features)
    ]
)

# Create feature matrices
X = df_modelling[continuous_features + categorical_features + binary_features]
y = df_modelling['Price (CHF/Month)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)



# Create and train the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=100.0))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
relative_error = np.abs(y_test - y_pred) / y_test * 100

print("\nModel Performance Metrics:")
print(f"R² Score: {r2:.3f}")
print(f"Root Mean Square Error: {rmse:.2f} CHF")
print(f"Mean Absolute Error: {mae:.2f} CHF")
print(f"Relative Error: {relative_error}%")

# Scatter plot of predicted vs actual values with perfect prediction line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Price (CHF/Month)')
plt.ylabel('Predicted Price (CHF/Month)')
plt.title('Ridge Regression: Predicted vs Actual Rental Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("predicted_actual_prices.png", dpi=300, bbox_inches="tight")
plt.show()

# Fetaure Importance
numeric_binary_features = continuous_features + binary_features


onehot_feature_names = model.named_steps['preprocessor'] \
                             .named_transformers_['cat'] \
                             .named_steps['onehot'] \
                             .get_feature_names_out(categorical_features)


all_feature_names = numeric_binary_features + list(onehot_feature_names)

coefficients = model.named_steps['regressor'].coef_

import pandas as pd
coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# Show the top features
print(coef_df)


# Plot feature importance to answer first Research Question
import matplotlib.pyplot as plt
import seaborn as sns

# Sort by absolute value of coefficients for strongest influence
coef_df_sorted = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=coef_df_sorted, palette='coolwarm')

plt.axvline(0, color='black', linewidth=1)
plt.title('Key Drivers of Rental Prices in Swiss Cities')
plt.xlabel('Effect on Monthly Rent (CHF)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("key_drivers_rental_prices.png", dpi=300)
plt.show()


# Distribution of residuals
import seaborn as sns

residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Error (Actual - Predicted)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()








