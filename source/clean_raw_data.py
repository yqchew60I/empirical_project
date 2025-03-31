# MAIN DIRECTORY LOCATIONS#
ROOT = "/Users/yqchew/Documents/DataScienceInEconomics/EmpiricalProject/"
RAW_DATA = ROOT+"original_data_sources/"

#IMPORTS#
import numpy as np
import pandas as pd
import os
import openpyxl

#------------------------------------------------------------------#
#EXTRACT UNIQUE MSOAs AND CORRESPONDING LSOAs#
file1 = "Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv"
unique_lsoa_msoa = pd.read_csv(RAW_DATA + file1)
print(unique_lsoa_msoa.columns)

#extract relevant LSOA and corresponding MSOA columns
unique_lsoa_msoa = unique_lsoa_msoa[['LSOA21CD', 'LSOA21NM','MSOA21CD','MSOA21NM']]
print(unique_lsoa_msoa)

#keeping unique LSOAs and their corresponding MSOAs
unique_lsoa_msoa = unique_lsoa_msoa.drop_duplicates()
print(f"Unique LSOAs and corresponding MSOAs:\n {unique_lsoa_msoa}")

#------------------------------------------------------------------#
#COMBINE ALL RECORDED CRIMES#
crime_files_path = RAW_DATA + "2021-04-crimes/"
crime_files = os.listdir(crime_files_path)
print(crime_files)

street_files = [file for file in crime_files if file.endswith('-street.csv')]
print(f'Number of street files: {len(street_files)}')

file2 = "2021-04-avon-and-somerset-street.csv"
first_crime_file = pd.read_csv(crime_files_path + file2)

crime_combine = pd.DataFrame()

for file in street_files:
    file_path = os.path.join(crime_files_path, file)
    df = pd.read_csv(file_path)
    print(f"Rows in {file}: {df.shape[0]}")
    crime_combine = pd.concat([crime_combine, df], ignore_index=True)
print(crime_combine)

#check if number of rows in crime_combine tallies
total_rows = 0 

for file in street_files:
    file_path = os.path.join(crime_files_path, file)
    df = pd.read_csv(file_path)
    total_rows += df.shape[0]

print(f"Total rows from all files: {total_rows}")

#------------------------------------------------------------------#
#TALLY CRIME COUNT FOR EACH LSOA#
lsoa_crime_count = crime_combine.groupby(['LSOA code', 'LSOA name']).size().reset_index(name= 'Crime Count')
print(lsoa_crime_count)

#Add column showing corresponding MSOA
unique_lsoa_msoa.rename(columns={'LSOA21CD':'LSOA code','LSOA21NM':'LSOA name'}, inplace=True)
lsoa_crime_count = pd.merge(lsoa_crime_count, unique_lsoa_msoa, on='LSOA code', how='outer')
print(f"LSOA crime count + corresponding MSOA: \n{lsoa_crime_count}")

#Count number of LSOAs that are missing MSOA data
print(f"Number of LSOAs missing MSOA data: {lsoa_crime_count['MSOA21NM'].isnull().sum()}")

#Assign MSOAs to LSOAs that are missing MSOA data
lsoa_crime_count['LSOA name_x'] = lsoa_crime_count['LSOA name_x'].astype(str)
lsoa_crime_count['LSOA name_x'] = lsoa_crime_count['LSOA name_x'].str.strip()

lsoa_crime_count.loc[lsoa_crime_count['MSOA21NM'].isnull(), 'MSOA21NM'] = lsoa_crime_count['LSOA name_x'].str[:-1]
lsoa_crime_count['MSOA21NM'].fillna(lsoa_crime_count['LSOA name_x'].str[:-1], inplace=True)

# Check if there are still any missing values in 'MSOA21NM' after the update
print(f"Update on Number of LSOAs missing MSOA data: {lsoa_crime_count['MSOA21NM'].isnull().sum()}")

#------------------------------------------------------------------#
#TALLY CRIME COUNT FOR EACH MSOA#
msoa_crime_count = lsoa_crime_count.groupby('MSOA21NM')['Crime Count'].sum().reset_index()
print(msoa_crime_count)

file3 = "MSOA_DEC_2021_EW_NC_v3_-8756108467756097162.xlsx"
census_MSOA = pd.read_excel(RAW_DATA + file3)
print(census_MSOA.head(10))

msoa_valid = pd.merge(msoa_crime_count, census_MSOA, on='MSOA21NM', how='outer')
print(msoa_valid)

#check for invalid MSOAs
print(msoa_valid[msoa_valid['MSOA21CD'].isna()])

#drop MSOAs that do not exist on Census 2021 
msoa_valid = msoa_valid.dropna(subset=['MSOA21CD'])
print(msoa_valid.shape)
