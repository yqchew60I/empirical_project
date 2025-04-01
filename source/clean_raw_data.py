# MAIN DIRECTORY LOCATIONS#
ROOT = "/Users/yqchew/Documents/DataScienceInEconomics/EmpiricalProject/"
RAW_DATA = ROOT+"original_data_sources/"
SOURCE = ROOT+"source/"

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

#Check if there are still any missing values in 'MSOA21NM' after the update
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

#------------------------------------------------------------------#
#GENERATE NEW VARIABLE UNEMPLOYMENT RATE#
file1 = "census2021-ts066-msoa.csv"
EconomicActivity = pd.read_csv(RAW_DATA + file1)
print(EconomicActivity.columns)

EconomicActivity['Unemployment Rate'] = \
    (EconomicActivity['Economic activity status: Economically active (excluding full-time students): Unemployed']/\
    EconomicActivity['Economic activity status: Economically active (excluding full-time students)'])*100
print(f"EconomicActivity: \n{EconomicActivity}")

#------------------------------------------------------------------#
#GENERATE NEW VARIABLE PROPORTION OF POPLN AGED 65+ #
file1 = "census2021-ts007a-msoa.csv"
AgeGroup = pd.read_csv(RAW_DATA + file1)
print(AgeGroup.columns)

AgeGroup['Age_abv65'] = AgeGroup['Age: Aged 65 to 69 years'] \
    + AgeGroup['Age: Aged 70 to 74 years'] \
    + AgeGroup['Age: Aged 75 to 79 years'] \
    + AgeGroup['Age: Aged 80 to 84 years'] \
    + AgeGroup['Age: Aged 85 years and over']
print(f"AgeGroup: \n {AgeGroup.head(10)}")

AgeGroup['Proportion_age_abv65'] = \
    (AgeGroup['Age_abv65']/AgeGroup['Age: Total'])*100
print(f"AgeGroup: \n {AgeGroup.head(10)}")

#------------------------------------------------------------------#
#GENERATE NEW VARIABLE PROPORTION OF WHITE POPULATION #
file1 = "census2021-ts021-msoa.csv"
Ethnicity = pd.read_csv(RAW_DATA + file1)
print(Ethnicity.columns)

Ethnicity['Proportion_white'] = \
    (Ethnicity['Ethnic group: White']/ \
    Ethnicity['Ethnic group: Total: All usual residents'])*100
print(f"Ethicity: {Ethnicity.head(10)}")


#------------------------------------------------------------------#
#CREATING FINAL DATASET#
emp_proj_dataset_clean = msoa_valid.copy()
emp_proj_dataset_clean= emp_proj_dataset_clean.drop(columns = ['MSOA21NMW', 'ObjectId'])

new_column_order = ['MSOA21NM', 'MSOA21CD', 'Crime Count']
emp_proj_dataset_clean = emp_proj_dataset_clean[new_column_order]

#Merge unemployment data to final dataset
EconomicActivity = EconomicActivity.rename(columns={'geography': 'MSOA21NM'})
EconomicActivity = EconomicActivity.rename(columns={'geography code': 'MSOA21CD'})
print(EconomicActivity.columns)

emp_proj_dataset_clean = pd.merge(
    emp_proj_dataset_clean, EconomicActivity, on=['MSOA21NM', 'MSOA21CD'],
    how='outer')

#Keep only unemployment variable from EconomicActivity
emp_proj_dataset_clean = emp_proj_dataset_clean[['MSOA21NM','MSOA21CD','Crime Count', 'Unemployment Rate']]
print(emp_proj_dataset_clean)

#Merge proportion of population age 65+ data to final dataset
AgeGroup = AgeGroup.rename(columns={'geography': 'MSOA21NM'})
AgeGroup = AgeGroup.rename(columns={'geography code': 'MSOA21CD'})

emp_proj_dataset_clean = pd.merge(
    emp_proj_dataset_clean, AgeGroup, 
    on=['MSOA21NM', 'MSOA21CD'], how='outer')

#Keep only proportion of popln age 65+ variable from AgeGroup
emp_proj_dataset_clean=\
    emp_proj_dataset_clean[['MSOA21NM','MSOA21CD',
    'Crime Count', 'Unemployment Rate', 'Proportion_age_abv65']]
print(emp_proj_dataset_clean)

#Merge proportion of white popln to final dataset
Ethnicity = Ethnicity.rename(columns={'geography': 'MSOA21NM'})
Ethnicity = Ethnicity.rename(columns={'geography code': 'MSOA21CD'})

#Keep only total popln and proportion of white popln from Ethnicity
Ethnicity= Ethnicity[['MSOA21NM', 'MSOA21CD', 
'Ethnic group: Total: All usual residents', 'Proportion_white']]

#Merge Ethnicity to final dataset
emp_proj_dataset_clean=\
    pd.merge(emp_proj_dataset_clean, Ethnicity,
    on=['MSOA21CD', 'MSOA21NM'], how='outer')

#Merge popln density to final dataset
file1 = "census2021-ts006-msoa.csv"
Population_Density = pd.read_csv(RAW_DATA + file1)

Population_Density = Population_Density.rename(
    columns={'geography': 'MSOA21NM'})
Population_Density = Population_Density.rename(
    columns={'geography code': 'MSOA21CD'})
print(f"Popln Density: \n{Population_Density}")

emp_proj_dataset_clean= pd.merge(
    emp_proj_dataset_clean, Population_Density,
    on=['MSOA21CD', 'MSOA21NM'], how='outer')

emp_proj_dataset_clean= emp_proj_dataset_clean.drop(
    columns=['date'])

emp_proj_dataset_clean= emp_proj_dataset_clean.rename(
    columns={'Population Density: Persons per square kilometre; measures: Value': 
    'Population Density: Persons per square kilometre'})

#Generate crime rate per 1000 resident
emp_proj_dataset_clean['Crime Rate per 1000 resident'] = \
(emp_proj_dataset_clean['Crime Count']/ \
 emp_proj_dataset_clean['Ethnic group: Total: All usual residents'])*1000

print(f"All: \n{emp_proj_dataset_clean}")

emp_proj_dataset_clean.to_csv(SOURCE + 'emp_proj_data.csv', index=False)