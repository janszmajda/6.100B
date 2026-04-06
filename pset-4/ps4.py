################################################################################
# 6.100B Spring 2026
# Problem Set 4 — Climate Change and Impacts
# Name: Jan Szmajda
# Collaborators: None
# Time: 11
#
# READ ME:
# - Do NOT rename this file or change existing function headers.
# - You may not import additional libraries beyond the ones already here.
# - You may (and should!) define additional helper functions if needed. For any
#   helper functions, include a docstring explaining its purpose, inputs, and outputs.
################################################################################

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry_convert as pc
import pymannkendall as mk
import sklearn

# DO NOT MODIFY ANY OF THE FUNCTION HEADERS BELOW
cheese_consumption_by_year = [
    6.95,
    7.35,
    8.15,
    8.78,
    9.44,
    9.10,
    10.33,
    10.40,
    11.27,
    11.68,
    12.01,
    12.94,
    13.52,
    13.80,
    15.35,
    16.48,
    16.76,
    17.28,
    17.13,
    17.38,
    17.80,
    18.11,
    19.02,
    19.12,
    19.57,
]


def load_data(file_path):
    """
    Loads a CSV or JSON file and returns its contents
    in the form of a DataFrame.

    file_path: path to a file, either .csv or .json

    Returns a DataFrame.
    """
    if ".json" in str(file_path):
        return pd.read_json(file_path)
    elif ".csv" in str(file_path):
        return pd.read_csv(file_path)


def process_temperature_data(df):
    """
    Processes the temperature data as specified in the handout.

    df: a DataFrame

    Returns a DataFrame.
    """
    # define the year columns and use pd.melt as in documentation
    year_cols = [col for col in df.columns if col.isdigit()]
    melted = pd.melt(frame=df, id_vars=["Country", "ISO3"], value_vars=year_cols, var_name="Year", value_name="Temperature")
    return melted

def process_population_data(df):
    """
    Processes the population data as specified in the handout.

    df: a DataFrame

    Returns a DataFrame.
    """
    # rename the columns accordingly and filter the year column between 1961-2024
    df.rename(columns={"Entity":"Country","Code":"ISO3","Population (historical)":"Population"},inplace=True)
    df = df[(df["Year"] >= 1961) & (df["Year"] <= 2024)]
    return df


def process_disaster_data(df):
    """
    Processes the disaster data as specified in the handout.

    df: a DataFrame

    Returns a DataFrame.
    """
    climate_types = ["Drought", "Extreme temperature", "Flood", "Storm", "Wildfire"]

    # filter for climate-related disaster types and exclude TOTAL rows
    mask = df["Indicator"].apply(lambda x: any(t in x for t in climate_types) and "TOTAL" not in x)
    df = df[mask]

    # get year columns
    year_cols = [col for col in df.columns if col.isdigit()]

    # melt to long format
    melted = pd.melt(frame=df,id_vars=["Country", "ISO3"],value_vars=year_cols,var_name="Year",value_name="Count")

    # aggregate total disasters per country-year pair
    aggregated = melted.groupby(["Country", "ISO3", "Year"], as_index=False)["Count"].sum()
    aggregated.rename(columns={"Count": "Total Climate-Related Disasters"}, inplace=True)

    return aggregated[["Country", "ISO3", "Year", "Total Climate-Related Disasters"]]


def country_to_continent(country_code):
    """
    Takes in a three-letter ISO3 code representing a country
    and returns the continent it belongs to in the form of a
    two-letter code.

    country_code: a valid ISO3 country code from one of the
                  three datasets we have provided

    Returns a two-letter code, one of "AF", "AS", "EU",
    "NA", "SA", or "OC".
    """
    # handle ISO3 codes not recognized by pycountry_convert
    special = {"ANT": "NA", "AZO": "EU", "DDR": "EU", "DFR": "EU",
    "ESH": "AF", "PCN": "OC", "SCG": "EU", "SPI": "AS","SUN": "EU",
    "SXM": "NA", "TLS": "AS", "VAT": "EU"}

    if country_code in special:
        return special[country_code]
    alpha2 = pc.country_alpha3_to_country_alpha2(country_code)
    return pc.country_alpha2_to_continent_code(alpha2)



# Implement your code below. Do not leave code in the global scope
# (i.e. outside of functions), unless it's in the main block below.

if __name__ == "__main__":
    df_disasters = load_data("data/disasters.csv")
    df_population = load_data("data/population.json")
    df_temp = load_data("data/temp_change.csv")

    ### Part 1 Test Code ###
    # print(df_disasters)
    # x = process_disaster_data(df_disasters)
    # print(x)

    # print(df_population)
    # x = process_population_data(df_population)
    # print(x)

    # # 1a unique countries
    # print("disasters:", df_disasters['ISO3'].nunique())
    # print("population:", df_population['Code'].nunique())
    # print("temp_change:", df_temp['ISO3'].nunique())

    # # 1a missing values
    # print("\ndisasters missing:")
    # print(df_disasters.isnull().sum()[df_disasters.isnull().sum() > 0])

    # print("\npopulation missing:")
    # print(df_population.isnull().sum()[df_population.isnull().sum() > 0])

    # print("\ntemp_change missing:")
    # print(df_temp.isnull().sum()[df_temp.isnull().sum() > 0])

    ### Part 2 Code ###
    temp_melted = process_temperature_data(df_temp)
    temp_melted["Year"] = temp_melted["Year"].astype(int)
    temp_melted["Temperature"] = pd.to_numeric(temp_melted["Temperature"], errors="coerce")

    # average temperature anomaly per year ignoring NaN
    yearly_avg = temp_melted.groupby("Year")["Temperature"].mean()

    # 5 year moving average using window=5, min_periods=1
    moving_avg = yearly_avg.rolling(window=5, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(yearly_avg.index, yearly_avg.values, label="Annual Mean", color="blue", alpha=0.7)
    plt.plot(moving_avg.index, moving_avg.values, label="5-Year Moving Average", color="orange", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly in Celsius")
    plt.title("Global Mean Temperature Anomaly Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Global Mean Temperature Anomaly Over Time.png")
    plt.show()

    ### Part 3 Code ###
    disaster_data = process_disaster_data(df_disasters)
    disaster_data["Year"] = disaster_data["Year"].astype(int)

    # total disasters per year globally
    yearly_disasters = disaster_data.groupby("Year")["Total Climate-Related Disasters"].sum()

    # only plot years present in both datasets
    common_years = sorted(set(yearly_avg.index) & set(yearly_disasters.index))
    temp_common = yearly_avg.loc[common_years]
    dis_common = yearly_disasters.loc[common_years]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(common_years, temp_common.values, color="blue", label="Avg Temperature Anomaly")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Temperature Anomaly in Celsius", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(common_years, dis_common.values, color="red", label="Total Climate-Related Disasters")
    ax2.set_ylabel("Number of Disasters", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Global Temperature Anomaly vs Climate-Related Disasters")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    plt.savefig("Temperature vs Disasters.png")
    plt.show()
