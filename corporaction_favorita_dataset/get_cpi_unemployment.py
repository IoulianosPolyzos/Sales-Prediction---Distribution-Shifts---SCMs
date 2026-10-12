import pandas as pd
import requests


def get_macro_data(start_date='2013-01-01', end_date='2017-08-31'):
    """
    Fetches National CPI and Unemployment data for Ecuador from the World Bank API.
    Expands the data to daily frequency so it can be merged directly with the daily Favorita dataset.
    """
    print("Fetching macroeconomic data from World Bank API...")

    # World Bank API URLs for Ecuador (ECU)
    # FP.CPI.TOTL: Consumer Price Index (2010 = 100)
    # SL.UEM.TOTL.ZS: Unemployment, total (% of total labor force)
    cpi_url = "http://api.worldbank.org/v2/country/ECU/indicator/FP.CPI.TOTL?format=json&date=2012:2018"
    unemp_url = "http://api.worldbank.org/v2/country/ECU/indicator/SL.UEM.TOTL.ZS?format=json&date=2012:2018"

    try:
        # Fetch CPI
        response_cpi = requests.get(cpi_url).json()
        cpi_data = {item['date']: item['value'] for item in response_cpi[1] if item['value'] is not None}

        # Fetch Unemployment
        response_unemp = requests.get(unemp_url).json()
        unemp_data = {item['date']: item['value'] for item in response_unemp[1] if item['value'] is not None}

    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return None

    # Create Annual DataFrame
    df_annual = pd.DataFrame({
        'year': list(cpi_data.keys()),
        'cpi': list(cpi_data.values()),
    })
    df_annual['unemployment'] = df_annual['year'].map(unemp_data)
    df_annual['year'] = df_annual['year'].astype(int)

    # Create a Daily DataFrame for the requested period
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_daily = pd.DataFrame({'date': daily_dates})
    df_daily['year'] = df_daily['date'].dt.year

    # Merge annual values to daily dataframe
    df_macro = pd.merge(df_daily, df_annual, on='year', how='left')

    # Clean up auxiliary columns
    df_macro.drop(columns=['year'], inplace=True)

    print("Successfully mapped CPI and Unemployment to daily frequency.")
    return df_macro