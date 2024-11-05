from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import yfinance as yf
import requests
import psycopg2
import json
from sec_edgar_api import EdgarClient
import openmeteo_requests
import requests_cache
from retry_requests import retry
from bs4 import BeautifulSoup
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from cache_system import FileCache, APICache
from utils import api_get, nan_series_handling, analyze_sentiment_finbert, winsorize_data, drop_corr_pair1
import sys
import ast
import re
from time import sleep
from datetime import datetime, timedelta
from functools import reduce
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

class DataSource(ABC):
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        pass

class DatabaseInterface(ABC):
    @abstractmethod
    def initialize_database(self) -> None:
        pass
    
    @abstractmethod
    def check_data_exists(self, entity: str, start_date: datetime=None, end_date: datetime=None) -> bool:
        pass

   
class PostgresDatabase(DatabaseInterface):
    def __init__(self, username: str, password: str, host: str, port: str, database: str, schema: str, dir: str) -> None:
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.schema_name = schema
        self.local_dir = dir

    def _connect(self) -> None:
        # Helper method to create a connection and cursor
        conn = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port
        )
        return conn, conn.cursor()
    
    def database_query_execute(self, query: str, params: tuple = None) -> Any:
        conn, cur = self._connect()
        try:
            if params:
                if isinstance(params, list) and len(params) > 1: # "bulk insert" ... these would be "records"
                    cur.executemany(query, params)
                else: # single record to be inserted
                    cur.execute(query, params)
            else:
                cur.execute(query)
            query_meta = query.strip().split('(')[0].strip().replace(';', '')
            if query.strip().lower().startswith(('insert', 'create', 'update', 'delete')):
                conn.commit()
                # query_meta = query.strip().split('(')[0].strip().replace(';', '')
                # print(f'{query_meta} query executed and committed successfully.')
            elif query.strip().lower().startswith("select"):
                # print(f'Select query ({query_meta}) executed successfully.')
                # get records
                rows = cur.fetchall()

                # Retrieve column names from the cursor description
                column_names = [desc[0] for desc in cur.description]

                # Create a DataFrame from the fetched data with column names
                df = pd.DataFrame(rows, columns=column_names)

                return df
            else:
                print('Query executed successfully.')
        except Exception as e:
            print('Query failed.')
            print(e)
            if conn:
                conn.rollback()  # Rollback in case of error
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def create_schema(self) -> None:
        create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};"
        self.database_query_execute(create_schema_query)

    def create_initial_tables(self) -> None:
    # def create_initial_tables(self, queries) -> None:
        create_securities_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.securities (
                security_id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL
            );
        """

        create_eia_info_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.eia_code_info (
                id SERIAL PRIMARY KEY,
                series_name VARCHAR(200) UNIQUE NOT NULL,
                description TEXT NOT NULL,
                api_call BOOLEAN NOT NULL,
                root VARCHAR(500),
                frequency VARCHAR(50),
                data_col VARCHAR(50),
                sort VARCHAR(50),
                sort_direction VARCHAR(50),
                offset_value INTEGER,
                length INTEGER,
                facets TEXT
            );
        """

        create_factor_dict_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.factor_dict (
                factor_id SERIAL PRIMARY KEY,
                factor_desc VARCHAR(200) UNIQUE NOT NULL,
                data_source VARCHAR(200) NOT NULL
            );
        """

        create_factor_ts_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.factor_time_series (
                id SERIAL PRIMARY KEY,
                end_date DATE NOT NULL,
                load_date DATE NOT NULL,
                factor_id INTEGER REFERENCES {self.schema_name}.factor_dict(factor_id),
                ts_date DATE NOT NULL,
                factor_value NUMERIC NOT NULL,
                UNIQUE (end_date, load_date, factor_id, ts_date)
            );
        """

        create_security_factor_map_query = f"""
            CREATE TABLE IF NOT EXISTS {self.schema_name}.security_factor_map (
                id SERIAL PRIMARY KEY,
                security_id INTEGER REFERENCES {self.schema_name}.securities(security_id),
                factor_id INTEGER REFERENCES {self.schema_name}.factor_dict(factor_id),
                version INTEGER NOT NULL
            );
        """


        queries = [
                   create_securities_table_query,
                   create_eia_info_table_query,
                   create_factor_dict_table_query,
                   create_factor_ts_table_query,
                   create_security_factor_map_query
                   ]
        
        for query in queries:
            self.database_query_execute(query)
    
    def populate_eia_code_info_table(self):
        try:
            df = pd.read_csv(os.path.join(self.local_dir, 'eia_code_info.csv'))
            df = df.where(pd.notna(df), None)
            df['offset_value'] = nan_series_handling(df['offset_value'], data_type='int') # need to handle the obscure issue
            df['length'] = nan_series_handling(df['length'], data_type='int') # same as above

            insert_query = f"""
                INSERT INTO {self.schema_name}.eia_code_info (series_name, description, api_call, root, frequency, data_col, sort, sort_direction, offset_value, length, facets)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (series_name) 
                DO NOTHING;
            """
            # df_records  = df.to_records('dict')
            df_records = list(df.itertuples(index=False, name=None))
            for record in df_records:
                self.database_query_execute(insert_query, record)

        except ValueError as value_error:
            print(f'Likely a data value error: {value_error}')

    def get_or_create_id_for_factor(self, desc_value: str, data_source: str) -> int:
        # Try to get the ID for the given value
        query_select = f"SELECT factor_id FROM {self.schema_name}.factor_dict WHERE factor_desc = %s"
        result = self.database_query_execute(query_select, (desc_value,))
        
        if not result.empty:
            return result.factor_id.values[0]
        else:
            # if value not found, insert a new record and get the new ID
            query_insert = f"""INSERT INTO {self.schema_name}.factor_dict (factor_desc, data_source) 
                             VALUES (%s, %s)
                             ON CONFLICT (factor_desc)
                             DO NOTHING;
                             """
            self.database_query_execute(query_insert, (desc_value, data_source))

            # now get id
            new_id = self.database_query_execute(query_select, (desc_value,))
            return new_id.factor_id.values[0]


    def insert_into_factor_ts_table(self, df: pd.DataFrame, data_source: str):
        end_date = df['period'].iloc[-1].strftime('%Y-%m-%d')
        load_date = datetime.now().strftime('%Y-%m-%d')

        for col in [col for col in df.columns if col != 'period']:
            print(f'Inserting {col} into {self.schema_name}.factor_time_series...')
            factor_ts = df[['period', col]]
            factor_id = self.get_or_create_id_for_factor(col, data_source)
            factor_ts.insert(0, 'factor_id', factor_id)
            factor_ts.insert(0, 'load_date', load_date)
            factor_ts.insert(0, 'end_date', end_date)
            factor_ts = factor_ts.rename(columns={'period': 'ts_date', col: 'factor_value'})
            
            insert_query = f"""
                INSERT INTO {self.schema_name}.factor_time_series (end_date, load_date, factor_id, ts_date, factor_value)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT
                DO NOTHING
            """
            self.database_query_execute(insert_query, list(factor_ts.itertuples(index=False, name=None)))

    def initialize_database(self) -> None:
        self.create_schema()
        self.create_initial_tables()
        self.populate_eia_code_info_table()
    
    def check_data_exists(self, entity: str, start_date: datetime, end_date: datetime) -> bool:
        pass

    def load_data(self, data: List[Dict[str, Any]], entity: str) -> None:
        pass

    def get_eia_code_info(self) -> pd.DataFrame:
        return self.database_query_execute(f"SELECT * FROM {self.schema_name}.eia_code_info")


class EIADataSource(DataSource):
    def __init__(self, dir: str, eia_codes: pd.DataFrame, api_key: str = None) -> None:
        self.elec_data_dir = dir
        self.api_key = api_key or os.get_env("EIA_API_KEY")
        self.eia_codes = eia_codes

    def get_eia_api_data(self, series_name: str, root: str, frequency: str, data_col: str, sort: str, \
                            sort_direction: str, offset: int, length: int, facets: List[Any]) -> pd.DataFrame:
        
        master_df = pd.DataFrame()
        data_length = 1
        offset = 0
        while data_length > 0:
            url_prefix = f'{root}?api_key={self.api_key}&frequency={frequency}&data[0]={data_col}'
            if facets:
                for facet in facets:
                    url_prefix += f'&facets[{facet[0]}][]={facet[1]}'

            url_suffix = f'&sort[0][column]={sort}&sort[0][direction]={sort_direction}&offset={offset}&length={length}'

            url = url_prefix + url_suffix

            data = api_get(url)
            data_length = len(data['response']['data'])
            if data_length > 0:
                df = pd.DataFrame(data['response']['data'])
                master_df = pd.concat([master_df, df], axis=0)
            offset += 1*5000

        master_df = master_df.sort_values(by='period', ascending=True).reset_index(drop=True)
        master_df['value'] = pd.to_numeric(master_df['value'])
        master_df['period'] = pd.to_datetime(master_df['period'])

        master_df = (master_df
                    .rename(columns={'value': series_name})
                    .sort_values(by='period')
                    .set_index('period'))[[series_name]]

        return master_df
     
    
    def scrape_new_elec_spot_prices(self) -> None:
        pass # TODO
    
    def get_flat_file_elec_spot_prices(self) -> pd.DataFrame:
        elec_spot_prices = pd.DataFrame()
        for filename in os.listdir(os.path.join(self.elec_data_dir, f'elec_spot_price_history')):
            print('     Processing: ' + filename)
            file_path = os.path.join(self.elec_data_dir, f'elec_spot_price_history/{filename}')
            df = pd.read_excel(file_path, sheet_name=0)
            if filename != 'Mass Hub.xlsx':
                df = df.rename(columns={'Price hub': 'Price Hub', 'Trade date': 'Trade Date', 'Delivery start date': 'Delivery Start Date', 'Delivery \nend date': 'Delivery End Date',
                                        'High price $/MWh': 'High Price $/MWh', 'Low price $/MWh': 'Low Price $/MWh', 'Wtd avg price $/MWh': 'Wtd Avg Price $/MWh',
                                        'Daily volume MWh': 'Daily Volume MWh', 'Number of trades': 'Number of Trades', 'Number of counterparties': 'Number of Companies'})
                df = df[df['Price Hub']=='Nepool MH DA LMP Peak']
            elec_spot_prices = pd.concat([elec_spot_prices, df], axis=0)
        elec_spot_prices = elec_spot_prices.drop_duplicates(subset=['Trade Date'], keep='last').reset_index(drop=True)
        elec_spot_prices['Trade Date'] = pd.to_datetime(elec_spot_prices['Trade Date'])
        elec_spot_prices = (elec_spot_prices
                            .sort_values(by='Trade Date')
                            .reset_index(drop=True)
                            .rename(columns={'Trade Date': 'period', 'Wtd Avg Price $/MWh': 'elec_spot_price'})
                            .set_index('period'))[['elec_spot_price']]
        
        return elec_spot_prices
    
    def fetch_data(self, start_time: datetime=None, end_time: datetime=None) -> pd.DataFrame:

        eia_dfs = []

        for _, data_item in self.eia_codes.iterrows():
            series_name = data_item['series_name']
            print(f'Acquiring: {series_name}')
            api_call = data_item['api_call']
            root = data_item['root']
            frequency = data_item['frequency']
            data_col = data_item['data_col']
            sort = data_item['sort']
            sort_direction = data_item['sort_direction']
            offset = int(data_item['offset_value']) if not np.isnan(data_item['offset_value']) else np.nan
            length = int(data_item['length']) if not np.isnan(data_item['length']) else np.nan
            facets = ast.literal_eval(data_item['facets']) if data_item['facets'] else None

            if api_call:
                eia_data = self.get_eia_api_data(series_name, root, frequency, data_col, \
                                                 sort, sort_direction, offset, length, facets)
            else:
                # flat files
                eia_data = self.get_flat_file_elec_spot_prices()
            
            eia_dfs.append(eia_data)

        df = pd.concat(eia_dfs, axis=1, join='inner')
        df = df.reset_index()
        df['period'] = pd.to_datetime(df['period'], errors='coerce')
        df = df.sort_values(by='period')

        return df



class MeteoDataSource(DataSource):
    def __init__(self, coordinates: List[Tuple], timezone: str = 'America/New_York', frequency: str = 'daily', start_date: str = '1986-01-01', \
                    end_date: str = datetime.now().strftime('%Y-%m-%d'), \
                    params: List[str] = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'sunshine_duration', \
                                            'precipitation_sum', 'rain_sum', 'snowfall_sum', 'wind_speed_10m_max']) -> None:
        self.timezone = timezone
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.params = params
        self.coordinates = coordinates

    def get_open_meteo_data(self, latitude, longitude):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            self.frequency: self.params,
            "timezone": self.timezone
        }
        try:
            response = openmeteo.weather_api(url, params=params)

            return response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def open_meteo_postprocessing(self, response):
        # Process first location. Add a for-loop for multiple locations or weather models
        response = response[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
        print("\n")

        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(5).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(6).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(7).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["sunshine_duration"] = daily_sunshine_duration
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["rain_sum"] = daily_rain_sum
        daily_data["snowfall_sum"] = daily_snowfall_sum
        daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

        daily_dataframe = pd.DataFrame(data = daily_data)
        # daily_dataframe['date'] = pd.to_datetime(daily_dataframe['date'])
        daily_dataframe = (daily_dataframe
                            .rename(columns={'date': 'period'})
                            .set_index('period'))

        return daily_dataframe
    
    def get_weather_dfs(self) -> List[pd.DataFrame]:
        weather_dfs = []
        for lat, lon in self.coordinates:
            weather_response = self.get_open_meteo_data(lat, lon)
            weather_df = self.open_meteo_postprocessing(weather_response)
            weather_dfs.append(weather_df)
            sleep(1.5)

        return weather_dfs
    
    def fetch_data(self, weights=None):
        weather_dfs = self.get_weather_dfs()
        if weights:
            weather_dfs = [self.weighted_avg_dataframes(weather_dfs, weights)]

        return pd.concat(weather_dfs, axis=1)

    @staticmethod
    def weighted_avg_dataframes(dfs: List[pd.DataFrame], weights: List[float]):
        """
        Compute the weighted average of multiple DataFrames dynamically with input validation.
        
        Parameters:
        dfs (list of pd.DataFrame): List of DataFrames to average.
        weights (list of float): List of weights corresponding to each DataFrame. Weights should sum to 1 and be between 0 and 1.
        
        Returns:
        pd.DataFrame: A DataFrame with the weighted average of the input DataFrames.
        
        Raises:
        ValueError: If weights don't sum to 1, contain values outside [0, 1], or lengths of dfs and weights don't match.
        """
        # Ensure the lengths of dfs and weights match
        assert len(dfs) == len(weights), "The number of DataFrames must match the number of weights."
        
        # Ensure all weights are between 0 and 1
        assert all(0 <= w <= 1 for w in weights), "All weights must be between 0 and 1."
        
        # Ensure weights sum to 1, allowing for a small tolerance due to floating-point precision
        assert abs(sum(weights) - 1) < 1e-6, "Weights must sum to 1."
        
        # Initialize the weighted sum with the first DataFrame multiplied by its weight
        weighted_sum = dfs[0].copy()
        weighted_sum *= weights[0]
        
        # Iterate over the remaining DataFrames and add them to the weighted sum
        for i in range(1, len(dfs)):
            weighted_sum += dfs[i] * weights[i]
        
        return weighted_sum




class AlphaVantageDataSource(DataSource):
    def __init__(self, api_key: str, ticker: str, url: str = 'https://www.alphavantage.co/query') -> None:
        self.api_key = api_key
        self.ticker = ticker
        self.url = url

    def fetch_data(self):
        fundamental_data = self._fetch_from_api()
        processed_df = self._process_data(fundamental_df=fundamental_data)
        return processed_df

    def _fetch_from_api(self) -> pd.DataFrame:
        dfs = []
        for statement in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']:
            params = {
                'symbol': self.ticker,
                'apikey': self.api_key,
                'function': statement
            }
            result = requests.get(self.url, params=params)
            data = result.json()
            # data = get_alpha_vantage_data(statement, ticker, token)
            df = pd.DataFrame(data['quarterlyReports'])
            df = df.drop('reportedCurrency', axis=1)
            dfs.append(df)
        return pd.concat(dfs, axis=1)


    def _process_data(self, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        fundamental_df = (fundamental_df
                    .loc[:, ~fundamental_df.columns.duplicated()] # drop latter two occurrences of fiscalDateEnding & netIncome
                    .set_index('fiscalDateEnding')
                    .replace('None', np.nan) # lot of 'None' strings
                    .astype(float))
    
        # drop columns of data that have a significant (> 25%) of data missing
        none_percentage = 100*fundamental_df.isna().mean().sort_values(ascending=False)
        columns_above_25_missing = none_percentage[none_percentage > 25].index
        fundamental_df = fundamental_df.drop(columns=columns_above_25_missing)

        # drop columns of data that have significant levels (> 25%) of 0s
        zero_percentage = ((fundamental_df == 0.0).mean() * 100).sort_values(ascending=False)
        columns_above_25_zeros = zero_percentage[zero_percentage > 25].index
        fundamental_df = fundamental_df.drop(columns=columns_above_25_zeros)

        # forward fill missing values and then backfill beginning missing values
        fundamental_df = fundamental_df.sort_index(ascending=True)
        fundamental_df = fundamental_df.loc['2009-06-30':] # this is the latest start date between IS, BS, and CF
        fundamental_df = fundamental_df.ffill().bfill()

        # additional ratios/metrics from fundamental data
        fundamental_df['netProfitMargin'] = fundamental_df['netIncome']/fundamental_df['totalRevenue']
        fundamental_df['ROA'] = fundamental_df['netIncome']/fundamental_df['totalAssets']
        fundamental_df['ROE'] = fundamental_df['netIncome']/fundamental_df['totalShareholderEquity']
        fundamental_df['debtEquityRatio'] = fundamental_df['shortLongTermDebtTotal']/fundamental_df['totalShareholderEquity']
        fundamental_df['debtAssetRatio'] = fundamental_df['shortLongTermDebtTotal']/fundamental_df['totalAssets']
        fundamental_df['interestCoverageRatio'] = fundamental_df['ebit']/fundamental_df['interestExpense']
        fundamental_df['assetTurnoverRatio'] = fundamental_df['totalRevenue']/fundamental_df['totalAssets']
        fundamental_df['currentRatio'] = fundamental_df['totalCurrentAssets']/fundamental_df['totalCurrentLiabilities']
        fundamental_df['quickRatio'] = (fundamental_df['totalCurrentAssets'] - fundamental_df['inventory'])/fundamental_df['totalCurrentLiabilities']
        fundamental_df['cashRatio'] = fundamental_df['cashAndCashEquivalentsAtCarryingValue']/fundamental_df['totalCurrentLiabilities']
        fundamental_df['capitalExpenditureEfficiency'] = fundamental_df['operatingCashflow']/fundamental_df['capitalExpenditures']
        fundamental_df['debtServiceCoverageRatio'] = fundamental_df['operatingIncome']/fundamental_df['interestAndDebtExpense']
        
        fundamental_df.index.name = 'period'
        fundamental_df = fundamental_df.reset_index()
        fundamental_df['period'] = pd.to_datetime(fundamental_df['period'])
        
        return fundamental_df


@dataclass
class Filing:
    accession_number: str
    report_date: str
    form: str
    primary_document: str
    cik: str
   
    @property
    def accession_number_cleaned(self) -> str:
        """Remove dashes from accession number."""
        return self.accession_number.replace('-', '')
    
    def get_edgar_url(self) -> str:
        """
        Generate the EDGAR URL for the filing.
        Format: https://www.sec.gov/Archives/edgar/data/CIK/AccessionNumber/PrimaryDocument
        """
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession_number_cleaned}/{self.primary_document}"
    
    def get_index_url(self) -> str:
        """
        Generate the URL for the filing's index page.
        Format: https://www.sec.gov/Archives/edgar/data/CIK/AccessionNumber/index.json
        """
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession_number_cleaned}/index.json"


@dataclass
class Coordinate:
    latitude: float
    longitude: float
    
    def __post_init__(self):
        # Basic validation
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)


def filter_filings(data: dict, cik: str, form_types: set = {'10-Q', '10-K'}) -> List[Filing]:
    """
    Filter SEC filings data to extract matched pairs of information for specific form types.
    
    Args:
        data: Dictionary containing SEC filings data
        form_types: Set of form types to filter for (default: {'10-Q', '10-K'})
    
    Returns:
        List of Filing objects containing matched information
    """
    recent = data['filings']['recent']
    
    # Get indices of forms that match our criteria
    matching_indices = [
        i for i, form in enumerate(recent['form'])
        if form in form_types
    ]
    
    # Create Filing objects for matched entries
    filtered_filings = [
        Filing(
            accession_number=recent['accessionNumber'][i],
            report_date=recent['reportDate'][i],
            form=recent['form'][i],
            primary_document=recent['primaryDocument'][i],
            cik=cik
        )
        for i in matching_indices
    ]
    
    return filtered_filings


class EDGARDataSource(DataSource):
    # Exact section markers
    SECTION_PATTERNS = {
        '10-K': {
            'start': r"(?:\*\*\s*)?ITEM\s*7\.?\s*(?:MANAGEMENT[’']S|MANAGEMENT'S)\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS\s*(?:\*\*)?(?!\s*\")",
            'end': r"(?:\*\*\s*)?ITEM\s*7A\.?\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK\s*(?:\*\*)?(?!\s*\")"
        },
        '10-Q': {
            'start': r"(?:\*\*\s*)?ITEM\s*2\.?\s*(?:MANAGEMENT[’']S|MANAGEMENT'S)\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+(?:RESULTS\s+OF\s+OPERATIONS\s+AND\s+FINANCIAL\s+CONDITION|FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS)\s*(?:\*\*)?(?!\s*\")",
            'end': r"(?:\*\*\s*)?ITEM\s*3\.?\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK\s*(?:\*\*)?(?!\s*\")"
        }
    }
    
    def __init__(self, user_agent: str, cik: str) -> None:
        """
        Initialize the EDGAR extractor.
        
        Args:
            user_agent: Your user agent string for SEC EDGAR requests
                       Format: "Your Name your.email@domain.com"
        """
        self.user_agent = user_agent
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        self.cik = cik
    
    def get_filing_metadata(self) -> Dict:
        edgar = EdgarClient(user_agent=self.user_agent)
        result = edgar.get_submissions(cik=self.cik)
        return result

    def get_filing_content(self, filing: Filing) -> Optional[str]:
        """Fetch the content of a filing from EDGAR."""
        try:
            sleep(0.1)  # Respect SEC's rate limiting
            response = requests.get(
                filing.get_edgar_url(),
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {filing.get_edgar_url()}: {str(e)}")
            return None
    
    def find_section_indices(self, text: str, form_type: str) -> Optional[Tuple[int, int]]:
        """Find the start and end indices of the MD&A section, using the second occurrence for 10-Q forms."""
        patterns = self.SECTION_PATTERNS.get(form_type)
        if not patterns:
            return None
        
        # Find all occurrences of start and end markers
        text = re.sub(r'[\xa0\t]+', ' ', text)
        start_matches = list(re.finditer(patterns['start'], text, re.IGNORECASE))
        end_matches = list(re.finditer(patterns['end'], text, re.IGNORECASE))
        
        if not start_matches:
            print(f"Could not find start pattern: {patterns['start']}")
            return None
        if not end_matches:
            print(f"Could not find end pattern: {patterns['end']}")
            return None
        
        # Try to use the second occurrence for 10-Q, fallback to the first if not available
        if form_type == '10-Q':
            if len(start_matches) >= 2:
                start_idx = start_matches[1].start()  # Use the second occurrence if available
            else:
                print("Only one start pattern found, using the first occurrence")
                start_idx = start_matches[0].start()  # Fallback to first if second isn't available
        else:
            # For 10-K, we use the first occurrence
            start_idx = start_matches[0].start()
        
        # Now find the end marker, fallback to first occurrence if second isn't found
        if form_type == '10-Q':
            # Filter end matches that occur after the start index
            filtered_end_matches = [match for match in end_matches if match.start() > start_idx]
            if len(filtered_end_matches) > 0:
                end_idx = filtered_end_matches[0].start()  # Use the first end match after the start
            else:
                print("Could not find a valid end pattern after the start")
                return None
        else:
            # For 10-K, find the first end marker that comes after the start marker
            end_idx = None
            for match in end_matches:
                if match.start() > start_idx:
                    end_idx = match.start()
                    break
        
        if end_idx is None:
            print("Found start but no matching end marker")
            return None
        
        return (start_idx, end_idx)

    def extract_mda_section(self, html_content: str, form_type: str) -> Optional[str]:
        """Extract MD&A section from the filing."""
        if not html_content:
            return None
            
        # Convert HTML to plain text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
            
        # Get text content
        text = ' '.join(soup.stripped_strings)
        
        # Find section boundaries
        indices = self.find_section_indices(text, form_type)
        if not indices:
            return None
            
        # Extract and clean the section
        section_text = text[indices[0]:indices[1]].strip()
        
        # Clean up whitespace
        section_text = re.sub(r'\s+', ' ', section_text)
        
        return section_text

    def create_mda_df(self, filings: List[Filing]) -> pd.DataFrame:
        """
        Create a DataFrame containing MD&A sections from the filings.
        
        Args:
            filings: List of Filing objects
            user_agent: User agent string for SEC EDGAR requests
        
        Returns:
            DataFrame with columns: Date, Form, Text
        """
        data = []
        
        for filing in filings:
            print(f"Processing {filing.form} from {filing.report_date}...")
            content = self.get_filing_content(filing)
            if content:
                section_text = self.extract_mda_section(content, filing.form)
                if section_text:
                    print("\nExample text snippet from first filing:")
                    print(section_text[:200] + "...")
                    data.append({
                        'Date': filing.report_date,
                        'Form': filing.form,
                        'Text': section_text
                    })
                else:
                    print(f"Could not find MD&A section in {filing.form} from {filing.report_date}")
        
        return pd.DataFrame(data)


    def fetch_data(self):

        # edgar_extractor = EDGARDataSource(user_agent=user_agent, cik=cik) # cik pulled from internet for DUK
        duk_filing_json = self.get_filing_metadata()
        duk_filings = filter_filings(data=duk_filing_json, cik=self.cik)
        text_df = self.create_mda_df(duk_filings)
        text_df.set_index('Date', inplace=True)
        text_df.sort_index(ascending=True, inplace=True)

        return text_df

class YahooFinanceDataSource(DataSource):
    def __init__(self, symbols: List[str]) -> None:
        self.symbols = symbols

    def fetch_data(self, start_date: datetime = datetime(1986,1,1), end_date: datetime = datetime.today().date()) -> pd.DataFrame:
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        data = yf.download(self.symbols, start=start_date, end=end_date)

        return data
    

class APIQuotaExceededError(Exception):
    """Custom exception for when API quota/rate limits are exceeded"""
    def __init__(self, message="Daily API quota has been exceeded", response=None):
        self.message = message
        self.response = response
        super().__init__(self.message)





def build_10QK_sentiment_scores(df):
    for date in df.index:
        extracted_text = df.loc[date, 'Text']
        cleaned_text = re.sub(r"\n|&#[0-9]+;", "", extracted_text)
        sentiment_score = analyze_sentiment_finbert(cleaned_text)
        df.loc[date, 'sentiment_score'] = sentiment_score
    
    sentiments = df[['sentiment_score']]

    return sentiments


class BaseDataCache(FileCache):
    """Base caching class with vendor-specific configurations"""
    def __init__(self, 
                 vendor_name: str,
                 cache_dir: str = os.path.join(os.path.dirname(__file__), "../data/cache"),
                 cache_duration: Optional[timedelta] = None):
        # Create vendor-specific cache directory
        vendor_cache_dir = os.path.join(cache_dir, vendor_name)
        super().__init__(cache_dir=vendor_cache_dir, cache_duration=cache_duration)
        self.vendor_name = vendor_name

class DataSourceWithCache(DataSource):
    def __init__(self, data_source: DataSource, cache_handler: BaseDataCache):
        if not isinstance(data_source, DataSource):
            raise TypeError("data_source must implement DataSource interface")
        self._data_source = data_source
        self._cache_handler = cache_handler

    def fetch_data(self, **kwargs) -> pd.DataFrame:
        # Create a cache key from the parameters
        cache_key = f"fetch_data"
        
        try:
            return self._cache_handler.process_with_cache(
                cache_key=cache_key,
                process_func=self._data_source.fetch_data,
                **kwargs
            )
        except Exception as e:
            # Try to get cached data as fallback
            cached_data = self._cache_handler.get_cached_result(cache_key)
            if cached_data is not None:
                print(f"Retrieved cached data for {cache_key}")
                return cached_data
            raise e
        
    def __getattr__(self, name):
        # Get the original method from DataSource
        attr = getattr(self._data_source, name)
        
        if callable(attr):
            # If it's a method, wrap it with caching
            def wrapped_method(*args, **kwargs):
                # Create a cache key from method name and arguments
                cache_key = f"{name}_{'_'.join(str(arg) for arg in args)}_{'_'.join(f'{k}_{v}' for k,v in kwargs.items())}"
                
                try:
                    return self._cache_handler.process_with_cache(
                        cache_key=cache_key,
                        process_func=attr,
                        *args,
                        **kwargs
                    )
                except Exception as e:
                    # If the process fails, try to get cached data
                    cached_data = self._cache_handler.get_cached_result(cache_key)
                    if cached_data is not None:
                        print(f"Retrieved cached data for {name}")
                        return cached_data
                    raise e
                    
            return wrapped_method
        return attr


# Example of how to set up caching for multiple vendors
class VendorCacheFactory:
    """Factory class to manage different vendor cache instances"""
    def __init__(self):
        self._cache_handlers = {}

    def get_cache_handler(self, vendor_name: str, cache_duration: Optional[timedelta] = None) -> BaseDataCache:
        """Get or create a cache handler for a specific vendor"""
        if vendor_name not in self._cache_handlers:
            self._cache_handlers[vendor_name] = BaseDataCache(
                vendor_name=vendor_name,
                cache_duration=cache_duration
            )
        return self._cache_handlers[vendor_name]






def main():
    load_dotenv()
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_schema = os.getenv("DB_SCHEMA")
    eia_api_key = os.getenv("EIA_API_KEY")
    av_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    local_data_dir = os.getenv("LOCAL_DATA_DIR")
    email = os.getenv("EMAIL")
    full_name = os.getenv("FULL_NAME")

    cik = "1326160"
    ticker = "DUK"

    # table_create_queries = ... --> then feed to PostgresDatabase object
    db = PostgresDatabase(db_username, db_password, db_host, db_port, db_name, db_schema, local_data_dir)
    db.initialize_database()

    # setup cache factory to manage api calls that may fail, expensive api calls (defer to flat file if exists)
    cache_factory = VendorCacheFactory()

    # get Alpha Vantage fundamental data
    alphavantage_cache = cache_factory.get_cache_handler("alphavantage", timedelta(hours=24))
    fundamental_source = AlphaVantageDataSource(api_key=av_api_key, ticker=ticker)
    fundamental_data = DataSourceWithCache(fundamental_source, alphavantage_cache)
    print('Downloading AlphaVantage fundamental data...')
    fundamental_df = fundamental_data.fetch_data()

    # insert data into the database
    db.insert_into_factor_ts_table(fundamental_df, 'AlphaVantage')    

    ## ALPHAVANTAGE POST-PROCESSING/FEATURE ENGINEERING ##
    # keep only the ratios
    ratios = ['netProfitMargin', 'ROA', 'ROE', 'debtEquityRatio', 'debtAssetRatio', 'interestCoverageRatio', 'assetTurnoverRatio', \
                'currentRatio', 'quickRatio', 'cashRatio', 'capitalExpenditureEfficiency', 'debtServiceCoverageRatio']
    fundamental_df.set_index('period', inplace=True)
    fundamental_df = fundamental_df[ratios]

    # add in temporal adjustments - push this to a feature engineering class later
    for col in fundamental_df.columns:
        fundamental_df[f'{col}_QoQ_Growth'] = fundamental_df[col].pct_change()
        fundamental_df[f'{col}_4Q_MA'] = fundamental_df[col].rolling(4).mean()
        fundamental_df[f'{col}_12Q_MA'] = fundamental_df[col].rolling(12).mean()
        epsilon = 1e-10
        denominator = fundamental_df[col].shift(4).replace(0, epsilon)
        fundamental_df[f'{col}_YOY'] = (fundamental_df[col] / denominator) - 1

    fundamental_df.dropna(inplace=True)

    # remove the factor(s) that are highly correlated (over 0.9 abs)
    fundamental_df = drop_corr_pair1(fundamental_df)

    # winsorize ratio data to 1.5% lower/98.5% upper
    for col in fundamental_df.columns:
        fundamental_df[col] = winsorize_data(fundamental_df[col], 1.5, 98.5)

    # resample to daily time series to match other datasets later
    fundamental_df_daily = fundamental_df.resample('D').ffill()
    

    # get EDGAR 10Qs and 10Ks
    print('Downloading EDGAR 10Qs and 10Ks...')
    user_agent = email + ' ' + full_name
    edgar_cache = cache_factory.get_cache_handler("edgar", timedelta(hours=24))
    edgar_source = EDGARDataSource(user_agent=user_agent, cik=cik)
    edgar_data = DataSourceWithCache(edgar_source, edgar_cache)
    text_df = edgar_data.fetch_data()

    print('Transforming text data into sentiment scores...')
    # sentiments = build_10QK_sentiment_scores(text_df)
    sentiment_cache = FileCache(cache_dir=os.path.join(os.path.dirname(__file__), "../data/cache"),
                                cache_duration=timedelta(hours=24))
    sentiments = sentiment_cache.process_with_cache(
        cache_key="sentiments",
        process_func=build_10QK_sentiment_scores,
        df=text_df
    )
    sentiments.index.name = 'period'
    sentiments.reset_index(inplace=True)
    sentiments['period'] = pd.to_datetime(sentiments['period'])

    db.insert_into_factor_ts_table(sentiments, 'EDGAR')

    sentiments.set_index('period', inplace=True)
    sentiments_daily = sentiments.resample('D').ffill().bfill()

    # # join to fundamental data since both are quarterly
    # # fundamental_df = fundamental_df.join(sentiments)
    # fundamental_df = pd.merge(fundamental_df, sentiments, on='period')

    # # need to resample to daily data
    # fundamental_df.set_index('period', inplace=True)
    # # fundamental_df.index = pd.to_datetime(fundamental_df.index)
    # fundamental_df.index.name = 'period'
    # fundamental_df_daily = fundamental_df.resample('D').ffill()
    # fundamental_df_daily.reset_index(inplace=True)

    # get EIA data: crude oil and natural gas spot prices 
    eia_codes = db.get_eia_code_info() # right now this is a list of tuples
    eia_cache = cache_factory.get_cache_handler("eia", timedelta(hours=24))
    eia_source = EIADataSource(dir=local_data_dir, api_key=eia_api_key, eia_codes=eia_codes)
    eia_data_cache = DataSourceWithCache(eia_source, eia_cache)
    eia_data = eia_data_cache.fetch_data()
    
    # insert data pulls into the time series database
    db.insert_into_factor_ts_table(eia_data, 'EIA')


    # get PJM electricity spot prices
    elec_spot_prices = pd.read_csv(os.path.join(local_data_dir, 'elec_spot_price_history/PJM-DailyAvg.RealTime.csv'))
    elec_spot_prices['HOURBEGINNING_TIME'] = pd.to_datetime(elec_spot_prices['HOURBEGINNING_TIME'])
    elec_spot_prices = (elec_spot_prices
                    .sort_values(by='HOURBEGINNING_TIME').reset_index(drop=True)
                    .rename(columns={'HOURBEGINNING_TIME': 'period', 'DOM': 'elec_spot_price'}))
    elec_spot_prices['elec_spot_price'] = elec_spot_prices['elec_spot_price'].str.replace('$', '').astype(float)

    db.insert_into_factor_ts_table(elec_spot_prices, 'PJM')

    # combine economic price data
    econ_data = pd.merge(eia_data, elec_spot_prices, on='period').dropna()

    ## ECON DATA FEATURE ENGINEERING ##
    # transform based on EDA
    econ_data['natural_gas_price'] = winsorize_data(econ_data['natural_gas_price'], 1, 99)
    econ_data['elec_spot_price'] = np.log1p(econ_data['elec_spot_price'])

    econ_data.set_index('period', inplace=True)



    ## WEATHER DATA ##
    # Charlotte, NC coordinates
    char_lat = 35.2271
    char_lon = -80.9379
    # coord1 = Coordinate(35.2271, -80.9379)

    # Orlando, FL coordinates
    orl_lat = 28.5383
    orl_lon = -81.3792
    # coord2 = Coordinate(28.5383, -81.3792)

    # Columbus, OH coordinates
    col_lat = 39.9612
    col_lon = -82.9988
    # coord3 = Coordinate(39.9612, -82.9988)

    coordinates = [(char_lat, char_lon), (orl_lat, orl_lon), (col_lat, col_lon)]

    weights = [(4/7), (2/7), (1/7)]


    meteo_cache = cache_factory.get_cache_handler("meteo", timedelta(hours=24))
    meteo_source = MeteoDataSource(coordinates=coordinates)
    weather_data = DataSourceWithCache(meteo_source, meteo_cache)
    print('Downloading Open-Meteo weather data...')
    weather_df = weather_data.fetch_data(weights=weights)

    # need to think about how to store the weather data since it's an average of three places' raw data
    weather_df = weather_df.reset_index()
    db.insert_into_factor_ts_table(weather_df, 'Meteo')

    ## WEATHER FEATURE ENGINEERING ##
    # features that are highly correlated are dropped
    weather_df.drop(['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1, inplace=True)
    
    # log transform snowfall and rainfall sums
    weather_df['snowfall_sum'] = np.log1p(weather_df['snowfall_sum'])
    weather_df['rain_sum'] = np.log1p(weather_df['rain_sum'])

    weather_df.set_index('period', inplace=True)


    ## RETURNS DATA ##
    # get price/rets data
    yfinance_cache = cache_factory.get_cache_handler("yfinance", timedelta(hours=24))
    yfinance_source = YahooFinanceDataSource(symbols=[ticker])
    yfinance_data = DataSourceWithCache(yfinance_source, yfinance_cache)

    print('Downloading YahooFinance price data...')
    price_data = yfinance_data.fetch_data()
    price_data[f'{ticker}_rets'] = np.log(price_data['Adj Close'] / price_data['Adj Close'].shift(1))
    price_data.index.name = 'period'
    price_data = price_data.reset_index()
    rets = price_data[['period', f'{ticker}_rets']]

    db.insert_into_factor_ts_table(rets, 'YahooFinance')

    rets.set_index('period', inplace=True)

    # combine all data sources into one final input dataset

    def period_to_string(df):
        # df['period'] = df['period'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df.index = df.index.strftime('%Y-%m-%d')
        return df
    
    

    datasets = [fundamental_df_daily, sentiments_daily, econ_data, weather_df, rets]
    datasets = [period_to_string(df) for df in datasets]
    # input_dataset = reduce(lambda left, right: pd.merge(left, right, on='period'), datasets)
    input_dataset = pd.concat(datasets, axis=1)
    input_dataset = (input_dataset
                    .sort_index(ascending=True)
                    .dropna())
    # input_dataset.to_csv(os.path.join(local_data_dir, "input_dataset.csv"), index=False)
    input_dataset.to_csv(os.path.join(local_data_dir, "input_dataset.csv"))


    



if __name__ == "__main__":
    sys.exit(main())