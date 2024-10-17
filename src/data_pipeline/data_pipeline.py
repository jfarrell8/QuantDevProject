import abc
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import yfinance as yf
import psycopg2
import os
from dotenv import load_dotenv
from utils import api_get, nan_series_handling
import sys
import ast

class DataSource(abc.ABC):
    @abc.abstractmethod
    def fetch_data(self, start_time: datetime, end_time:datetime, **kwargs) -> pd.DataFrame:
        pass

class DatabaseInterface(abc.ABC):
    @abc.abstractmethod
    def initialize_database(self) -> None:
        pass
    
    @abc.abstractmethod
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
                print(f'{query_meta} query executed and committed successfully.')
            elif query.strip().lower().startswith("select"):
                print(f'Select query ({query_meta}) executed successfully.')
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
            df['offset_value'] = nan_series_handling(df['offset_value'], data_type='Int64') # need to handle the obscure issue
            df['length'] = nan_series_handling(df['length'], data_type='Int64') # same as above

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
    def __init__(self, timezone: str = 'America%2FNew_York', lat: float=40.7143, long: float=-74.006, frequency: str = 'daily', \
                    params: List[str] = ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'sunshine_duration', \
                                            'precipitation_sum', 'rain_sum', 'snowfall_sum', 'wind_speed_10m_max']):
        self.timezone = timezone
        self.latitude = lat
        self.longitude = long
        self.frequency = frequency
        self.params = params

    # weather data
    def fetch_data(self, start_time: datetime = datetime(1986,1,1), end_time: datetime = datetime.today().date()) -> pd.DataFrame:
        start_date_str = start_time.strftime('%Y-%m-%d') 
        end_date_str = end_time.strftime('%Y-%m-%d')
        
        # build url
        url_root = 'https://archive-api.open-meteo.com/v1/archive'
        url = url_root + f'?latitude={self.latitude}&longitude={self.longitude}&start_date={start_date_str}&end_date={end_date_str}&{self.frequency}='
        if self.params:
            url += ",".join(self.params)
        url += f'&timezone={self.timezone}'

        weather_json = api_get(url)
        weather_df = pd.DataFrame(weather_json['daily']).dropna()
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        # weather_df = (weather_df
        #       .rename(columns={'time': 'period'})
        #       .set_index('period'))
        weather_df = weather_df.rename(columns={'time': 'period'})
        
        return weather_df



class YahooFinanceDataSource(DataSource):
    def __init__(self, symbols: List[str]) -> None:
        self.symbols = symbols

    def fetch_data(self, start_time: datetime = datetime(1986,1,1), end_time: datetime = datetime.today().date()) -> pd.DataFrame:
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        data = yf.download(self.symbols, start=start_date, end=end_date)

        return data
    

def main():
    load_dotenv()
    db_username = os.getenv("DB_USERNAME")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_schema = os.getenv("DB_SCHEMA")
    eia_api_key = os.getenv("EIA_API_KEY")
    local_data_dir = os.getenv("LOCAL_DATA_DIR")

    # table_create_queries = ... --> then feed to PostgresDatabase object
    db = PostgresDatabase(db_username, db_password, db_host, db_port, db_name, db_schema, local_data_dir)
    db.initialize_database()

    # get EIA data    
    eia_codes = db.get_eia_code_info() # right now this is a list of tuples
    eia = EIADataSource(dir=local_data_dir, api_key=eia_api_key, eia_codes=eia_codes)
    try:
        eia_data = eia.fetch_data()
        eia_data.to_csv(os.path.join(local_data_dir, "eia_data.csv"), index=False)
    except Exception as e:
        print('Error trying to fetch EIA data...')
        print('Using cached data instead')
        eia_data = pd.read(os.path.join(local_data_dir, "eia_data.csv"))
    

    # get weather data
    weather = MeteoDataSource()
    weather_data = weather.fetch_data()
    weather_data.to_csv(os.path.join(local_data_dir, "weather_data.csv"), index=False)

    # insert data pulls into the time series database
    db.insert_into_factor_ts_table(eia_data, 'EIA')
    db.insert_into_factor_ts_table(weather_data, 'Meteo')

    # combine EIA and weather data into one final input dataset to be used for forecasting elec spot prices
    input_dataset = eia_data.merge(weather_data, how='inner', on='period')
    # move the target variable to the last column
    cols = input_dataset.columns.difference(['elec_spot_price'], sort=False).tolist()
    input_dataset = input_dataset[cols + ['elec_spot_price']]

    input_dataset.to_csv(os.path.join(local_data_dir, "input_dataset.csv"), index=False)

    # shift the target variable
    input_dataset['elec_spot_price_shifted'] = input_dataset['elec_spot_price'].shift(-1)
    input_dataset.drop('elec_spot_price', axis=1, inplace=True)
    input_dataset = input_dataset.dropna()

    



if __name__ == "__main__":
    sys.exit(main())