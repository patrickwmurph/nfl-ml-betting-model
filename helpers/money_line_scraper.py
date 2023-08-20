from bs4 import BeautifulSoup
import pandas as pd
import requests
from time import sleep
import random

def correct_date(date_str, year, prev_date=None):
    if len(date_str) == 3:
        formatted_date = f"{year}-0{date_str[0]}-{date_str[1:]}"
    else:
        formatted_date = f"{year}-{date_str[:2]}-{date_str[2:]}"
    
    # Update year if prev date is less than current date    
    if prev_date and formatted_date < prev_date:
        year = str(int(year) + 1)  
        return correct_date(date_str, year)
    
    return formatted_date

def get_money_lines(headers,years_dct): 
    
    dataframes = []

    for year,abb in years_dct.items():
        print(f'Started Scraping {year}')
        url = f'https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{year}-{abb}/'
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'lxml')

        table = soup.findAll('table')
        df = pd.read_html(str(table))[0]
        df.columns = df.iloc[0]
        df = df.drop(0)
        df.reset_index(drop=True, inplace=True)
        
        # Format dates
        formatted_dates = []
        prev_date = None
        for date in df['Date']:
            current_date = correct_date(date, year, prev_date)
            formatted_dates.append(current_date)
            prev_date = current_date

        df['Date'] = formatted_dates
        
        dataframes.append(df)
        
        sleep(random.uniform(0, 10)) #add random sleep between each request
        
        print(f'Finished Scraping {year}')
        
    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_csv('data/moneyline-data.csv', index=False)

