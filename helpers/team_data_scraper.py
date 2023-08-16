from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import sleep
from pytz import timezone
import re

def utc_time_converter(row):
    eastern_tz = timezone('US/Eastern')

    datetime_str = f"{row['Date']} {row['Time'].split(' ')[0]}"

    datetime_obj = pd.to_datetime(datetime_str, format='%Y-%m-%d %I:%M%p')

    localized_time = eastern_tz.localize(datetime_obj)

    utc_time = localized_time.astimezone(timezone('UTC'))

    return utc_time.strftime('%I:%M%p')

def get_team_abb():
    
    teams = {}
    seen_abbreviations = set()
    pattern = r'/teams/([a-z]+)/2022\.htm'

    url = 'https://www.pro-football-reference.com/years/2022/'

    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')

    tags = soup.find_all('a', href=re.compile(pattern))

    for tag in tags:
        
        link = tag.get('href')
        team_abb = re.findall(pattern, link)
        
        if team_abb and team_abb[0] not in seen_abbreviations:
            team_name = tag.text.strip()
            teams[team_name] = team_abb[0]
            seen_abbreviations.add(team_abb[0]) 

    return teams


def reformat_date_without_year(df, year) :
    # Create a mapping of month names to month numbers
    month_mapping = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }
    # Define the starting month and year

    current_year = year
    current_month = month_mapping.get(df['Date'].str.split(' ')[0][0])

    # Initialize a variable to hold the previous day
    previous_day = 0



    # Iterate through the Date column and reformat the dates
    for index, row in df.iterrows():
        # Extract the day from the Date column
        day = int(row['Date'].split(' ')[1])

        # Check if the current day is earlier than the previous day (i.e., we've crossed into the next month)
        if day < previous_day:
            current_month += 1
            # If we've crossed into the next year
            if current_month > 12:
                current_month = 1
                current_year += 1

        # Convert to the new date format with the current year and month
        new_date = f"{current_year}-{current_month:02d}-{day:02d}"
        
        # Update the Date column with the new date format
        df.at[index, 'Date'] = new_date
        
        # Update the previous day
        previous_day = day
        
    return df


def web_scraper(teams, years) :

    team_years = {team : years for team in teams.values()}
    request_count = 0
    max_count_per_min = 15
    dataframes = []
    
    for team in team_years :
        
            for k, v in teams.items() :
                if v == team :
                    team_full_name = k
                    
            print(f'Begin scraping for {team_full_name}')
            for year in team_years[team] :
                
                # Request delay to not exceed cite limit

                if request_count >= max_count_per_min :
                    print(f"Reached {max_count_per_min} requests, sleeping for 60 seconds...")
                    sleep(60)
                    request_count = 0
                
                request_count += 1
                
                # Parse URL
                
                url = f'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
                html = urlopen(url)
                soup = BeautifulSoup(html, features = 'lxml')

                headers = soup.findAll('tr')[7]
                rows = soup.findAll('tbody')[1].findAll('tr')


                # Extract header
                
                header_ls = [th.get_text().strip() for th in headers.find_all('th')]

    
                # Extract Rows
                
                data = []
                for row in rows:
                    values = [th.get_text() for th in row.find_all('th')]
                    values += [td.get_text() for td in row.find_all('td')]
                    data.append(values)

                
                # Create DataFrame
                
                df = pd.DataFrame(data, columns=header_ls)
                
                # Remove Unnecessary Data
                
                df.columns.values[3] = 'Time'
                df.columns.values[4] = 'Canceled'
                df.columns.values[5] = 'W/L'
                df.columns.values[8] = 'Location'
                df.drop(index = df[~(df['Canceled'] == 'boxscore')].index, inplace = True)
                df.drop(columns = ['Canceled', 'OT'], inplace = True)
                df = df[~(df['Day']== '')]
                
                # Reformat Dates
                df = df[df['Date'].str.contains(r'\d')]

                df = reformat_date_without_year(df, year)
                                
                
                # Change Numerics
                '''
                #! Glossary 
                    - Rec : Win = 1, Loss = -1, Tie = 0
                    - Location : Home = 1, Away = 0
                    
                '''
                                
                df['Location'] = np.where(df['Location'] == '@', 0,1) # Home = 1, Away = 0
                
                if len(df['Rec'].str.split('-',expand=True).values[0]) > 2 :
                    df[['tmp1','tmp2','tmp3']] = df['Rec'].str.split('-',expand=True)
                    df.fillna(0, inplace=True)
                    df['Rec'] = df['tmp1'].astype(int) - df['tmp2'].astype(int) + (0*df['tmp3'].astype(int))
                    df.drop(columns = ['tmp1','tmp2','tmp3'], inplace = True)
                else :
                    df[['tmp1','tmp2']] = df['Rec'].str.split('-',expand=True)
                    df['Rec'] = df['tmp1'].astype(int) - df['tmp2'].astype(int)
                    df.drop(columns = ['tmp1','tmp2'], inplace = True)

                df['Team'] = team_full_name
                df['Season'] = year
                
                dataframes.append(df)
                
                print(f'{year} data retrived for {team_full_name}')
            
            print(f'Completed scraping for {team_full_name}')
            
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    last_date = combined_df['Date'].max()
            
    combined_df.to_csv(f'data/{last_date}-data.csv')

    print(f'Completed scraping all years {years[0]}-{years[-1]}')

