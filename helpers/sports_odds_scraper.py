from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs

def get_betting_team_code() :

    url = f'https://www.sportsoddshistory.com/nfl-game-odds/'
    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')
    # Find the <h3> tag with the specific text and the table immediately following it
    h3_tag = soup.find('h3', string='View Odds by Team')
    table = h3_tag.find_next_sibling('table') if h3_tag else None

    # Extract the team names and href links considering all cells in each row
    teams_and_links = []
    if table:
        for row in table.find_all('tr'):
            for cell in row.find_all('td'):
                link = cell.find('a')
                if link:
                    team_name = link.text.strip()
                    href = link.get('href')
                    teams_and_links.append((team_name, href))

    # Create a dictionary with team names and corresponding team codes
    team_name_code_dict = {}
    for team_name, href in teams_and_links:
        query_params = urlparse(href).query
        team_code = parse_qs(query_params).get('tm', [None])[0]
        if team_code:
            team_name_code_dict[team_name] = team_code


    # Printing the result
    return team_name_code_dict



def get_historical_bets(teams, decades) :
    
    request_count = 0  # Counter for requests
    dataframes = []

    for team_name, team_code in teams.items():  # Modified to loop through team codes and names
        print(f'Begin scraping for {team_name}')


        for decade in decades :
            
            if decade == 2000 :
                year = 2009
            elif decade == 2010 :
                year = 2019
            elif decade == 2020:
                year = 2023
            else :
                print(f'Decade {decade} not found')
            
            url = f'https://www.sportsoddshistory.com/nfl-game-team/?tm={team_code.upper()}&d={decade}#{year}'
            html = urlopen(url)
            soup = BeautifulSoup(html, features='lxml')
            
            request_count += 1  # Increment the request counter
            if request_count % 30 == 0:  # Check if 30 requests have been made
                print("Pausing for 60 seconds...")
                time.sleep(60)  # Pause for 60 seconds
            
            tables_betting = soup.findAll('table')[2].findAll('table')     
            
            for table in tables_betting:
                headers = [header.text.strip() for header in table.find_all('th')]
                if headers and (headers[0] == 'Week #' or headers[0] == 'Round'):
                    rows = table.find_all('tr')
                    table_data = []
                    for row in rows[1:]:  # Skip the header row
                        columns = row.find_all('td')
                        columns = [col.text.strip() for col in columns]
                        table_data.append(columns)
                    df = pd.DataFrame(table_data, columns=headers)
                    df['Team'] = team_name
                    dataframes.append(df)
            
            print(f'{year} data retrived for {team_name}')
            
        print(f'Completed scraping for {team_name}')

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Cleaning

    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.strftime('%Y-%m-%d')

    combined_df['Spread_Result'], combined_df['Spread_Value'] = combined_df['Spread'].str.split(' ', 1).str

    combined_df['O/U'], combined_df['O/U_Points'] = combined_df['Total'].str.split(' ', 1).str

    combined_df = combined_df[['Week #', 'Date', 'Team','Opponent', 'Spread_Result', 'Spread_Value', 'O/U', 'O/U_Points']]

    combined_df.rename(columns={'Opponent' : 'Opp', 'Week #' : 'Week'}, inplace=True)

    combined_df.to_csv('data/historical-betting-data.csv')
 
