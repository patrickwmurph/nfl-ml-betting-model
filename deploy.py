from helpers.team_data_scraper import web_scraper, get_team_abb
from helpers.sports_odds_scraper import get_historical_bets, get_betting_team_code

# Run Webscraper
'''
#! Glossary :
    - Teams : dict in format of Team Name : Team Abb
    - Year : List of Years
'''

teams = get_team_abb() # gets dict of team data and abb
past_years = [2005,2006,2007,2008,2009,2010,
              2011,2012,2013,2014,2015,2016,
              2017,2018,2019,2020,2021,2022] # input the years you want the team data to be downloaded for

# Get histrical team data

web_scraper(teams, past_years)

# Get historical bet data

teams_betting = get_betting_team_code()

past_decades = [2000,2010,2020]

get_historical_bets(teams_betting, past_decades)


# Update Data

current_year = [2023]

