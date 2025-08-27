#!/usr/bin/env python3
"""
Massey Ratings Web Scraper
Scrapes soccer game data from masseyratings.com
"""

import requests
import re
import os
import argparse
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional


class MasseyRatingsScraper:
    """Scraper for Massey Ratings soccer data"""
    
    def __init__(self, base_url: str = "https://masseyratings.com/scores.php"):
        self.base_url = base_url
        self.session = requests.Session()
        # Set headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_soccer_scores(self, params: Dict[str, str]) -> str:
        """
        Scrape the raw HTML content from the Massey Ratings website
        
        Args:
            params: Dictionary of URL parameters
            
        Returns:
            Raw HTML content as string
        """
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error scraping website: {e}")
            return ""
    
    def parse_game_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parse a single game line from the text format
        
        Expected format: "2024-11-09  Lawrence                  2  Lake Forest               1 O1"
        or: "2024-11-09 @N Central IL              2  Carroll WI                1 O3"
        or: "2025-08-19  Concordia IL              0 @Judson IL                 0 Sch"
        
        Args:
            line: Single line of game data
            
        Returns:
            Dictionary with parsed game data or None if parsing fails
        """
        # Remove extra whitespace and split
        line = line.strip()
        if not line or len(line) < 20:  # Basic validation
            return None
        
        # Enhanced pattern to match the game format with overtime and scheduled games
        # Date, Home team (with @), Home score, Away team, Away score, Overtime/Scheduled info
        pattern = r'(\d{4}-\d{2}-\d{2})\s+(@?)([^0-9]+?)\s+(\d+)\s+(@?)([^0-9]+?)\s+(\d+)(?:\s+(O\d+|Sch))?'
        
        match = re.search(pattern, line)
        if not match:
            return None
        
        date, team1_at, team1, team1_score, team2_at, team2, team2_score, game_note = match.groups()
        
        # Clean up team names
        team1 = team1.strip()
        team2 = team2.strip()
        
        # Determine which team is home/away based on @ symbol
        # The team WITH @ symbol is the HOME team
        # The team WITHOUT @ symbol is the AWAY team
        if team1_at == '@':
            # Team1 has @ symbol, so Team1 is HOME, Team2 is AWAY
            home_team = team1
            away_team = team2
            home_score = int(team1_score)
            away_score = int(team2_score)
        elif team2_at == '@':
            # Team2 has @ symbol, so Team2 is HOME, Team1 is AWAY
            home_team = team2
            away_team = team1
            home_score = int(team2_score)
            away_score = int(team1_score)
        else:
            # Neither team has @ symbol, this shouldn't happen in normal data
            # But if it does, assume Team1 is HOME, Team2 is AWAY
            home_team = team1
            away_team = team2
            home_score = int(team1_score)
            away_score = int(team2_score)
        
        # Clean up game note info (overtime or scheduled)
        game_note = game_note.strip() if game_note else ""
        
        return {
            'date': date,
            'away_team': away_team,
            'away_score': away_score,
            'home_team': home_team,
            'home_score': home_score,
            'game_note': game_note
        }
    
    def extract_games_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract game data from the scraped text content
        
        Args:
            text: Raw HTML/text content from the website
            
        Returns:
            List of dictionaries containing game data
        """
        games = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines that match the game pattern
            if re.match(r'\d{4}-\d{2}-\d{2}', line):
                game_data = self.parse_game_line(line)
                if game_data:
                    games.append(game_data)
        
        return games
    
    def scrape_and_parse_games(self, params: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Main method to scrape and parse games in one step
        
        Args:
            params: URL parameters for the request
            
        Returns:
            List of dictionaries containing game data
        """
        print(f"Scraping data from: {self.base_url}")
        print(f"Parameters: {params}")
        
        # Scrape the website
        raw_text = self.scrape_soccer_scores(params)
        
        if not raw_text:
            print("Failed to retrieve data from website")
            return []
        
        # Parse the games
        games = self.extract_games_from_text(raw_text)
        
        print(f"Successfully parsed {len(games)} games")
        return games
    
    def save_games_to_csv(self, games: List[Dict[str, str]], filename: str):
        """
        Save parsed games to a CSV file
        
        Args:
            games: List of game dictionaries
            filename: Output CSV filename
        """
        if not games:
            print("No games to save")
            return
        
        df = pd.DataFrame(games)
        df.to_csv(filename, index=False)
        print(f"Saved {len(games)} games to {filename}")





def get_season_params(season_year: int) -> Dict[str, str]:
    """
    Get URL parameters for a specific season
    
    Args:
        season_year: The season year to scrape
        
    Returns:
        Dictionary of URL parameters
    """
    season_params = {
        2024: {
            's': '594814',
            'sub': '11620', 
            'all': '1',
            'mode': '3',
            'sch': 'on',
            'format': '0'
        },
        2025: {
            's': '637470',  # Updated for 2025 season
            'sub': '11620', 
            'all': '1',
            'mode': '3',
            'sch': 'on',
            'format': '0'
        }
    }
    
    if season_year not in season_params:
        raise ValueError(f"Season {season_year} not configured. Available seasons: {list(season_params.keys())}")
    
    return season_params[season_year]


def main():
    """Main function to scrape Massey Ratings data"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scrape Massey Ratings soccer data')
    parser.add_argument('--year', '-y', type=int, default=2025, 
                       help='Season year to scrape (default: 2025)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Custom output directory (default: data/{year})')
    
    args = parser.parse_args()
    
    # Configuration
    season_year = args.year
    
    # Create scraper instance
    scraper = MasseyRatingsScraper()
    
    try:
        # Get parameters for the specified season
        params = get_season_params(season_year)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    print(f"🎯 Scraping {season_year} season data...")
    print(f"🔗 URL: {scraper.base_url}?s={params['s']}&sub={params['sub']}&all={params['all']}&mode={params['mode']}&sch={params['sch']}&format={params['format']}")
    
    # Scrape and parse the games
    games = scraper.scrape_and_parse_games(params)
    
    if games:
        # Display first few games as a preview
        print(f"\n📊 First 5 games from {season_year}:")
        for i, game in enumerate(games[:5]):
            print(f"   {i+1}. {game['date']}: {game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")
            if game['game_note']:
                if 'O' in game['game_note']:
                    print(f"      Overtime: {game['game_note']}")
                elif game['game_note'] == 'Sch':
                    print(f"      Status: Scheduled")
                else:
                    print(f"      Note: {game['game_note']}")
        
        # Create the target directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up from 'scripts' to project root
        
        if args.output_dir:
            target_dir = args.output_dir
        else:
            target_dir = os.path.join(project_root, "data", str(season_year))
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Save to CSV with season year in filename in the target directory
        output_file = os.path.join(target_dir, f"massey_games_{season_year}.csv")
        scraper.save_games_to_csv(games, output_file)
        
        print(f"\n✅ Scraping Complete!")
        print(f"📈 Total games scraped: {len(games)}")
        print(f"🏆 Season year: {season_year}")
        print(f"💾 Saved to: {output_file}")
        
        # Show some statistics
        if games:
            dates = [game['date'] for game in games]
            unique_dates = len(set(dates))
            print(f"📅 Games span {unique_dates} unique dates")
            
            # Check for games with notes (overtime, scheduled, etc.)
            noted_games = [game for game in games if game['game_note']]
            if noted_games:
                ot_games = [game for game in games if 'O' in game['game_note']]
                sch_games = [game for game in games if game['game_note'] == 'Sch']
                other_notes = [game for game in games if game['game_note'] and 'O' not in game['game_note'] and game['game_note'] != 'Sch']
                
                if ot_games:
                    print(f"⏰ Found {len(ot_games)} overtime games")
                if sch_games:
                    print(f"📅 Found {len(sch_games)} scheduled games")
                if other_notes:
                    print(f"📝 Found {len(other_notes)} games with other notes")
    else:
        print("❌ No games were successfully scraped")
        print("💡 Check the URL parameters and ensure the website is accessible")


if __name__ == "__main__":
    main()
