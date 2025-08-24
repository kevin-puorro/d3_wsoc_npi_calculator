#!/usr/bin/env python3
"""
NPI Calculator Script
Calculates National Power Index (NPI) ratings for teams based on filtered Massey games data.

NPI Formula: Base NPI = (win_component × 0.20) + (opponent_npi × 0.80)
            Quality Bonus = max(0, (opponent_npi - 54.00) × 0.5) if won
            Total NPI = base_npi + quality_bonus
            
            Where win_component = 100 (win), 50 (tie), 0 (loss)
"""

import os
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict


def load_filtered_games(data_dir: str, year: int, start_date: str = "2024-08-30", end_date: str = "2024-11-10") -> pd.DataFrame:
    """
    Load the NPI games data for NPI calculation, filtered by date range
    
    Args:
        data_dir: Directory containing the data
        year: Year to load data for
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        
    Returns:
        DataFrame with NPI games data
    """
    try:
        # Look for the NPI games file
        npi_file = os.path.join(data_dir, f"massey_games_{year}_npi.csv")
        
        if not os.path.exists(npi_file):
            raise FileNotFoundError(f"NPI games file not found: {npi_file}")
        
        print(f"Loading NPI games from: {npi_file}")
        games_df = pd.read_csv(npi_file)
        print(f"Loaded {len(games_df)} NPI games")
        
        # Handle empty dataset
        if games_df.empty:
            print(f"⚠️  No NPI games found for {year} - this may be normal for seasons with only scheduled games")
            return games_df
        
        # Check if game_note column exists, if not add it as empty
        if 'game_note' not in games_df.columns:
            # Check if overtime column exists and rename it
            if 'overtime' in games_df.columns:
                print("🔄 Renaming 'overtime' column to 'game_note'")
                games_df = games_df.rename(columns={'overtime': 'game_note'})
            else:
                print("⚠️  No game_note column found, adding empty column")
                games_df['game_note'] = ""
        
        # Convert date column to datetime for filtering
        games_df['date'] = pd.to_datetime(games_df['date'])
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        
        # Filter games between start and end dates (inclusive)
        filtered_games = games_df[(games_df['date'] >= start_datetime) & (games_df['date'] <= end_date)].copy()
        
        print(f"Filtered to {len(filtered_games)} games between {start_date} and {end_date}")
        
        return filtered_games
        
    except Exception as e:
        print(f"Error loading NPI games: {e}")
        return pd.DataFrame()


def calculate_game_npi(won: bool, tied: bool, opponent_npi: float, iteration_number: int = 1) -> float:
    """
    Calculate NPI for a single game using the formula:
    Base NPI = (win_component × 0.20) + (opponent_npi × 0.80)
    Quality Bonus = max(0, (opponent_npi - 54.00) × 0.5) if won, or × 0.25 if tied (but not on first iteration)
    Total NPI = base_npi + quality_bonus
    
    Args:
        won: Whether the team won the game
        tied: Whether the game ended in a tie
        opponent_npi: NPI rating of the opponent
        iteration_number: Current iteration number (1-based)
        
    Returns:
        NPI rating for this game
    """
    # Determine win component
    if won:
        win_component = 100
    elif tied:
        win_component = 50
    else:
        win_component = 0
    
    # Calculate base NPI
    base_npi = (win_component * 0.20) + (opponent_npi * 0.80)
    
    # Calculate quality bonus (only if won or tied AND after first 3 iterations)
    quality_bonus = 0
    if won and iteration_number > 3:
        quality_bonus = max(0, (opponent_npi - 54.00) * 0.500)
    elif tied and iteration_number > 3:
        # Calculate full quality bonus, then divide by 2 for ties
        full_bonus = max(0, (opponent_npi - 54.00) * 0.500)
        quality_bonus = full_bonus / 2
    
    # Calculate total NPI
    total_npi = base_npi + quality_bonus
    return total_npi


def process_games_iteration(games_df: pd.DataFrame, valid_teams: set, previous_iteration_npis: Dict[str, float] = None, iteration_number: int = 1) -> Dict[str, Dict]:
    """
    Process games for one iteration of NPI calculation using the basketball method.
    This includes qualifying wins/losses and qualifying threshold logic.
    
    Args:
        games_df: DataFrame with games data
        valid_teams: Set of valid team names
        previous_iteration_npis: NPI ratings from previous iteration
        iteration_number: Current iteration number
        
    Returns:
        Dictionary mapping team names to their complete data
    """
    # Calculate OWP for this iteration
    owp = calculate_owp(games_df)
    
    # Set up opponent NPIs early
    if iteration_number <= 1:
        # Use OWP for first 3 iterations to establish stable baseline
        opponent_npis = {team: owp.get(team, 50.0) for team in valid_teams}
    else:
        opponent_npis = {}
        for team in valid_teams:
            opponent_npis[team] = previous_iteration_npis.get(team, owp.get(team, 50.0))
    
    # Initialize teams dict with all required keys
    teams = {
        team: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "npi": opponent_npis[team],
            "game_npis": [],
            "all_game_npis": [],
            "team_name": team,
            "qualifying_wins": 0,
            "qualifying_losses": 0,
            "partial_wins": [],
            "has_games": False,
        }
        for team in valid_teams
    }
    
    # DEBUG: Track Wash U games specifically
    wash_u_team_names = [team for team in valid_teams if 'North Central College' in team]
    wash_u_debug_games = []
    
    # Combine first two passes - record stats and calculate game NPIs
    for _, game in games_df.iterrows():
        team1_id = game['home_team']
        team2_id = game['away_team']
        team1_score = int(game['home_score'])
        team2_score = int(game['away_score'])
        
        # DEBUG: Track Wash U games
        if any('North Central College' in team for team in [team1_id, team2_id]):
            wash_u_debug_games.append({
                'home': team1_id, 'away': team2_id,
                'home_score': team1_score, 'away_score': team2_score,
                'date': game.get('date', 'Unknown')
            })
        
        # Skip invalid games (only check for valid teams)
        if (team1_id not in valid_teams or 
            team2_id not in valid_teams):
            continue
        
        # Update basic stats
        teams[team1_id]["has_games"] = True
        teams[team2_id]["has_games"] = True
        teams[team1_id]["games"] += 1
        teams[team2_id]["games"] += 1
        
        # Check if game went to overtime - if so, treat as tie regardless of score
        overtime_occurred = pd.notna(game['game_note']) and str(game['game_note']).strip() != "" and any(x in str(game['game_note']).lower() for x in ['o1', 'o2', 'o3', 'o4', 'o5'])
        
        # Update wins/losses and calculate NPIs in same pass
        if overtime_occurred:
            # Overtime game - automatically treat as tie
            teams[team1_id]["ties"] += 1
            teams[team2_id]["ties"] += 1
            team1_won, team2_won = False, False
        elif team1_score > team2_score:
            teams[team1_id]["wins"] += 1
            teams[team2_id]["losses"] += 1
            team1_won, team2_won = True, False
        elif team2_score > team1_score:
            teams[team2_id]["wins"] += 1
            teams[team1_id]["losses"] += 1
            team1_won, team2_won = False, True
        else:
            teams[team1_id]["ties"] += 1
            teams[team2_id]["ties"] += 1
            team1_won, team2_won = False, False
        
        # Calculate and store game NPIs using your formula
        team1_tied = overtime_occurred or (team1_score == team2_score)
        team2_tied = overtime_occurred or (team1_score == team2_score)
        team1_game_npi = calculate_game_npi(team1_won, team1_tied, opponent_npis[team2_id], iteration_number)
        team2_game_npi = calculate_game_npi(team2_won, team2_tied, opponent_npis[team1_id], iteration_number)
        
        # DEBUG: Track Wash U game storage
        if 'North Central College' in team1_id:
            print(f"         🔍 Storing for {team1_id}: NPI {team1_game_npi:.2f}, won={team1_won}, tied={team1_tied}")
        if 'North Central College' in team2_id:
            print(f"         🔍 Storing for {team2_id}: NPI {team2_game_npi:.2f}, won={team2_won}, tied={team2_tied}")
        
        teams[team1_id]["all_game_npis"].append((team1_game_npi, team1_won, team1_tied))
        teams[team2_id]["all_game_npis"].append((team2_game_npi, team2_won, team2_tied))
    
    # DEBUG: Show Wash U game processing details
    if wash_u_debug_games and iteration_number == 1:
        print(f"\n🔍 DEBUG: Wash U games processed in iteration {iteration_number}:")
        for game in wash_u_debug_games:
            home_won = game['home_score'] > game['away_score']
            away_won = game['away_score'] > game['home_score']
            tied = game['home_score'] == game['away_score']
            
            if tied:
                result = "TIE"
            elif home_won:
                result = f"{game['home']} WINS"
            else:
                result = f"{game['away']} WINS"
                
            print(f"  {game['date']}: {game['home']} {game['home_score']} - {game['away_score']} {game['away']} → {result}")
    
    # Optimized third pass: filter and calculate final NPIs
    for team_id, team_data in teams.items():
        if not team_data["has_games"]:
            continue
        
        initial_npi = opponent_npis[team_id]
        all_games = team_data["all_game_npis"]
        used_npis = []
        
        # Split and sort wins/losses/ties once
        wins = []
        losses = []
        ties = []
        for npi, won, tied in all_games:
            if tied:
                ties.append(npi)
            elif won:
                wins.append(npi)
            else:
                losses.append(npi)
        
        # DEBUG: Show tie detection for Wash U
        if 'North Central College' in team_id:
            print(f"      🔍 DEBUG: {team_id} tie detection:")
            print(f"         All games: {len(all_games)}")
            print(f"         Ties found: {len(ties)}")
            print(f"         Wins found: {len(wins)}")
            print(f"         Losses found: {len(losses)}")
            print(f"         Raw game data:")
            for i, (npi, won, tied) in enumerate(all_games):
                result = "TIE" if tied else "WIN" if won else "LOSS"
                print(f"           Game {i+1}: NPI {npi:.2f} → {result}")
        
        # Sort once
        wins.sort(reverse=True)
        losses.sort()
        
        # Process wins and ties to meet minimum 8.0 win credits threshold
        qualifying_threshold = 8.0  # Minimum win credits (wins + 0.5 × ties)
        qualifying_games = 0
        
        # First, add ALL ties (they count as 0.5 win credits each)
        for tie_npi in ties:
            used_npis.append(tie_npi)
            qualifying_games += 0.5
        
        # Then add ALL wins above initial NPI
        for win_npi in wins:
            if win_npi >= initial_npi:
                used_npis.append(win_npi)
                qualifying_games += 1
        
        # Finally add wins below initial NPI until we meet the 8.0 threshold
        # Allow partial wins (0.5) to meet the threshold exactly
        for win_npi in wins:
            if win_npi < initial_npi and qualifying_games < qualifying_threshold:
                remaining_needed = qualifying_threshold - qualifying_games
                if remaining_needed >= 1.0:
                    # Add full win
                    used_npis.append(win_npi)
                    qualifying_games += 1
                elif remaining_needed >= 0.5:
                    # Add half win (0.5)
                    used_npis.append(win_npi)
                    qualifying_games += 0.5
                    # Mark this as a partial win for tracking
                    if 'partial_wins' not in team_data:
                        team_data['partial_wins'] = []
                    team_data['partial_wins'].append(win_npi)
        
        # Process losses more efficiently
        if losses:
            # Add all instances of the worst loss
            worst_loss = losses[0]
            used_npis.extend(npi for npi in losses if npi == worst_loss)
            
            # Add all losses below initial NPI, grouping identical NPIs
            seen_npis = {worst_loss}
            for loss_npi in losses:
                if loss_npi < initial_npi and loss_npi not in seen_npis:
                    seen_npis.add(loss_npi)
                    used_npis.extend(npi for npi in losses if npi == loss_npi)
        
        # Calculate final NPI and stats
        if used_npis:
            team_data["game_npis"] = used_npis
            team_data["npi"] = sum(used_npis) / len(used_npis)
            # Count qualifying games
            team_data["qualifying_wins"] = sum(1 for npi in used_npis if npi in wins and npi not in team_data.get('partial_wins', []))
            team_data["qualifying_losses"] = sum(1 for npi in used_npis if npi in losses)
            team_data["qualifying_ties"] = sum(1 for npi in used_npis if npi in ties)
            # Add partial wins count
            team_data["partial_wins_count"] = len(team_data.get('partial_wins', []))
        else:
            team_data["game_npis"] = []
            team_data["npi"] = initial_npi
            team_data["qualifying_wins"] = 0
            team_data["qualifying_losses"] = 0
            team_data["qualifying_ties"] = 0
            team_data["partial_wins_count"] = 0
    
    return teams





def calculate_owp(games_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Opponents' Win Percentage (OWP) for each team.
    OWP is the average win percentage of a team's opponents, excluding games against the team itself.
    
    Args:
        games_df: DataFrame with games data
        
    Returns:
        Dictionary mapping team names to their OWP percentage
    """
    if games_df.empty:
        return {}
    
    # Get all unique teams
    all_teams = set()
    for _, game in games_df.iterrows():
        all_teams.add(game['home_team'])
        all_teams.add(game['away_team'])
    
    valid_teams = list(all_teams)
    
    # Pre-initialize records with known structure
    records = {team: {"wins": 0, "losses": 0, "ties": 0, "games": 0} for team in valid_teams}
    
    # Build opponent stats per team
    opponent_stats = {
        team: {"total_wins": 0, "total_losses": 0, "total_ties": 0, "games": []}
        for team in valid_teams
    }
    
    # Single pass through games to build both records and collect opponent games
    for _, game in games_df.iterrows():
        team1_id = game['home_team']
        team2_id = game['away_team']
        team1_score = int(game['home_score'])
        team2_score = int(game['away_score'])
        
        # Skip invalid games (both scores 0 might indicate no game played)
        if team1_score == 0 and team2_score == 0:
            continue
        
        # Update records
        records[team1_id]["games"] += 1
        records[team2_id]["games"] += 1
        
        if team1_score > team2_score:
            # Home team wins
            records[team1_id]["wins"] += 1
            records[team2_id]["losses"] += 1
        elif team2_score > team1_score:
            # Away team wins
            records[team2_id]["wins"] += 1
            records[team1_id]["losses"] += 1
        else:
            # Tie
            records[team1_id]["ties"] += 1
            records[team2_id]["ties"] += 1
        
        # Store game references for each team's opponents
        opponent_stats[team1_id]["games"].append((team2_id, game))
        opponent_stats[team2_id]["games"].append((team1_id, game))
    
    # Calculate OWP for each team
    owp = {}
    for team_id, opp_data in opponent_stats.items():
        opponents_total_wins = 0
        opponents_total_losses = 0
        opponents_total_ties = 0
        
        # Process each opponent's record once
        for opp_id, game in opp_data["games"]:
            opp_record = records[opp_id]
            opp_wins = opp_record["wins"]
            opp_losses = opp_record["losses"]
            opp_ties = opp_record["ties"]
            
            # Adjust for head-to-head results (remove the game against the current team)
            if game["home_team"] == team_id:
                # Current team was home team
                if game["home_score"] > game["away_score"]:
                    opp_losses -= 1  # Opponent lost to current team
                elif game["away_score"] > game["home_score"]:
                    opp_wins -= 1    # Opponent beat current team
                else:
                    opp_ties -= 1   # Opponent tied current team
            else:
                # Current team was away team
                if game["away_score"] > game["home_score"]:
                    opp_losses -= 1  # Opponent lost to current team
                elif game["home_score"] > game["away_score"]:
                    opp_wins -= 1    # Opponent beat current team
                else:
                    opp_ties -= 1   # Opponent tied current team
            
            opponents_total_wins += opp_wins
            opponents_total_losses += opp_losses
            opponents_total_ties += opp_ties
        
        # Calculate OWP (ties count as 0.5 wins)
        total_games = opponents_total_wins + opponents_total_losses + opponents_total_ties
        if total_games > 0:
            # Convert to percentage (ties count as 0.5 wins)
            owp_value = ((opponents_total_wins + 0.5 * opponents_total_ties) / total_games) * 100
            owp[team_id] = owp_value
        else:
            owp[team_id] = 50.0  # Default to 50% if no games
    
    return owp


def calculate_opponents_owp(games_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Opponents' Opponents' Win Percentage (OOWP) for each team.
    This is the average OWP of a team's opponents.
    
    Args:
        games_df: DataFrame with games data
        
    Returns:
        Dictionary mapping team names to their OOWP percentage
    """
    if games_df.empty:
        return {}
    
    # First calculate OWP for all teams
    team_owp = calculate_owp(games_df)
    
    # Get all unique teams
    all_teams = set()
    for _, game in games_df.iterrows():
        all_teams.add(game['home_team'])
        all_teams.add(game['away_team'])
    
    valid_teams = list(all_teams)
    
    # Build opponent lists for each team
    opponent_lists = {team: set() for team in valid_teams}
    
    for _, game in games_df.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        opponent_lists[home_team].add(away_team)
        opponent_lists[away_team].add(home_team)
    
    # Calculate OOWP for each team
    oowp = {}
    for team in valid_teams:
        opponents = opponent_lists[team]
        if opponents:
            # Get OWP values for all opponents
            opponent_owps = [team_owp.get(opp, 50.0) for opp in opponents]
            oowp[team] = np.mean(opponent_owps)
        else:
            oowp[team] = 50.0  # Default to 50% if no opponents
    
    return oowp


def create_placeholder_summary(teams_list):
    """Create placeholder NPI summary for teams with no completed games"""
    summary_data = []
    
    for team in teams_list:
        summary_data.append({
            'team': team,
            'npi_rating': 50.00,  # Default NPI rating
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'qualifying_wins': 0,
            'qualifying_losses': 0,
            'qualifying_ties': 0,
            'owp': 50.0,  # Default OWP
            'oowp': 50.0,  # Default OOWP
            'partial_wins': 0
        })
    
    # Create DataFrame and sort alphabetically by team name
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('team').reset_index(drop=True)
    
    return summary_df


def create_npi_summary(games_df: pd.DataFrame, team_npi_ratings: Dict[str, float]) -> pd.DataFrame:
    """
    Create a comprehensive summary DataFrame with NPI ratings, team records, OWP, OOWP, and qualifying stats
    
    Args:
        games_df: DataFrame with games data
        team_npi_ratings: Dictionary of team NPI ratings
        
    Returns:
        DataFrame with comprehensive NPI summary
    """
    # Get all unique teams
    all_teams = set()
    for _, game in games_df.iterrows():
        all_teams.add(game['home_team'])
        all_teams.add(game['away_team'])
    
    valid_teams = all_teams
    
    # Get the final iteration data to extract qualifying wins/losses
    final_teams_data = process_games_iteration(games_df, valid_teams, team_npi_ratings, 999)
    
    # Calculate OWP and OOWP
    team_owp = calculate_owp(games_df)
    team_oowp = calculate_opponents_owp(games_df)
    
    # Create summary data
    summary_data = []
    for team in team_npi_ratings.keys():
        team_data = final_teams_data[team]
        summary_data.append({
            'team': team,
            'wins': team_data['wins'],
            'losses': team_data['losses'],
            'ties': team_data['ties'],
            'games': team_data['games'],
            'win_pct': (team_data['wins'] + 0.5 * team_data['ties']) / team_data['games'] if team_data['games'] > 0 else 0.0,
            'owp': team_owp.get(team, 50.0),
            'oowp': team_oowp.get(team, 50.0),
            'qualifying_wins': team_data['qualifying_wins'],
            'qualifying_losses': team_data['qualifying_losses'],
            'qualifying_ties': team_data['qualifying_ties'],
            'partial_wins': team_data.get('partial_wins_count', 0),
            'npi_rating': team_npi_ratings[team]
        })
    
    # Create DataFrame and sort by NPI rating (highest first)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('npi_rating', ascending=False).reset_index(drop=True)
    
    return summary_df


def save_npi_results(summary_df: pd.DataFrame, output_dir: str, year: int) -> str:
    """
    Save NPI results to CSV
    
    Args:
        summary_df: DataFrame with NPI summary
        output_dir: Directory to save the file
        year: Year for the filename
        
    Returns:
        Path to saved file
    """
    if summary_df.empty:
        print("No NPI results to save")
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"npi_ratings_{year}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"Saved NPI ratings for {len(summary_df)} teams to: {output_path}")
    
    return output_path


def check_convergence(previous_npis, current_teams, threshold=0.001):
    """
    Check if NPI ratings have converged by measuring the maximum change
    
    Args:
        previous_npis: NPI ratings from previous iteration
        current_teams: Team data from current iteration
        threshold: Maximum allowed change to consider converged
        
    Returns:
        tuple: (converged, max_change, avg_change)
    """
    if not previous_npis or not current_teams:
        return False, 0, 0
    
    max_change = 0
    total_teams = 0
    total_change = 0
    
    for team_id, team_data in current_teams.items():
        if team_id in previous_npis and team_data.get("has_games", False):
            # Extract NPI from team data
            current_npi = team_data.get("npi", 0)
            previous_npi = previous_npis.get(team_id, 0)
            
            change = abs(previous_npi - current_npi)
            max_change = max(max_change, change)
            total_change += change
            total_teams += 1
    
    # Only consider converged if we have teams to compare
    if total_teams == 0:
        return False, 0, 0
    
    avg_change = total_change / total_teams
    converged = max_change < threshold
    
    return converged, max_change, avg_change


def main(use_season_results=False):
    """Main entry point for the application."""
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate NPI ratings for teams')
    parser.add_argument('--year', '-y', type=int, default=2025, 
                       help='Season year to process (default: 2025)')
    parser.add_argument('--season-only', '-s', action='store_true',
                       help='Use season results mode (fewer iterations)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for filtering games (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for filtering games (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    # Configuration
    year = args.year
    use_season_results = args.season_only or use_season_results
    NUM_ITERATIONS = 10 if use_season_results else 60
    
    # Set default date ranges based on year
    if args.start_date is None:
        if year == 2025:
            start_date = "2025-08-29"  # 2025 season start
        else:
            start_date = "2024-08-30"  # 2024 season start
    
    if args.end_date is None:
        if year == 2025:
            end_date = "2025-11-09"  # 2025 season end
        else:
            end_date = "2024-11-10"  # 2024 season end
    
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define paths
    data_dir = os.path.join(project_root, "data", str(year))
    output_dir = data_dir  # Save results in same directory
    
    print(f"🏆 NPI Calculator for {year}")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"🔄 Iterations: {NUM_ITERATIONS}")
    print(f"🎯 Mode: {'Season Results' if use_season_results else 'Full Calculation'}")
    print("-" * 50)
    
    try:
        # Step 1: Load NPI games data
        print("1️⃣ Loading NPI games data...")
        games_df = load_filtered_games(data_dir, year, start_date, end_date)
        
        if games_df.empty:
            print(f"⚠️  No NPI games found for {year} season.")
            print("💡 This is normal for seasons with only scheduled games (like 2025).")
            print("📋 Cannot calculate NPI ratings without completed games.")
            print("🏆 Creating placeholder NPI output with default ratings...")
            
            # Try to load the filtered games to get team list
            try:
                filtered_file = os.path.join(data_dir, f"massey_games_{year}_filtered.csv")
                if os.path.exists(filtered_file):
                    filtered_df = pd.read_csv(filtered_file)
                    all_teams = set()
                    for _, game in filtered_df.iterrows():
                        all_teams.add(game['home_team'])
                        all_teams.add(game['away_team'])
                    
                    teams_list = sorted(list(all_teams))
                    print(f"\n📊 Found {len(teams_list)} teams in the filtered data")
                    
                    # Create placeholder data for NPI calculation
                    valid_teams = teams_list
                    start_total_time = time.time()
                    
                    # Set all teams to default NPI of 50.0
                    opponent_npis = {team_id: 50.0 for team_id in valid_teams}
                    final_teams = None
                    total_games = 0
                    final_iteration = 1
                    converged = True
                    
                    print(f"🏆 Created placeholder data for {len(valid_teams)} teams")
                    
                    # Skip to creating summary with placeholder data
                    print("\n3️⃣ Creating placeholder NPI summary...")
                    team_owp = {team: 50.0 for team in valid_teams}
                    team_oowp = {team: 50.0 for team in valid_teams}
                    print(f"✅ Created placeholder OWP and OOWP for {len(team_owp)} teams")
                    
                    # Create placeholder summary
                    print("\n4️⃣ Creating NPI summary...")
                    summary_df = create_placeholder_summary(valid_teams)
                    print(f"✅ Created placeholder summary for {len(summary_df)} teams")
                    
                    # Save results
                    print("\n5️⃣ Saving placeholder NPI results...")
                    output_file = save_npi_results(summary_df, output_dir, year)
                    
                    total_time = time.time() - start_total_time
                    
                    # Summary
                    print("\n" + "=" * 50)
                    print("📋 PLACEHOLDER NPI CALCULATION SUMMARY")
                    print("=" * 50)
                    print(f"📊 Total games processed: 0 (placeholder)")
                    print(f"🏆 Total teams: {len(summary_df)}")
                    print(f"💾 Output file: {output_file}")
                    print(f"⏱️  Total processing time: {total_time:.3f} seconds")
                    print(f"🔄 Total iterations completed: 1 (placeholder)")
                    print(f"🎯 Convergence status: ✅ Placeholder (no games to process)")
                    print(f"🎮 Total games processed: 0 (placeholder)")
                    
                    print(f"\n🏅 Top 25 Teams by Default NPI Rating:")
                    for i, (_, team) in enumerate(summary_df.head(25).iterrows()):
                        print(f"   {i+1:2d}. {team['team']:<35} NPI: {team['npi_rating']:.4f}")
                        print(f"       Record: {team['wins']:2d}W-{team['losses']:2d}L-{team['ties']:2d}T | Qualifying: {team['qualifying_wins']:2d}W-{team['qualifying_losses']:2d}L-{team['qualifying_ties']:2d}T")
                        print(f"       ✅ PLACEHOLDER DATA (no completed games)")
                        print(f"       OWP: {team['owp']:.1f}% | OOWP: {team['oowp']:.1f}%")
                    
                    print(f"\n📊 PLACEHOLDER DATA ANALYSIS:")
                    print(f"   ⚠️  All teams have default NPI rating of 50.00")
                    print(f"   ⚠️  All teams have default OWP and OOWP of 50.0%")
                    print(f"   ⚠️  No actual games were processed")
                    print(f"   💡 This is normal for seasons with only scheduled games")
                    
                    print(f"\n🎯 Placeholder NPI calculation complete!")
                    print(f"📁 Results saved to: {output_file}")
                    print(f"💡 Run again when games are completed for actual NPI calculations")
                    
                    return final_teams
                    
                else:
                    print("❌ No filtered games file found to extract team list.")
                    print("❌ Exiting without NPI calculations.")
                    return
            except Exception as e:
                print(f"❌ Error loading team list: {e}")
                print("❌ Exiting without NPI calculations.")
                return
        
        print(f"📊 Loaded {len(games_df)} NPI games")
        
        # Check if we have any completed games (not scheduled)
        completed_games = games_df[games_df['game_note'] != 'Sch']
        scheduled_games = games_df[games_df['game_note'] == 'Sch']
        
        print(f"📅 Found {len(completed_games)} completed games and {len(scheduled_games)} scheduled games")
        
        if len(completed_games) == 0:
            print("⚠️  Warning: All games are scheduled (Sch). NPI calculations will be based on scheduled games only.")
            print("💡 This is normal for future seasons that haven't started yet.")
        
        # Get all unique teams
        all_teams = set()
        for _, game in games_df.iterrows():
            all_teams.add(game['home_team'])
            all_teams.add(game['away_team'])
        
        valid_teams = all_teams
        print(f"🏆 Found {len(valid_teams)} teams")
        
        start_total_time = time.time()
        
        # Initialize once - use OWP for first iteration
        team_owp = calculate_owp(games_df)
        opponent_npis = {team_id: team_owp.get(team_id, 50.0) for team_id in valid_teams}
        final_teams = None
        total_games = 0
        
        # Step 2: Calculate team NPI ratings through iterations
        print("\n2️⃣ Calculating NPI ratings...")
        print("   🔍 First 3 iterations: Using OWP values, NO quality bonus")
        print("   🔍 Iterations 4+: Using NPI values, WITH quality bonus")
        print("   🎯 Convergence threshold: 0.01 NPI points")
        
        converged = False
        final_iteration = NUM_ITERATIONS
        
        for i in range(NUM_ITERATIONS):
            iteration_number = i + 1
            print(f"   Iteration {iteration_number}/{NUM_ITERATIONS}...")
            
            teams = process_games_iteration(
                games_df, valid_teams, opponent_npis, iteration_number
            )
            
            # Check convergence (skip first iteration)
            if i > 0:
                converged, max_change, avg_change = check_convergence(opponent_npis, teams)
                if converged:
                    print(f"   ✅ CONVERGED after {iteration_number} iterations!")
                    print(f"      Max change: {max_change:.4f} | Avg change: {avg_change:.4f}")
                    final_iteration = iteration_number
                    final_teams = teams
                    break
            
            if iteration_number == NUM_ITERATIONS:
                final_teams = teams
                print(f"   ⚠️  Reached maximum iterations ({NUM_ITERATIONS}) without convergence")
            
            # Calculate total games in final iteration
            for team_id, team_data in teams.items():
                total_games += len(team_data["all_game_npis"])
            
            # Update in place for next iteration
            opponent_npis.clear()
            opponent_npis.update(
                {
                    team_id: stats["npi"]
                    for team_id, stats in teams.items()
                    if stats["has_games"]
                }
            )
        
        # Step 3: Calculate OWP and OOWP
        print("\n3️⃣ Calculating OWP and OOWP...")
        team_owp = calculate_owp(games_df)
        team_oowp = calculate_opponents_owp(games_df)
        print(f"✅ Calculated OWP and OOWP for {len(team_owp)} teams")
        
        # Step 4: Create summary
        print("\n4️⃣ Creating NPI summary...")
        summary_df = create_npi_summary(games_df, opponent_npis)
        print(f"✅ Created summary for {len(summary_df)} teams")
        
        # Step 5: Save results
        print("\n5️⃣ Saving NPI results...")
        output_file = save_npi_results(summary_df, output_dir, year)
        
        total_time = time.time() - start_total_time
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 NPI CALCULATION SUMMARY")
        print("=" * 50)
        print(f"📊 Total games processed: {len(games_df)}")
        print(f"🏆 Total teams: {len(summary_df)}")
        print(f"💾 Output file: {output_file}")
        print(f"⏱️  Total processing time: {total_time:.3f} seconds")
        print(f"⏱️  Average time per iteration: {total_time/final_iteration:.3f} seconds")
        print(f"🔄 Total iterations completed: {final_iteration}")
        print(f"🎯 Convergence status: {'✅ Converged' if converged else '⚠️  Max iterations reached'}")
        print(f"🎮 Total games processed in final iteration: {total_games}")
        
        # Show top 25 teams
        print(f"\n🏅 Top 25 Teams by NPI Rating:")
        for i, (_, team) in enumerate(summary_df.head(100).iterrows()):
            print(f"   {i+1:2d}. {team['team']:<35} NPI: {team['npi_rating']:.4f}")
            print(f"       Record: {team['wins']:2d}W-{team['losses']:2d}L-{team['ties']:2d}T | Qualifying: {team['qualifying_wins']:2d}W-{team['qualifying_losses']:2d}L-{team['qualifying_ties']:2d}T")
            
            # Show partial wins information
            partial_wins = team.get('partial_wins', 0)
            if partial_wins > 0:
                print(f"       🟡 PARTIAL WINS: {partial_wins} (using 0.5 win credits)")
            else:
                print(f"       ✅ FULL WINS ONLY (no partial wins needed)")
                
            print(f"       OWP: {team['owp']:.1f}% | OOWP: {team['oowp']:.1f}%")
        
        # Summary of partial wins usage
        teams_with_partial_wins = summary_df[summary_df['partial_wins'] > 0]
        teams_full_wins_only = summary_df[summary_df['partial_wins'] == 0]
        
        print(f"\n📊 PARTIAL WINS ANALYSIS:")
        print(f"   🟡 Teams using partial wins: {len(teams_with_partial_wins)}")
        print(f"   ✅ Teams using full wins only: {len(teams_full_wins_only)}")
        
        if len(teams_with_partial_wins) > 0:
            print(f"\n🟡 Teams using partial wins (top 5):")
            for i, (_, team) in enumerate(teams_with_partial_wins.head(5).iterrows()):
                print(f"   {i+1:2d}. {team['team']:<35} Partial wins: {team['partial_wins']}")
        
        print(f"\n🎯 NPI calculation complete!")
        print(f"📁 Results saved to: {output_file}")
        
        return final_teams
        
    except Exception as e:
        print(f"❌ Error processing: {e}")
        raise


if __name__ == "__main__":
    # Run with command line arguments
    # Examples:
    # python npi_calculator.py                    # 2025 season, full calculation
    # python npi_calculator.py --year 2024       # 2024 season, full calculation
    # python npi_calculator.py --season-only     # 2025 season, season results only
    # python npi_calculator.py -y 2024 -s        # 2024 season, season results only
    main()
