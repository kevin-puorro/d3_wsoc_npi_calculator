#!/usr/bin/env python3
"""
Script to create massey_pred_games_2025.csv by merging massey_games_2025.csv 
with prediction data and adding prediction columns.
"""

import pandas as pd
import re
from pathlib import Path

def parse_prediction_data(pred_file_path):
    """
    Parse the prediction data file and extract game predictions.
    Handles repeated headers and duplicate games by only parsing each unique game once.
    Returns a dictionary mapping game keys to prediction data.
    """
    predictions = {}
    seen_games = set()  # Track games we've already seen
    
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip header rows
        if line == "Date	Team	Standing	Scr	Pred	Pwin	Margin	Total":
            i += 1
            continue
        
        # Look for date lines (e.g., "Fri 08.29")
        if re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{2}\.\d{2}$', line):
            date = line
            
            # Next few lines should contain team information
            if i + 1 < len(lines) and i + 2 < len(lines):
                team1_line = lines[i + 1].strip()
                team2_line = lines[i + 2].strip()
                
                # Look for @ symbol to identify away/home teams
                if '@' in team2_line:
                    # Format: "Home Team" followed by "@ Away Team"
                    # But we need to swap to match games data format where @ = away team
                    away_team = team1_line  # First team becomes away team
                    home_team = team2_line.split('@')[1].strip()  # Second team (after @) becomes home team
                else:
                    # Format: "Home Team" followed by "Away Team" (no @ symbol)
                    # Swap to match games data format
                    away_team = team1_line  # First team becomes away team
                    home_team = team2_line  # Second team becomes home team
                
                # Create a unique game identifier
                game_id = f"{away_team}_{home_team}_{date}"
                
                # Only process this game if we haven't seen it before
                if game_id not in seen_games:
                    seen_games.add(game_id)
                    
                    # Look for scores and win probabilities in subsequent lines
                    actual_scores = []  # First two scores: away, home
                    predicted_scores = []  # Second two scores: away, home
                    win_probs = []  # First = away, second = home
                    
                    j = i + 3
                    while j < min(i + 15, len(lines)) and (len(actual_scores) < 2 or len(predicted_scores) < 2 or len(win_probs) < 2):
                        data_line = lines[j].strip()
                        
                        # Look for score patterns (e.g., "0", "1", "2")
                        if re.match(r'^\d+$', data_line):
                            if len(actual_scores) < 2:
                                actual_scores.append(data_line)
                            elif len(predicted_scores) < 2:
                                predicted_scores.append(data_line)
                        # Look for win percentage patterns (e.g., "86 %", "5 %")
                        elif re.match(r'^\d+\s*%$', data_line):
                            win_probs.append(data_line)
                        
                        j += 1
                    
                    # Store predictions if we have all the data we need
                    if len(actual_scores) >= 2 and len(predicted_scores) >= 2 and len(win_probs) >= 2:
                        try:
                            # Extract predicted scores (away, home) - first team is away, second is home
                            away_pred_score = float(predicted_scores[0])
                            home_pred_score = float(predicted_scores[1])
                            
                            # Extract win probabilities (away, home) - first percentage is away, second is home
                            away_win_prob = float(win_probs[0].replace('%', '').strip()) / 100
                            home_win_prob = float(win_probs[1].replace('%', '').strip()) / 100
                            
                            predictions[game_id] = {
                                'home_pred_score': home_pred_score,
                                'away_pred_score': away_pred_score,
                                'home_win_prob': round(home_win_prob, 3),
                                'away_win_prob': round(away_win_prob, 3)
                            }
                        except (ValueError, IndexError) as e:
                            pass
                    
                    i = j
                else:
                    # Skip to next date to avoid processing duplicate games
                    i += 1
                    while i < len(lines) and not re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{2}\.\d{2}$', lines[i].strip()):
                        i += 1
            else:
                i += 1
        else:
            i += 1
    
    return predictions

def create_massey_pred_games():
    """
    Main function to create massey_pred_games_2025.csv
    """
    # File paths
    base_dir = Path(__file__).parent.parent
    games_file = base_dir / "data" / "2025" / "massey_games_2025.csv"
    pred_file = base_dir / "data" / "2025" / "massey_pred_data_2025.txt"
    output_file = base_dir / "data" / "2025" / "massey_pred_games_2025.csv"
    
    print(f"Reading games data from: {games_file}")
    print(f"Reading prediction data from: {pred_file}")
    print(f"Output will be saved to: {output_file}")
    
    # Read the original games data
    try:
        games_df = pd.read_csv(games_file)
        print(f"Successfully read {len(games_df)} games from massey_games_2025.csv")
    except FileNotFoundError:
        print(f"Error: Could not find {games_file}")
        return
    except Exception as e:
        print(f"Error reading games file: {e}")
        return
    
    # Parse prediction data
    try:
        predictions = parse_prediction_data(pred_file)
        print(f"Successfully parsed {len(predictions)} unique game predictions")
        print(f"Total unique games found: {len(set(predictions.keys()))}")
    except Exception as e:
        print(f"Error parsing prediction data: {e}")
        return
    
    # Create a copy of the games dataframe
    pred_games_df = games_df.copy()
    
    # Add the new prediction columns
    pred_games_df['home_pred_score'] = None
    pred_games_df['away_pred_score'] = None
    pred_games_df['home_win_prob'] = None
    pred_games_df['away_win_prob'] = None
    
    # Match predictions to games
    matched_count = 0
    for idx, row in pred_games_df.iterrows():
        # Create a game key to match with predictions
        # Format: away_team_home_team_date
        date_str = pd.to_datetime(row['date']).strftime('%a %m.%d')
        game_key = f"{row['away_team']}_{row['home_team']}_{date_str}"
        
        # Try to find a match in predictions
        if game_key in predictions:
            pred_data = predictions[game_key]
            pred_games_df.at[idx, 'home_pred_score'] = pred_data['home_pred_score']
            pred_games_df.at[idx, 'away_pred_score'] = pred_data['away_pred_score']
            pred_games_df.at[idx, 'home_win_prob'] = pred_data['home_win_prob']
            pred_games_df.at[idx, 'away_win_prob'] = pred_data['away_win_prob']
            matched_count += 1
    
    print(f"Matched predictions for {matched_count} out of {len(pred_games_df)} games")
    
    # Save the new file
    try:
        pred_games_df.to_csv(output_file, index=False)
        print(f"Successfully created {output_file}")
        print(f"File contains {len(pred_games_df)} games with {matched_count} predictions")
        
        # Show sample of the new columns
        print("\nSample of new prediction columns:")
        sample_cols = ['date', 'away_team', 'home_team', 'home_pred_score', 'away_pred_score', 'home_win_prob', 'away_win_prob']
        print(pred_games_df[sample_cols].head(10))
        
        # Show some statistics
        print(f"\nPrediction coverage:")
        print(f"Games with predictions: {matched_count}")
        print(f"Games without predictions: {len(pred_games_df) - matched_count}")
        print(f"Coverage percentage: {(matched_count / len(pred_games_df) * 100):.1f}%")
        
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    create_massey_pred_games()
