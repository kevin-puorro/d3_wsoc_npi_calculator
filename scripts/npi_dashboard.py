#!/usr/bin/env python3
"""
NPI Dashboard using Streamlit
Simple dashboard to view NPI ratings for all teams
"""

import streamlit as st
import pandas as pd
import os
import uuid
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NPI Dashboard",
    page_icon="⚽",
    layout="wide"
)



def load_npi_data(year):
    """Load NPI results data for specified year"""
    try:
        # Use absolute path resolution to avoid working directory issues
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Look for NPI results file
        npi_file = data_dir / f"npi_ratings_{year}.csv"
        
        if not npi_file.exists():
            st.error(f"NPI results file not found for {year}: {npi_file.absolute()}")
            return None
        
        # Load data
        df = pd.read_csv(npi_file)
        return df
        
    except Exception as e:
        st.error(f"Error loading {year} data: {e}")
        return None

def load_games_data(year):
    """Load games data for schedules for specified year"""
    try:
        # Use absolute path resolution to avoid working directory issues
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Look for filtered games file
        games_file = data_dir / f"massey_games_{year}_filtered.csv"
        
        if not games_file.exists():
            st.error(f"Games file not found for {year}: {games_file.absolute()}")
            return None
        
        # Load data
        df = pd.read_csv(games_file)
        return df
        
    except Exception as e:
        st.error(f"Error loading {year} games data: {e}")
        return None

def get_available_years():
    """Get list of years that have data available"""
    try:
        # Use absolute path resolution to avoid working directory issues
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data"
        
        if not data_dir.exists():
            st.error(f"Data directory not found: {data_dir.absolute()}")
            return [2024]  # Fallback to 2024
        
        available_years = []
        for year_dir in data_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = int(year_dir.name)
                npi_file = year_dir / f"npi_ratings_{year}.csv"
                if npi_file.exists():
                    available_years.append(year)
        
        return sorted(available_years, reverse=True)
    except Exception as e:
        st.error(f"Error checking available years: {e}")
        return [2024]  # Fallback to 2024


def run_npi_simulation(year, user_id):
    """Run NPI calculator for simulation with user-specific files"""
    try:
        # Import the npi_calculator module directly instead of using subprocess
        import sys
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir))
        
        # Import the main function from npi_calculator
        from npi_calculator import main
        
        # Set up arguments for simulation mode by monkey-patching sys.argv
        import sys
        original_argv = sys.argv.copy()
        sys.argv = ['npi_calculator.py', '--year', str(year), '--simulation', '--user-id', str(user_id)]
        
        try:
            # Capture stdout to get the output
            import io
            import contextlib
            
            output_capture = io.StringIO()
            with contextlib.redirect_stdout(output_capture):
                # Run the main function
                main()
            
            output = output_capture.getvalue()
            return True, output, None
        finally:
            # Restore original argv
            sys.argv = original_argv
        
    except Exception as e:
        return False, None, str(e)


def reset_simulation_data(year, user_id):
    """Reset simulation data by copying filtered data back to user-specific simulation file"""
    try:
        # Get script directory and project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Source file (filtered data)
        filtered_file = data_dir / f"massey_games_{year}_filtered.csv"
        
        # Destination file (user-specific simulation data)
        sim_file = data_dir / f"massey_games_{year}_simulation_{user_id}.csv"
        
        # User-specific simulation NPI results file (to be cleared)
        sim_npi_file = data_dir / f"npi_ratings_{year}_simulation_{user_id}.csv"
        
        if not filtered_file.exists():
            return False, f"Filtered data file not found: {filtered_file}"
        
        # Read the filtered data
        filtered_df = pd.read_csv(filtered_file)
        
        # Copy to user-specific simulation file
        filtered_df.to_csv(sim_file, index=False)
        
        # Clear any existing user-specific simulation NPI results
        if sim_npi_file.exists():
            sim_npi_file.unlink()  # Delete the file
            print(f"Cleared existing simulation NPI results for user {user_id}: {sim_npi_file}")
        
        return True, f"Reset simulation data for user {user_id}: copied {len(filtered_df)} games from filtered data and cleared NPI results"
        
    except Exception as e:
        return False, f"Error resetting simulation data: {e}"


def apply_simulation_changes(year, selected_team, sim_data, user_id):
    """Apply simulation changes to the user-specific simulation data file"""
    try:
        # Get script directory and project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Load the user-specific simulation data file
        sim_file = data_dir / f"massey_games_{year}_simulation_{user_id}.csv"
        
        # Safety check - verify we're working with simulation file, not original
        original_npi_file = data_dir / f"npi_ratings_{year}.csv"
        if original_npi_file.exists():
            original_size = original_npi_file.stat().st_size
            print(f"SAFETY CHECK: Original NPI file size: {original_size} bytes")
        
        if not sim_file.exists():
            # Create user-specific simulation file by copying from filtered data
            print(f"User-specific simulation file not found, creating from filtered data...")
            filtered_file = data_dir / f"massey_games_{year}_filtered.csv"
            
            if not filtered_file.exists():
                return False, f"Filtered data file not found: {filtered_file}"
            
            # Copy filtered data to user-specific simulation file
            import shutil
            shutil.copy2(filtered_file, sim_file)
            print(f"Created user-specific simulation file: {sim_file}")
        
        print(f"SAFETY CHECK: Working with simulation file: {sim_file}")
        print(f"SAFETY CHECK: Original NPI file: {original_npi_file}")
        
        # Load current simulation data
        sim_df = pd.read_csv(sim_file)
        
        # Apply each change to the simulation data
        changes_applied = 0
        print(f"DEBUG: Total changes to process: {len([c for c in sim_data if c['changed']])}")
        print(f"DEBUG: All sim_data: {sim_data}")
        
        for change in sim_data:
            if change['changed']:
                print(f"Processing change: {change}")  # Debug info
                
                # Find the game in simulation data using multiple matching strategies
                # First try to match by opponent name and date
                date_match = pd.to_datetime(change['date']).strftime('%Y-%m-%d')
                
                # Try multiple matching strategies
                mask = None
                
                # Strategy 1: Direct opponent name + date match (handles multiple games between same teams)
                # Convert change date to match CSV date format
                change_date = pd.to_datetime(change['date']).strftime('%Y-%m-%d')
                print(f"DEBUG: Looking for game - {selected_team} vs {change['opponent']} ({change['home_away']}) on {change_date}")
                print(f"DEBUG: Available teams in sim_df: {sorted(sim_df['home_team'].unique())}")
                print(f"DEBUG: Available away teams in sim_df: {sorted(sim_df['away_team'].unique())}")
                print(f"DEBUG: Looking for date: {change_date}")
                
                # ENHANCED GAME MATCHING: Handle multiple games between same teams and team role confusion
                # First, find ALL games between these two teams
                all_games_between_teams = sim_df[
                    ((sim_df['home_team'] == selected_team) & (sim_df['away_team'] == change['opponent'])) |
                    ((sim_df['away_team'] == selected_team) & (sim_df['home_team'] == change['opponent']))
                ]
                
                print(f"DEBUG: Found {len(all_games_between_teams)} total games between {selected_team} and {change['opponent']}")
                
                # Try to match by date first (most precise)
                date_matches = all_games_between_teams[
                    pd.to_datetime(all_games_between_teams['date']).dt.strftime('%Y-%m-%d') == change_date
                ]
                
                if len(date_matches) > 0:
                    print(f"DEBUG: Found {len(date_matches)} games on {change_date}")
                    
                    # If multiple games on same date, we need to determine which one based on team roles
                    if len(date_matches) > 1:
                        print(f"DEBUG: Multiple games on same date - need to determine correct one")
                        print(f"DEBUG: UI shows {selected_team} as {change['home_away']}")
                        
                        # Check each game to see which one matches the UI description
                        for _, game in date_matches.iterrows():
                            if game['home_team'] == selected_team:
                                csv_role = "HOME"
                            else:
                                csv_role = "AWAY"
                            
                            print(f"DEBUG: Game {game.name}: {selected_team} is {csv_role} in CSV (vs {change['home_away']} in UI)")
                            
                            if csv_role == change['home_away']:
                                print(f"DEBUG: ✅ Found matching game with correct team role!")
                                mask = sim_df.index == game.name
                                break
                        else:
                            print(f"DEBUG: ❌ No game found with matching team role")
                            mask = pd.Series([False] * len(sim_df), index=sim_df.index)
                    else:
                        # Single game on this date - use it
                        game = date_matches.iloc[0]
                        mask = sim_df.index == game.name
                        print(f"DEBUG: Single game on {change_date} - using it")
                else:
                    print(f"DEBUG: No games found on {change_date}")
                    
                    # Fallback: Try to find any game between these teams (date might be slightly off)
                    if len(all_games_between_teams) > 0:
                        print(f"DEBUG: No date match, but found games between teams - using first one")
                        game = all_games_between_teams.iloc[0]
                        mask = sim_df.index == game.name
                        print(f"DEBUG: WARNING: Using game on {game['date']} instead of {change_date}")
                    else:
                        print(f"DEBUG: No games found between {selected_team} and {change['opponent']} at all")
                        mask = pd.Series([False] * len(sim_df), index=sim_df.index)
                
                print(f"DEBUG: Enhanced matching result: {mask.any()} matches found")
                
                # Strategy 2: If still no match, try fuzzy matching on opponent names (legacy fallback)
                if not mask.any():
                    print(f"No enhanced match found for {change['opponent']}, trying fuzzy match...")
                    # Look for games where the selected team plays and try to match opponent names
                    team_games = sim_df[(sim_df['home_team'] == selected_team) | (sim_df['away_team'] == selected_team)]
                    
                    for _, game in team_games.iterrows():
                        home_opponent = game['home_team'] if game['away_team'] == selected_team else game['away_team']
                        if change['opponent'] in home_opponent or home_opponent in change['opponent']:
                            # Found a fuzzy match
                            mask = sim_df.index == game.name
                            print(f"Fuzzy match found: {change['opponent']} matches {home_opponent}")
                            break
                
                if mask.any():
                    print(f"DEBUG: Mask type: {type(mask)}, Mask: {mask}")
                    
                    # Get the first matching game index
                    if isinstance(mask, pd.Series):
                        game_idx = mask.idxmax()
                        print(f"Using pandas idxmax: {game_idx}")
                    else:
                        # If mask is a numpy array, find the first True index
                        game_idx = mask.nonzero()[0][0]
                        print(f"Using numpy nonzero: {game_idx}")
                    
                    print(f"Found game at index {game_idx}: {sim_df.loc[game_idx]}")  # Debug info
                    print(f"Updating game: {change['opponent']} from {change['original_result']} to {change['simulated_result']}")
                    
                    # Update scores based on simulated result
                    # First, determine the selected team's actual role in this specific game
                    selected_team_is_home = sim_df.loc[game_idx, 'home_team'] == selected_team
                    print(f"DEBUG: In CSV data, {selected_team} is {'HOME' if selected_team_is_home else 'AWAY'}")
                    if change['simulated_result'] == 'W':
                        if selected_team_is_home:
                            # Selected team is home_team in CSV, so they should win (higher score)
                            sim_df.loc[game_idx, 'home_score'] = 2  # home_team wins
                            sim_df.loc[game_idx, 'away_score'] = 1  # away_team loses
                            print(f"DEBUG WIN: {selected_team} (home_team in CSV) wins against {change['opponent']} - Score: {sim_df.loc[game_idx, 'home_score']}-{sim_df.loc[game_idx, 'away_score']}")
                        else:
                            # Selected team is away_team in CSV, so they should win (higher score)
                            sim_df.loc[game_idx, 'away_score'] = 2  # away_team wins
                            sim_df.loc[game_idx, 'home_score'] = 1  # home_team loses
                            print(f"DEBUG WIN: {selected_team} (away_team in CSV) wins against {change['opponent']} - Score: {sim_df.loc[game_idx, 'home_score']}-{sim_df.loc[game_idx, 'away_score']}")
                    elif change['simulated_result'] == 'L':
                        if selected_team_is_home:
                            # Selected team is home_team in CSV, so they should lose (lower score)
                            sim_df.loc[game_idx, 'home_score'] = 1  # home_team loses
                            sim_df.loc[game_idx, 'away_score'] = 2  # away_team wins
                            print(f"DEBUG LOSS: {selected_team} (home_team in CSV) loses to {change['opponent']} - Score: {sim_df.loc[game_idx, 'home_score']}-{sim_df.loc[game_idx, 'away_score']}")
                        else:
                            # Selected team is away_team in CSV, so they should lose (lower score)
                            sim_df.loc[game_idx, 'away_score'] = 1  # away_team loses
                            sim_df.loc[game_idx, 'home_score'] = 2  # home_team wins
                            print(f"DEBUG LOSS: {selected_team} (away_team in CSV) loses to {change['opponent']} - Score: {sim_df.loc[game_idx, 'home_score']}-{sim_df.loc[game_idx, 'away_score']}")
                    else: # Tie
                        sim_df.loc[game_idx, 'home_score'] = 1
                        sim_df.loc[game_idx, 'away_score'] = 1
                        print(f"DEBUG TIE: {selected_team} ties with {change['opponent']} - Score: {sim_df.loc[game_idx, 'home_score']}-{sim_df.loc[game_idx, 'away_score']}")
                    
                    # Update game_note to remove SCH if it was scheduled
                    if sim_df.loc[game_idx, 'game_note'] == 'Sch':
                        sim_df.loc[game_idx, 'game_note'] = ''
                    
                    # Check if result differs from predicted result and remove "pred" indicator if it does
                    if pd.notna(sim_df.loc[game_idx, 'game_note']) and 'pred' in str(sim_df.loc[game_idx, 'game_note']):
                        predicted_result = sim_df.loc[game_idx, 'predicted_result']
                        if predicted_result and change['simulated_result'] != predicted_result:
                            # Result differs from prediction, remove "pred" indicator
                            current_note = str(sim_df.loc[game_idx, 'game_note'])
                            note_parts = [part.strip() for part in current_note.split(';') if part.strip() != 'pred']
                            if note_parts:
                                sim_df.loc[game_idx, 'game_note'] = "; ".join(note_parts)
                            else:
                                sim_df.loc[game_idx, 'game_note'] = ''
                            print(f"Removed 'pred' indicator for {change['opponent']} - result changed from {predicted_result} to {change['simulated_result']}")
                    
                    # DEBUG: Show the exact game after score update
                    print(f"DEBUG UPDATE: Game details after score update:")
                    print(f"  - Home team: {sim_df.loc[game_idx, 'home_team']}")
                    print(f"  - Away team: {sim_df.loc[game_idx, 'away_team']}")
                    print(f"  - Home score: {sim_df.loc[game_idx, 'home_score']} (updated)")
                    print(f"  - Away score: {sim_df.loc[game_idx, 'away_score']} (updated)")
                    print(f"  - Date: {sim_df.loc[game_idx, 'date']}")
                    
                    # DEBUG: Verify the score assignment logic
                    if change['simulated_result'] == 'W':
                        if change['home_away'] == 'HOME':
                            # Selected team is away_team, should have higher score
                            expected_away = 2
                            expected_home = 1
                            actual_away = sim_df.loc[game_idx, 'away_score']
                            actual_home = sim_df.loc[game_idx, 'home_score']
                            print(f"  - HOME WIN VERIFICATION: away_score={actual_away} (expected {expected_away}), home_score={actual_home} (expected {expected_home})")
                            if actual_away == expected_away and actual_home == expected_home:
                                print(f"  ✅ HOME WIN SCORES CORRECT")
                            else:
                                print(f"  ❌ HOME WIN SCORES WRONG!")
                        else:
                            # Selected team is home_team, should have higher score
                            expected_home = 2
                            expected_away = 1
                            actual_home = sim_df.loc[game_idx, 'home_score']
                            actual_away = sim_df.loc[game_idx, 'away_score']
                            print(f"  - AWAY WIN VERIFICATION: home_score={actual_home} (expected {expected_home}), away_score={actual_away} (expected {expected_away})")
                            if actual_home == expected_home and actual_away == expected_away:
                                print(f"  ✅ AWAY WIN SCORES CORRECT")
                            else:
                                print(f"  ❌ AWAY WIN SCORES WRONG!")
                    
                    changes_applied += 1
                    print(f"Applied change {changes_applied}: {change['opponent']} -> {change['simulated_result']}")
                else:
                    print(f"WARNING: Could not find game for {selected_team} vs {change['opponent']} ({change['home_away']})")
                    print(f"Available games for {selected_team}:")
                    team_games = sim_df[(sim_df['home_team'] == selected_team) | (sim_df['away_team'] == selected_team)]
                    print(team_games[['home_team', 'away_team', 'home_score', 'away_score', 'game_note']].to_string())
        
        # Save modified simulation data
        sim_df.to_csv(sim_file, index=False)
        
        # Debug: Verify the changes were actually saved
        print(f"DEBUG: Saved simulation file: {sim_file}")
        print(f"DEBUG: File size: {sim_file.stat().st_size} bytes")
        
        # Verify the changes are in the saved file
        verification_df = pd.read_csv(sim_file)
        print(f"DEBUG: Verification - loaded {len(verification_df)} games from saved file")
        
        # ENHANCED DEBUGGING: Check which games were NOT found/updated
        print(f"DEBUG: SUMMARY - Total changes requested: {len([c for c in sim_data if c['changed']])}")
        print(f"DEBUG: SUMMARY - Changes successfully applied: {changes_applied}")
        
        if changes_applied < len([c for c in sim_data if c['changed']]):
            print(f"DEBUG: ⚠️  {len([c for c in sim_data if c['changed']]) - changes_applied} changes were NOT applied!")
            print("DEBUG: Checking which games failed to match...")
            
            # Find all games for the selected team in the verification data
            team_games_in_file = verification_df[(verification_df['home_team'] == selected_team) | (verification_df['away_team'] == selected_team)]
            print(f"DEBUG: Found {len(team_games_in_file)} games for {selected_team} in saved file")
            
            # Show all the requested changes
            for change in sim_data:
                if change['changed']:
                    print(f"DEBUG: Requested change: {change['opponent']} ({change['home_away']}) on {change['date']} -> {change['simulated_result']}")
                    
                    # Try to find this game in the saved file
                    if change['home_away'] == 'HOME':
                        # Team is HOME in UI, so they are actually away_team in CSV
                        match_mask = (verification_df['away_team'] == selected_team) & (verification_df['home_team'] == change['opponent'])
                    else:
                        # Team is AWAY in UI, so they are actually home_team in CSV
                        match_mask = (verification_df['home_team'] == selected_team) & (verification_df['away_team'] == change['opponent'])
                    
                    if match_mask.any():
                        matched_game = verification_df[match_mask].iloc[0]
                        print(f"  ✅ Found in saved file: {matched_game['home_team']} vs {matched_game['away_team']} - Score: {matched_game['home_score']}-{matched_game['away_score']}")
                        
                        # Check if the score reflects the requested change
                        if change['simulated_result'] == 'W':
                            if change['home_away'] == 'HOME':
                                # Selected team (away_team) should have higher score
                                expected_score = matched_game['away_score'] > matched_game['home_score']
                            else:
                                # Selected team (home_team) should have higher score
                                expected_score = matched_game['home_score'] > matched_game['away_score']
                            
                            if expected_score:
                                print(f"  ✅ WIN correctly applied")
                            else:
                                print(f"  ❌ WIN NOT correctly applied - scores don't reflect win")
                        elif change['simulated_result'] == 'L':
                            if change['home_away'] == 'HOME':
                                # Selected team (away_team) should have lower score
                                expected_score = matched_game['away_score'] < matched_game['home_score']
                            else:
                                # Selected team (home_team) should have lower score
                                expected_score = matched_game['home_score'] < matched_game['away_score']
                            
                            if expected_score:
                                print(f"  ✅ LOSS correctly applied")
                            else:
                                print(f"  ❌ LOSS NOT correctly applied - scores don't reflect loss")
                        elif change['simulated_result'] == 'T':
                            if matched_game['home_score'] == matched_game['away_score']:
                                print(f"  ✅ TIE correctly applied")
                            else:
                                print(f"  ❌ TIE NOT correctly applied - scores are not equal")
                    else:
                        print(f"  ❌ NOT FOUND in saved file!")
                        print(f"     Looking for: {selected_team} vs {change['opponent']} ({change['home_away']})")
                        print(f"     Available opponents for {selected_team}:")
                        available_opponents = []
                        for _, game in team_games_in_file.iterrows():
                            opp = game['away_team'] if game['home_team'] == selected_team else game['home_team']
                            available_opponents.append(opp)
                        print(f"     {sorted(set(available_opponents))}")
        
        # Check if our changes are visible in the saved file
        for change in sim_data:
            if change['changed']:
                print(f"DEBUG VERIFICATION: Looking for change: {change['opponent']} ({change['home_away']}) on {change['date']}")
                
                # Find the game in the verification data
                if change['home_away'] == 'HOME':
                    mask = (verification_df['away_team'] == selected_team) & (verification_df['home_team'] == change['opponent'])
                    print(f"  - Looking for away_team={selected_team} AND home_team={change['opponent']}")
                else:
                    mask = (verification_df['home_team'] == selected_team) & (verification_df['away_team'] == change['opponent'])
                    print(f"  - Looking for home_team={selected_team} AND away_team={change['opponent']}")
                
                print(f"  - Found {mask.sum()} matching games")
                
                if mask.any():
                    game_idx = mask.idxmax() if isinstance(mask, pd.Series) else mask.nonzero()[0][0]
                    saved_game = verification_df.loc[game_idx]
                    print(f"DEBUG: Change verification for {change['opponent']}:")
                    print(f"  - Home score: {saved_game['home_score']}")
                    print(f"  - Away score: {saved_game['away_score']}")
                    print(f"  - Game note: {saved_game['game_note']}")
                    print(f"  - Date: {saved_game['date']}")
                    
                    # Additional verification for ties
                    if change['simulated_result'] == 'T':
                        if saved_game['home_score'] == saved_game['away_score']:
                            print(f"  ✅ TIE VERIFIED: Both scores are equal ({saved_game['home_score']})")
                        else:
                            print(f"  ❌ TIE ERROR: Scores are not equal! Home: {saved_game['home_score']}, Away: {saved_game['away_score']}")
                    
                    # Additional verification for wins
                    if change['simulated_result'] == 'W':
                        if change['home_away'] == 'HOME':
                            # Selected team is away_team, should have higher score
                            if saved_game['away_score'] > saved_game['home_score']:
                                print(f"  ✅ WIN VERIFIED: Away team (selected) has higher score ({saved_game['away_score']} > {saved_game['home_score']})")
                            else:
                                print(f"  ❌ WIN ERROR: Away team (selected) should have higher score but doesn't! Away: {saved_game['away_score']}, Home: {saved_game['home_score']}")
                        else:
                            # Selected team is home_team, should have higher score
                            if saved_game['home_score'] > saved_game['away_score']:
                                print(f"  ✅ WIN VERIFIED: Home team (selected) has higher score ({saved_game['home_score']} > {saved_game['away_score']})")
                            else:
                                print(f"  ❌ WIN ERROR: Home team (selected) should have higher score but doesn't! Home: {saved_game['home_score']}, Away: {saved_game['away_score']}")
                else:
                    print(f"  ❌ NO MATCH FOUND for {change['opponent']} on {change['date']}")
        
        # Final safety check - verify original file wasn't modified
        if original_npi_file.exists():
            new_original_size = original_npi_file.stat().st_size
            if new_original_size == original_size:
                print(f"SAFETY CHECK: Original NPI file size unchanged: {new_original_size} bytes")
            else:
                print(f"SAFETY CHECK: WARNING! Original NPI file size changed from {original_size} to {new_original_size} bytes!")
                return False, f"SAFETY VIOLATION: Original NPI file was modified! This should never happen."
        
        return True, f"Applied {changes_applied} changes to data"
        
    except Exception as e:
        return False, f"Error applying simulation changes: {e}"


def cleanup_user_files(user_id: str):
    """Clean up all files associated with a specific user"""
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data"
        
        files_removed = 0
        
        # Clean up files for all years
        for year_dir in data_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                year = year_dir.name
                
                # Remove user-specific simulation files
                sim_games_file = year_dir / f"massey_games_{year}_simulation_{user_id}.csv"
                sim_npi_file = year_dir / f"npi_ratings_{year}_simulation_{user_id}.csv"
                
                if sim_games_file.exists():
                    sim_games_file.unlink()
                    files_removed += 1
                    print(f"Removed user simulation games file: {sim_games_file}")
                
                if sim_npi_file.exists():
                    sim_npi_file.unlink()
                    files_removed += 1
                    print(f"Removed user simulation NPI file: {sim_npi_file}")
        
        return True, f"Cleaned up {files_removed} user files"
        
    except Exception as e:
        return False, f"Error cleaning up user files: {e}"


def cleanup_expired_sessions():
    """Clean up files for expired sessions (older than 1 hour)"""
    try:
        current_time = time.time()
        expired_users = []
        
        # Check for expired sessions (older than 1 hour)
        for user_id, session_data in st.session_state.get('active_sessions', {}).items():
            if current_time - session_data['last_activity'] > 3600:  # 1 hour = 3600 seconds
                expired_users.append(user_id)
        
        # Clean up expired sessions
        for user_id in expired_users:
            cleanup_user_files(user_id)
            del st.session_state['active_sessions'][user_id]
            print(f"Cleaned up expired session for user: {user_id}")
        
        return len(expired_users)
        
    except Exception as e:
        print(f"Error cleaning up expired sessions: {e}")
        return 0


def apply_predicted_scores(year, selected_team, user_id):
    """Apply predicted scores from Massey prediction data to simulation data"""
    try:
        # Get script directory and project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Source files
        pred_games_file = data_dir / f"massey_pred_games_{year}_filtered_complete.csv"
        sim_games_file = data_dir / f"massey_games_{year}_simulation_{user_id}.csv"
        
        # Check if files exist
        if not pred_games_file.exists():
            return False, f"Predicted games file not found: {pred_games_file}"
        
        if not sim_games_file.exists():
            # Create user-specific simulation file by copying from filtered data
            filtered_file = data_dir / f"massey_games_{year}_filtered.csv"
            
            if not filtered_file.exists():
                return False, f"Filtered data file not found: {filtered_file}"
            
            # Copy filtered data to user-specific simulation file
            import shutil
            shutil.copy2(filtered_file, sim_games_file)
            print(f"Created user-specific simulation file: {sim_games_file}")
        
        # Load both datasets
        pred_df = pd.read_csv(pred_games_file)
        sim_df = pd.read_csv(sim_games_file)
        
        # Find all scheduled games in simulation data that have predictions
        # We'll process ALL games with predictions, not just the selected team's games
        scheduled_sim_games = sim_df[
            (pd.notna(sim_df['game_note'])) & 
            (sim_df['game_note'].str.contains('Sch', na=False))
        ]
        
        if scheduled_sim_games.empty:
            return False, f"No scheduled games found in simulation data"
        
        # Apply predicted scores to simulation data
        changes_applied = 0
        for _, sim_game in scheduled_sim_games.iterrows():
            # Find matching game in prediction data
            pred_mask = (
                (pred_df['home_team'] == sim_game['home_team']) & 
                (pred_df['away_team'] == sim_game['away_team']) &
                (pred_df['date'] == sim_game['date'])
            )
            
            if pred_mask.any():
                pred_game = pred_df[pred_mask].iloc[0]
                
                # Check if this game has predicted scores
                if pd.notna(pred_game.get('home_pred_score')) and pd.notna(pred_game.get('away_pred_score')):
                    
                    # Find the game index in simulation data
                    game_idx = sim_game.name
                    
                    # Update scores - directly assign predicted scores to the correct CSV columns
                    sim_df.loc[game_idx, 'home_score'] = int(round(pred_game['home_pred_score']))
                    sim_df.loc[game_idx, 'away_score'] = int(round(pred_game['away_pred_score']))
                    
                    # Calculate predicted result from the perspective of the selected team
                    home_score = int(round(pred_game['home_pred_score']))
                    away_score = int(round(pred_game['away_pred_score']))
                    
                    # Determine if the selected team is home or away in this game
                    if sim_game['home_team'] == selected_team:
                        # Selected team is home team
                        if home_score > away_score:
                            predicted_result = "W"  # Selected team wins
                        elif away_score > home_score:
                            predicted_result = "L"  # Selected team loses
                        else:
                            predicted_result = "T"  # Tie
                    else:
                        # Selected team is away team
                        if away_score > home_score:
                            predicted_result = "W"  # Selected team wins
                        elif home_score > away_score:
                            predicted_result = "L"  # Selected team loses
                        else:
                            predicted_result = "T"  # Tie
                    
                    # Store predicted result in a new column for easy access
                    sim_df.loc[game_idx, 'predicted_result'] = predicted_result
                    
                    # Add "pred" game note and remove "Sch" if it exists
                    current_note = str(sim_df.loc[game_idx, 'game_note']) if pd.notna(sim_df.loc[game_idx, 'game_note']) else ""
                    
                    # Remove "Sch" and add "pred"
                    if current_note.strip() == "":
                        sim_df.loc[game_idx, 'game_note'] = "pred"
                    else:
                        # Remove "Sch" and add "pred"
                        note_parts = [part.strip() for part in current_note.split(';') if part.strip() != 'Sch']
                        if "pred" not in note_parts:
                            note_parts.append("pred")
                        sim_df.loc[game_idx, 'game_note'] = "; ".join(note_parts)
                    
                    changes_applied += 1
                    team_role = "HOME" if sim_game['home_team'] == selected_team else "AWAY"
                    print(f"Applied predicted scores for {sim_game['home_team']} vs {sim_game['away_team']} - {selected_team} is {team_role} - Predicted result: {predicted_result}")
        
        # Save updated simulation data
        sim_df.to_csv(sim_games_file, index=False)
        
        return True, f"Applied predicted scores to {changes_applied} scheduled games across all teams"
        
    except Exception as e:
        return False, f"Error applying predicted scores: {e}"


def load_simulation_npi_data(year):
    """Load simulation NPI results for a specific year"""
    try:
        # Use absolute path resolution to avoid working directory issues
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Look for simulation NPI ratings file
        sim_npi_file = data_dir / f"npi_ratings_{year}_simulation.csv"
        
        if not sim_npi_file.exists():
            return None  # Don't show error for simulation data
        
        # Load data
        df = pd.read_csv(sim_npi_file)
        return df
        
    except Exception as e:
        return None  # Don't show error for simulation data


def get_opponent_npi(opponent_name, npi_data):
    """
    Get opponent NPI rating from the NPI ratings CSV.
    
    Args:
        opponent_name: Name of the opponent team
        npi_data: DataFrame containing NPI ratings
        
    Returns:
        NPI rating if found, None otherwise
    """
    if npi_data is None or npi_data.empty:
        print(f"DEBUG: No NPI data available")
        return None
    
    # Direct lookup - team names should match exactly
    opponent_npi_row = npi_data[npi_data['team'] == opponent_name]
    
    if not opponent_npi_row.empty:
        npi_rating = opponent_npi_row.iloc[0]['npi_rating']
        print(f"DEBUG: Found NPI for '{opponent_name}': {npi_rating}")
        return npi_rating
    else:
        print(f"DEBUG: No NPI data found for opponent: '{opponent_name}'")
        return None


def calculate_npi_for_game_result(opponent_npi, won=True, tied=False):
    """
    Calculate NPI score for a specific game result against an opponent.
    
    Args:
        opponent_npi: NPI rating of the opponent
        won: Whether the team won (True) or lost (False)
        tied: Whether the game ended in a tie
        
    Returns:
        NPI score for this specific game result
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
    
    # Calculate quality bonus (using iteration > 3 logic for consistency)
    quality_bonus = 0
    if won:
        quality_bonus = max(0, (opponent_npi - 54.00) * 0.500)
    elif tied:
        # Calculate full quality bonus, then divide by 2 for ties
        full_bonus = max(0, (opponent_npi - 54.00) * 0.500)
        quality_bonus = full_bonus / 2
    
    # Calculate total NPI
    total_npi = base_npi + quality_bonus
    return total_npi


def get_team_schedule(team_name, games_df, user_id=None, year=None):
    """Get schedule for a specific team, optionally including simulation data"""
    if games_df is None:
        return None
    
    # Try to load simulation data if user_id and year are provided
    sim_games_df = None
    if user_id and year:
        try:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            data_dir = project_root / "data" / str(year)
            sim_file = data_dir / f"massey_games_{year}_simulation_{user_id}.csv"
            
            if sim_file.exists():
                sim_games_df = pd.read_csv(sim_file)
                print(f"Loaded simulation data for {team_name}: {sim_file}")
        except Exception as e:
            print(f"Could not load simulation data: {e}")
    
    # Use simulation data if available, otherwise use original games data
    source_df = sim_games_df if sim_games_df is not None else games_df
    
    # Load NPI data to get opponent ratings for NPI score calculations
    npi_data = None
    if year:
        try:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            data_dir = project_root / "data" / str(year)
            npi_file = data_dir / f"npi_ratings_{year}.csv"
            
            print(f"DEBUG NPI LOADING: Looking for NPI file: {npi_file}")
            print(f"DEBUG NPI LOADING: File exists: {npi_file.exists()}")
            
            if npi_file.exists():
                npi_data = pd.read_csv(npi_file)
                print(f"DEBUG NPI LOADING: Loaded NPI data with {len(npi_data)} teams")
                print(f"DEBUG NPI LOADING: NPI data columns: {list(npi_data.columns)}")
                print(f"DEBUG NPI LOADING: First 5 teams: {npi_data['team'].head().tolist()}")
            else:
                print(f"DEBUG NPI LOADING: NPI file not found: {npi_file}")
        except Exception as e:
            print(f"DEBUG NPI LOADING: Error loading NPI data: {e}")
    
    # Find games where the team is either home or away
    team_games = source_df[
        (source_df['home_team'] == team_name) | 
        (source_df['away_team'] == team_name)
    ].copy()
    
    if team_games.empty:
        return None
    
    # Create a more readable schedule format
    schedule_data = []
    for _, game in team_games.iterrows():
        if game['home_team'] == team_name:
            # Team is playing at home
            opponent = game['away_team']
            home_away = "HOME"
            team_score = game['home_score']
            opp_score = game['away_score']
        else:
            # Team is playing away
            opponent = game['home_team']
            home_away = "AWAY"
            team_score = game['away_score']
            opp_score = game['home_score']
        
        # Check if game is scheduled
        is_scheduled = pd.notna(game['game_note']) and str(game['game_note']).strip() == 'Sch'
        
        # Determine result (only for completed games)
        if is_scheduled:
            result = "SCH"  # Scheduled
        else:
            if team_score > opp_score:
                result = "W"
            elif team_score < opp_score:
                result = "L"
            else:
                result = "T"
            
            # Debug: Show result calculation for ties
            if result == "T":
                print(f"DEBUG TIE CALCULATION: {team_name} vs {opponent} - Team Score: {team_score}, Opp Score: {opp_score} = TIE")
        
        # Check for overtime (only for completed games)
        overtime = ""
        if not is_scheduled and pd.notna(game['game_note']) and str(game['game_note']).strip() != "":
            if 'o1' in str(game['game_note']).lower() or 'o2' in str(game['game_note']).lower() or 'o3' in str(game['game_note']).lower():
                overtime = "OT"
        
        # Check for predicted scores and whether current result matches prediction
        has_pred_note = pd.notna(game['game_note']) and 'pred' in str(game['game_note']).lower()
        
        # Debug: Check what columns are available
        print(f"DEBUG: Available columns in game data: {list(game.keys())}")
        
        # Try different possible column names for predicted result
        predicted_result = None
        if 'predicted_result' in game:
            predicted_result = game['predicted_result']
        elif 'Predicted_Result' in game:
            predicted_result = game['Predicted_Result']
        elif 'predicted_result' in game:
            predicted_result = game['predicted_result']
        
        print(f"DEBUG: Predicted result found: {predicted_result}")
        
        # A game is considered "predicted" if it has the pred note AND has a predicted result
        # We don't require the current result to match the predicted result for display purposes
        is_predicted = has_pred_note and predicted_result is not None
        
        # Calculate NPI scores for beating, losing, and tying this opponent
        npi_win_score = None
        npi_loss_score = None
        npi_tie_score = None
        
        # Get opponent's NPI rating
        opponent_npi = get_opponent_npi(opponent, npi_data)
        
        if opponent_npi is not None:
            # Calculate NPI scores for win, loss, and tie scenarios
            npi_win_score = calculate_npi_for_game_result(opponent_npi, won=True, tied=False)
            npi_loss_score = calculate_npi_for_game_result(opponent_npi, won=False, tied=False)
            npi_tie_score = calculate_npi_for_game_result(opponent_npi, won=False, tied=True)
            print(f"DEBUG: Calculated NPI scores for {team_name} vs {opponent}: Win={npi_win_score:.2f}, Loss={npi_loss_score:.2f}, Tie={npi_tie_score:.2f}")
        else:
            print(f"DEBUG: No NPI data available for {opponent}")
        
        schedule_data.append({
            'Date': game['date'],
            'Opponent': opponent,
            'Home/Away': home_away,
            'Team Score': team_score,
            'Opponent Score': opp_score,
            'Result': result,
            'Overtime': overtime,
            'Status': 'Scheduled' if is_scheduled else 'Completed',
            'Predicted': is_predicted,
            'Predicted_Result': predicted_result,
            'NPI Win Value': npi_win_score,
            'NPI Tie Value': npi_tie_score,
            'NPI Loss Value': npi_loss_score
        })
    
    # Convert to DataFrame and sort by date
    schedule_df = pd.DataFrame(schedule_data)
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date'])
    schedule_df = schedule_df.sort_values('Date').reset_index(drop=True)
    
    return schedule_df

def main():
    # Initialize session tracking if not exists
    if 'active_sessions' not in st.session_state:
        st.session_state['active_sessions'] = {}
    
    # Generate a unique user ID for this session if not exists
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())
    
    # Track user activity
    current_time = time.time()
    user_id = st.session_state['user_id']
    st.session_state['active_sessions'][user_id] = {
        'last_activity': current_time,
        'created_at': current_time
    }
    
    # Clean up expired sessions (older than 1 hour)
    expired_count = cleanup_expired_sessions()
    if expired_count > 0:
        print(f"Cleaned up {expired_count} expired sessions")
    
    # Custom CSS for title background
    st.markdown("""
    <style>
        .title-background {
            background-color: #830019;
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title with custom background
    st.markdown('<div class="title-background"><h1> Division 3 Women\'s Soccer NPI Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Get available years and create year selector
    available_years = get_available_years()
    
    if not available_years:
        st.error("No data found for any year. Please ensure NPI calculations have been run.")
        st.stop()
    
    # Create sidebar for season selection
    with st.sidebar:
        st.header("Season Selection")
        
        # Year selection
        selected_year = st.selectbox(
            "Select Season:",
            options=available_years,
            index=0,
            help="Choose which season's data to display"
        )
        
        # Show available seasons info
        st.subheader("Available Seasons")
        for year in available_years:
            if year == selected_year:
                st.markdown(f"**🟢 {year}** (Currently Selected)")
            else:
                st.markdown(f"⚪ {year}")
        
        # Show special note for 2025 (mostly scheduled games)
        if selected_year == 2025:
            st.markdown("---")
    
    # Show current season in main area
    st.markdown(f"## {selected_year} Season Dashboard")
    
    # Load data for selected year
    npi_df = load_npi_data(selected_year)
    games_df = load_games_data(selected_year)
    sim_npi_df = load_simulation_npi_data(selected_year)
    
    # Check if we have at least one data source
    if npi_df is None and games_df is None:
        st.error(f"No data found for {selected_year}. Please ensure either NPI ratings or games data exists.")
        st.stop()
    
    # Show warning if no NPI data available
    if npi_df is None:
        st.warning(f"⚠️ No NPI ratings available for {selected_year} (this is normal for seasons with only scheduled games)")
        st.markdown("""
        <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <strong>📊 You can still view team schedules and individual game data.</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏆 NPI Rankings", "📅 Team Schedules", "🎯 NPI Projections"])
    
    # Tab 1: NPI Ratings
    with tab1:
        st.subheader("NPI Rankings")
        
        # Show NPI data from the CSV file (regardless of content)
        if npi_df is not None and not npi_df.empty:
            st.markdown("*Note: These NPI ratings have a margin of error of ±0.3*")
            
            # Sort by NPI rating (highest first)
            df_sorted = npi_df.sort_values('npi_rating', ascending=False).reset_index(drop=True)
            
            # Add ranking
            df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
            
            # Select only the columns we want to display
            columns_to_show = ['Rank', 'team', 'npi_rating', 'wins', 'losses', 'ties']
            df_display = df_sorted[columns_to_show].copy()
            
            # Round NPI rating to 2 decimal places
            df_display['npi_rating'] = df_display['npi_rating'].round(2)
            
            # Rename columns for cleaner display
            df_display = df_display.rename(columns={
                'team': 'Team',
                'npi_rating': 'NPI Rating',
                'wins': 'Win',
                'losses': 'Loss', 
                'ties': 'Tie'
            })
            
            # Display all teams
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>📅 **No NPI ratings available for this season.**</strong><br><br>
                This typically happens when:<br>
                - All games are scheduled (like 2025 season)<br>
                - No completed games exist yet<br>
                - NPI calculations haven't been run<br><br>
                <strong>You can still view team schedules in the Team Schedules tab!</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Team Schedules
    with tab2:
        st.subheader("Team Lookup")
        
        if games_df is None:
            st.error("Unable to load games data. Please ensure the filtered games file exists.")
            st.stop()
        
        # Team selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get list of all teams from games data (more reliable than NPI data)
            if games_df is not None and not games_df.empty:
                all_teams = sorted(list(set(games_df['home_team'].tolist() + games_df['away_team'].tolist())))
            else:
                # Fallback to NPI data if available
                all_teams = sorted(npi_df['team'].tolist()) if npi_df is not None and not npi_df.empty else []
            
            if not all_teams:
                st.error("No teams found in the data. Please ensure the games file exists.")
                st.stop()
            
            # Try to find University of Wisconsin-La Crosse as default, otherwise use first team
            default_index = 0
            target_team = "University of Wisconsin-La Crosse"
            if target_team in all_teams:
                default_index = all_teams.index(target_team)
            
            selected_team = st.selectbox(
                "Select a team to view their schedule:",
                options=all_teams,
                index=default_index
            )
        
        with col2:
            st.write("")
            st.write("")
            if selected_team:
                # Calculate team record from actual games data (more accurate)
                if games_df is not None and not games_df.empty:
                    team_games = games_df[
                        (games_df['home_team'] == selected_team) | 
                        (games_df['away_team'] == selected_team)
                    ].copy()
                    
                    # Filter out scheduled games for record calculation
                    completed_games = team_games[
                        ~(pd.notna(team_games['game_note']) & (team_games['game_note'].str.strip() == 'Sch'))
                    ]
                    
                    wins = len(completed_games[
                        ((completed_games['home_team'] == selected_team) & (completed_games['home_score'] > completed_games['away_score'])) |
                        ((completed_games['away_team'] == selected_team) & (completed_games['away_score'] > completed_games['home_score']))
                    ])
                    
                    losses = len(completed_games[
                        ((completed_games['home_team'] == selected_team) & (completed_games['home_score'] < completed_games['away_score'])) |
                        ((completed_games['away_team'] == selected_team) & (completed_games['away_score'] < completed_games['home_score']))
                    ])
                    
                    ties = len(completed_games[
                        completed_games['home_score'] == completed_games['away_score']
                    ])
                    
                    record_text = f"{wins}W-{losses}L-{ties}T"
                else:
                    # Fallback to NPI data if games data not available
                    if npi_df is not None and not npi_df.empty:
                        team_record = npi_df[npi_df['team'] == selected_team].iloc[0]
                        record_text = f"{team_record['wins']}W-{team_record['losses']}L-{team_record['ties']}T"
                    else:
                        record_text = "No data available"
                
                st.metric(
                    "Division 3 Record",
                    record_text
                )
        
        # Display team schedule
        if selected_team and games_df is not None:
            schedule_df = get_team_schedule(selected_team, games_df, year=selected_year)
            
            if schedule_df is not None and not schedule_df.empty:
                st.subheader(f"📅 {selected_team} Schedule")
                
                
                # Debug: Show what's in the schedule DataFrame
                print(f"DEBUG SCHEDULE DISPLAY: Schedule DataFrame shape: {schedule_df.shape}")
                print(f"DEBUG SCHEDULE DISPLAY: Schedule DataFrame columns: {list(schedule_df.columns)}")
                if 'NPI Win Value' in schedule_df.columns:
                    print(f"DEBUG SCHEDULE DISPLAY: NPI Win Value values: {schedule_df['NPI Win Value'].tolist()}")
                if 'NPI Loss Value' in schedule_df.columns:
                    print(f"DEBUG SCHEDULE DISPLAY: NPI Loss Value values: {schedule_df['NPI Loss Value'].tolist()}")
                if 'NPI Tie Value' in schedule_df.columns:
                    print(f"DEBUG SCHEDULE DISPLAY: NPI Tie Value values: {schedule_df['NPI Tie Value'].tolist()}")
                
                # Format date for display
                schedule_display = schedule_df.copy()
                schedule_display['Date'] = schedule_display['Date'].dt.strftime('%m/%d/%Y')
                
                # Clean up display columns for better readability
                display_columns = ['Date', 'Opponent', 'Home/Away', 'Team Score', 'Opponent Score', 'Result', 'NPI Win Value', 'NPI Tie Value', 'NPI Loss Value']
                schedule_display = schedule_display[display_columns]
                
                # Format NPI values to 2 decimal places and handle missing data
                if 'NPI Win Value' in schedule_display.columns:
                    # Convert to numeric, replacing non-numeric values with NaN
                    schedule_display['NPI Win Value'] = pd.to_numeric(schedule_display['NPI Win Value'], errors='coerce')
                    # Round numeric values to 2 decimal places
                    schedule_display['NPI Win Value'] = schedule_display['NPI Win Value'].round(2)
                    # Replace NaN values with "N/A" for display
                    schedule_display['NPI Win Value'] = schedule_display['NPI Win Value'].fillna('N/A')
                if 'NPI Tie Value' in schedule_display.columns:
                    # Convert to numeric, replacing non-numeric values with NaN
                    schedule_display['NPI Tie Value'] = pd.to_numeric(schedule_display['NPI Tie Value'], errors='coerce')
                    # Round numeric values to 2 decimal places
                    schedule_display['NPI Tie Value'] = schedule_display['NPI Tie Value'].round(2)
                    # Replace NaN values with "N/A" for display
                    schedule_display['NPI Tie Value'] = schedule_display['NPI Tie Value'].fillna('N/A')
                if 'NPI Loss Value' in schedule_display.columns:
                    # Convert to numeric, replacing non-numeric values with NaN
                    schedule_display['NPI Loss Value'] = pd.to_numeric(schedule_display['NPI Loss Value'], errors='coerce')
                    # Round numeric values to 2 decimal places
                    schedule_display['NPI Loss Value'] = schedule_display['NPI Loss Value'].round(2)
                    # Replace NaN values with "N/A" for display
                    schedule_display['NPI Loss Value'] = schedule_display['NPI Loss Value'].fillna('N/A')
                
                # Check if all NPI values are N/A and show a note
                if ('NPI Win Value' in schedule_display.columns and 
                    'NPI Tie Value' in schedule_display.columns and
                    'NPI Loss Value' in schedule_display.columns):
                    all_npi_na = ((schedule_display['NPI Win Value'] == 'N/A').all() and 
                                 (schedule_display['NPI Tie Value'] == 'N/A').all() and
                                 (schedule_display['NPI Loss Value'] == 'N/A').all())
                    if all_npi_na:
                        st.info("ℹ️ NPI values are not available because NPI ratings data is not available for this season. This is normal for seasons with only scheduled games.")
                
                # Display schedule
                st.dataframe(schedule_display, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No schedule data found for {selected_team}")
    
    # Tab 3: NPI Projections
    with tab3:
        st.subheader(f"NPI Projections")
        st.markdown(f"See how different game outcomes for the **{selected_year} season** will affect NPI rankings.")
        
        # Check if we have the required data
        if games_df is None:
            st.warning("⚠️ NPI projections require games data. Please ensure the filtered games file exists.")
        elif npi_df is None or npi_df.empty:
            st.warning("⚠️ NPI projections require NPI ratings data. Please run NPI calculations first.")
            st.markdown("""
            <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>💡 **For 2025 season:** Run NPI calculator with `--simulation` flag to create simulation NPI data.</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Team selection for simulation
            st.write("**Step 1: Select a team to edit**")
            
            # Get list of teams that have NPI ratings
            if npi_df is not None and not npi_df.empty:
                sim_teams = sorted(npi_df['team'].tolist())
                
                # Try to find University of Wisconsin-La Crosse as default, otherwise use first team
                default_sim_index = 0
                target_team = "University of Wisconsin-La Crosse"
                if target_team in sim_teams:
                    default_sim_index = sim_teams.index(target_team)
                
                # Track previous team selection to clear session state when switching
                if 'previous_sim_team' not in st.session_state:
                    st.session_state['previous_sim_team'] = None
                
                # Debug: Show current session state before team selection
                print(f"DEBUG: Current session state keys: {list(st.session_state.keys())}")
                if 'previous_sim_team' in st.session_state:
                    print(f"DEBUG: Previous team was: {st.session_state['previous_sim_team']}")
                else:
                    print(f"DEBUG: No previous team recorded")
                
                selected_sim_team = st.selectbox(
                    f"Choose a team ({selected_year} season):",
                    options=sim_teams,
                    index=default_sim_index,
                    key="sim_team_select"
                )
                
                print(f"DEBUG: Selected team is: {selected_sim_team}")
                
                # Clear session state when switching teams
                if (st.session_state['previous_sim_team'] is not None and 
                    st.session_state['previous_sim_team'] != selected_sim_team):
                    
                    print(f"DEBUG: TEAM SWITCH DETECTED!")
                    print(f"DEBUG: Previous: {st.session_state['previous_sim_team']}")
                    print(f"DEBUG: Current: {selected_sim_team}")
                    
                    # Clear all session state for the previous team
                    previous_team = st.session_state['previous_sim_team']
                    previous_session_key = f"sim_data_{st.session_state['user_id']}_{selected_year}_{previous_team}"
                    
                    print(f"DEBUG: Looking for session key: {previous_session_key}")
                    print(f"DEBUG: This key exists in session state: {previous_session_key in st.session_state}")
                    
                    if previous_session_key in st.session_state:
                        old_value = st.session_state[previous_session_key]
                        del st.session_state[previous_session_key]
                        print(f"DEBUG: ✅ Cleared session state for previous team: {previous_team}")
                        print(f"DEBUG: Old value was: {old_value}")
                    else:
                        print(f"DEBUG: ❌ Session key not found: {previous_session_key}")
                    
                    # Also clear any other team-specific session state
                    cleared_keys = []
                    for key in list(st.session_state.keys()):
                        if (key.startswith(f"sim_data_{st.session_state['user_id']}_{selected_year}_") and 
                            key != f"sim_data_{st.session_state['user_id']}_{selected_year}_{selected_sim_team}"):
                            old_value = st.session_state[key]
                            del st.session_state[key]
                            cleared_keys.append(key)
                            print(f"DEBUG: ✅ Cleared additional session state: {key} (was: {old_value})")
                    
                    if not cleared_keys:
                        print(f"DEBUG: No additional session state keys found to clear")
                    
                    # Clear any cached team data to force fresh loading
                    team_cache_key = f"team_games_{st.session_state['user_id']}_{selected_year}_{previous_team}"
                    if team_cache_key in st.session_state:
                        del st.session_state[team_cache_key]
                        print(f"DEBUG: ✅ Cleared team games cache for previous team: {previous_team}")
                    else:
                        print(f"DEBUG: ❌ Team games cache key not found: {team_cache_key}")
                    
                    print(f"DEBUG: Switched from {previous_team} to {selected_sim_team} - cleared old session state and cache")
                    print(f"DEBUG: Session state after clearing: {list(st.session_state.keys())}")
                else:
                    print(f"DEBUG: No team switch detected")
                    if st.session_state['previous_sim_team'] is not None:
                        print(f"DEBUG: Same team selected: {st.session_state['previous_sim_team']}")
                    else:
                        print(f"DEBUG: First time selecting a team")
                
                # Update the previous team tracker
                st.session_state['previous_sim_team'] = selected_sim_team
                print(f"DEBUG: Updated previous_sim_team to: {selected_sim_team}")
            else:
                st.error(f"No NPI data available for {selected_year} season")
                st.stop()
            
            if selected_sim_team:
                # Show current team info
                st.write("---")
                st.write("**Step 2: Current Team Status**")
                
                # Define data directory for later use
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                data_dir = project_root / "data" / str(selected_year)
                
                # Get current NPI data
                team_data = npi_df[npi_df['team'] == selected_sim_team].iloc[0]
                current_rank = npi_df[npi_df['team'] == selected_sim_team].index[0] + 1
                current_npi = team_data['npi_rating']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Current Rank", f"#{current_rank}")
                with col2:
                    st.metric("Current NPI", f"{current_npi:.2f}")
                with col3:
                    st.metric("Wins", team_data['wins'])
                with col4:
                    st.metric("Losses", team_data['losses'])
                with col5:
                    st.metric("Ties", team_data['ties'])
                
                # Get team's games for simulation
                st.write("---")
                st.write("**Step 3 (Optional): Apply Projected Scores to Scheduled [Unplayed] Games**")
                
                # Check if predicted scores data exists
                pred_games_file = data_dir / f"massey_pred_games_{selected_year}_filtered_complete.csv"
                pred_scores_available = pred_games_file.exists()
                
                if pred_scores_available:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🎯 Apply Projected Scores", type="primary", use_container_width=True):
                            with st.spinner("Applying predicted scores..."):
                                success, message = apply_predicted_scores(selected_year, selected_sim_team, st.session_state['user_id'])
                            
                            if success:
                                # Clear the session state for this team to force re-initialization with predicted results
                                session_key = f"sim_data_{st.session_state['user_id']}_{selected_year}_{selected_sim_team}"
                                if session_key in st.session_state:
                                    del st.session_state[session_key]
                                
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to apply projected scores: {message}")
                else:
                    st.info("📊 No predicted scores data available for this season.")
                
                st.write("---")
                st.write("**Step 4: Modify Game Results**")
                
                # Force reload simulation data to ensure we have the latest updates
                print(f"DEBUG: Force reloading simulation data for {selected_sim_team}")
                
                # Clear any potential file system cache by reloading the simulation file
                script_dir = Path(__file__).resolve().parent
                project_root = script_dir.parent
                data_dir = project_root / "data" / str(selected_year)
                sim_file = data_dir / f"massey_games_{selected_year}_simulation_{st.session_state['user_id']}.csv"
                
                if sim_file.exists():
                    # Force reload by reading the file again
                    current_time = pd.Timestamp.now()
                    print(f"DEBUG: Reloading simulation file at {current_time}: {sim_file}")
                    
                    # Read the simulation data directly
                    sim_df = pd.read_csv(sim_file)
                    print(f"DEBUG: Reloaded simulation data: {len(sim_df)} total games")
                    
                    # Filter for the selected team
                    raw_team_games = sim_df[
                        (sim_df['home_team'] == selected_sim_team) | 
                        (sim_df['away_team'] == selected_sim_team)
                    ].copy()
                    
                    if not raw_team_games.empty:
                        print(f"DEBUG: Found {len(raw_team_games)} games for {selected_sim_team} in reloaded data")
                        print(f"DEBUG: Sample game data: {raw_team_games.iloc[0].to_dict()}")
                        
                        # Convert to the same format that get_team_schedule returns
                        schedule_data = []
                        for _, game in raw_team_games.iterrows():
                            if game['home_team'] == selected_sim_team:
                                # Team is playing at home
                                opponent = game['away_team']
                                home_away = "HOME"
                                team_score = game['home_score']
                                opp_score = game['away_score']
                            else:
                                # Team is playing away
                                opponent = game['home_team']
                                home_away = "AWAY"
                                team_score = game['away_score']
                                opp_score = game['home_score']
                            
                            # Check if game is scheduled
                            is_scheduled = pd.notna(game['game_note']) and str(game['game_note']).strip() == 'Sch'
                            
                            # Determine result (only for completed games)
                            if is_scheduled:
                                result = "SCH"  # Scheduled
                            else:
                                if team_score > opp_score:
                                    result = "W"
                                elif team_score < opp_score:
                                    result = "L"
                                else:
                                    result = "T"
                            
                            # Check for overtime
                            overtime = ""
                            if not is_scheduled and pd.notna(game['game_note']) and str(game['game_note']).strip() != "":
                                if 'o1' in str(game['game_note']).lower() or 'o2' in str(game['game_note']).lower() or 'o3' in str(game['game_note']).lower():
                                    overtime = "OT"
                            
                            # Check for predicted scores
                            has_pred_note = pd.notna(game['game_note']) and 'pred' in str(game['game_note']).lower()
                            predicted_result = game.get('predicted_result', None)
                            is_predicted = has_pred_note and predicted_result is not None
                            
                            schedule_data.append({
                                'Date': game['date'],
                                'Opponent': opponent,
                                'Home/Away': home_away,
                                'Team Score': team_score,
                                'Opponent Score': opp_score,
                                'Result': result,
                                'Overtime': overtime,
                                'Status': 'Scheduled' if is_scheduled else 'Completed',
                                'Predicted': is_predicted,
                                'Predicted_Result': predicted_result
                            })
                        
                        # Convert to DataFrame and sort by date
                        team_games = pd.DataFrame(schedule_data)
                        team_games['Date'] = pd.to_datetime(team_games['Date'])
                        team_games = team_games.sort_values('Date').reset_index(drop=True)
                        
                        print(f"DEBUG: Converted to schedule format: {len(team_games)} games")
                    else:
                        print(f"DEBUG: No games found for {selected_sim_team} in reloaded data")
                        team_games = None
                else:
                    print(f"DEBUG: Simulation file not found, using original games data")
                    team_games = get_team_schedule(selected_sim_team, games_df, st.session_state['user_id'], selected_year)
                
                # Force refresh of team_games to ensure we have the latest data
                if team_games is not None and not team_games.empty:
                    print(f"DEBUG: Final loaded {len(team_games)} games for {selected_sim_team}")
                    print(f"DEBUG: First game data: {team_games.iloc[0].to_dict()}")
                else:
                    print(f"DEBUG: No games loaded for {selected_sim_team}")
                
                if team_games is not None and not team_games.empty:
                    
                    # Create editable game results using session state
                    sim_data = []
                    
                    # Initialize session state for this team if not exists
                    session_key = f"sim_data_{st.session_state['user_id']}_{selected_year}_{selected_sim_team}"
                    if session_key not in st.session_state:
                        st.session_state[session_key] = {}
                    
                    for idx, game in team_games.iterrows():
                        st.write("---")
                        
                        # Game header
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            header_text = f"**{game['Opponent']}** ({game['Home/Away']}) - {game['Date']}"
                            st.write(header_text)
                        with col2:
                            if game['Overtime']:
                                st.write(f"⚽ {game['Overtime']}")
                        

                        
                        # Game details
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            # Show original result
                            original_result = game['Result']
                            if original_result == 'SCH':
                                st.write("**Scheduled Game**")
                            else:
                                st.write(f"**Current Result:** {original_result}")
                        
                        with col2:
                            # Show original score if available
                            if original_result != 'SCH':
                                st.write(f"Score: {game['Team Score']}-{game['Opponent Score']}")
                            else:
                                st.write("Score: TBD")
                        
                        with col3:
                            # Simulation selector with session state
                            game_key = f"game_{idx}"
                            
                            # ALWAYS initialize with current data from CSV, never use old session state
                            # This ensures dropdowns always show the correct current values
                            
                            # Determine what the dropdown should show based on current CSV data
                            if game['Predicted'] and game['Predicted_Result']:
                                # Game has predicted scores applied - set to PRED to indicate it's using predicted scores
                                dropdown_default = "PRED"
                                print(f"DEBUG: Game {idx} - Game has predicted scores, setting dropdown to PRED")
                            else:
                                # Game has no prediction - use current result
                                dropdown_default = original_result
                                print(f"DEBUG: Game {idx} - Using current result as default: {dropdown_default}")
                            
                            # Force update session state with current data
                            st.session_state[session_key][game_key] = dropdown_default
                            print(f"DEBUG: Game {idx} - Forced session state update: {game_key} = {dropdown_default}")
                            
                            # Debug: Show what's in session state after update
                            print(f"DEBUG: Game {idx} - Session state for {game_key}: {st.session_state[session_key].get(game_key, 'NOT SET')}")
                            print(f"DEBUG: Game {idx} - Original result: {original_result}")
                            print(f"DEBUG: Game {idx} - Dropdown default: {dropdown_default}")
                            
                            if original_result == 'SCH':
                                sim_result = st.selectbox(
                                    "Simulate Result:",
                                    options=["SCH", "W", "L", "T", "PRED"],
                                    key=f"sim_{selected_year}_{selected_sim_team}_{idx}",
                                    index=["SCH", "W", "L", "T", "PRED"].index(st.session_state[session_key][game_key]) if st.session_state[session_key][game_key] in ["SCH", "W", "L", "T", "PRED"] else 0,
                                    label_visibility="collapsed"
                                )
                            else:
                                # Simplified index calculation to avoid confusion
                                current_value = st.session_state[session_key].get(game_key, original_result)
                                try:
                                    dropdown_index = ["SCH", "W", "L", "T", "PRED"].index(current_value)
                                except ValueError:
                                    dropdown_index = 0
                                    print(f"WARNING: Invalid value '{current_value}' for game {idx}, defaulting to index 0")
                                
                                sim_result = st.selectbox(
                                    "Change to:",
                                    options=["SCH", "W", "L", "T", "PRED"],
                                    key=f"change_{selected_year}_{selected_sim_team}_{idx}",
                                    index=dropdown_index,
                                    label_visibility="collapsed"
                                )
                            
                            # Debug: Show what was selected
                            print(f"DEBUG: Game {idx} - Dropdown selected: {sim_result}")
                            
                            # Update session state
                            st.session_state[session_key][game_key] = sim_result
                            print(f"DEBUG: Updated session state: {game_key} = {sim_result}")
                            
                            # Verify the update worked
                            actual_value = st.session_state[session_key].get(game_key)
                            print(f"DEBUG: Verification - session state now contains: {game_key} = {actual_value}")
                            
                                                    # No caption text below dropdowns
                        
                        with col4:
                            # Show change indicator - compare against the true original result (predicted if available)
                            original_game_idx = team_games[team_games['Date'] == game['Date']].index
                            if len(original_game_idx) > 0:
                                original_game = team_games.loc[original_game_idx[0]]
                                # CRITICAL FIX: Use same logic as simulation data collection
                                if original_game['Predicted'] and original_game['Predicted_Result']:
                                    # Check if predicted scores were actually applied
                                    has_pred_note = pd.notna(original_game.get('game_note')) and 'pred' in str(original_game.get('game_note')).lower()
                                    if has_pred_note:
                                        true_original_result = original_game['Predicted_Result']
                                    else:
                                        true_original_result = original_game['Result']
                                else:
                                    true_original_result = original_game['Result']
                            else:
                                true_original_result = original_result
                            
                            if sim_result == "PRED":
                                st.info("🎯 Using Projected Scores")
                            elif true_original_result == 'SCH' or sim_result != true_original_result:
                                if sim_result == "W":
                                    st.success("🟢 Win")
                                elif sim_result == "L":
                                    st.error("🔴 Loss")
                                elif sim_result == "T":
                                    st.info("🟡 Tie")
                                elif sim_result == "SCH":
                                    st.info("📅 Scheduled")
                            else:
                                st.write("✅ No change")
                        
                        # Store simulation data - use original team_games to get all games, not filtered view
                        # Find the corresponding game in the full team_games dataset
                        original_game_idx = team_games[team_games['Date'] == game['Date']].index
                        if len(original_game_idx) > 0:
                            original_game = team_games.loc[original_game_idx[0]]
                            
                            # Determine the true original result - if it's a predicted game, use the predicted result as the "original"
                            print(f"DEBUG: Game {original_game['Opponent']} - Raw data:")
                            print(f"  - Predicted: {original_game['Predicted']}")
                            print(f"  - Predicted_Result: {original_game['Predicted_Result']}")
                            print(f"  - Result: {original_game['Result']}")
                            print(f"  - game_note: {original_game.get('game_note', 'N/A')}")
                            
                            # CRITICAL FIX: After applying predicted scores, the predicted result becomes the new baseline
                            # We need to check if this game has been modified by predicted scores
                            if original_game['Predicted'] and original_game['Predicted_Result']:
                                # Check if the game_note contains 'pred' - this means predicted scores were applied
                                has_pred_note = pd.notna(original_game.get('game_note')) and 'pred' in str(original_game.get('game_note')).lower()
                                if has_pred_note:
                                    # Predicted scores were applied, so use predicted result as baseline
                                    true_original_result = original_game['Predicted_Result']
                                    print(f"  - Using Predicted_Result as baseline (pred applied): {true_original_result}")
                                else:
                                    # Predicted scores not yet applied, use original result
                                    true_original_result = original_game['Result']
                                    print(f"  - Using Result as baseline (pred not applied): {true_original_result}")
                            else:
                                true_original_result = original_game['Result']
                                print(f"  - Using Result as baseline (no prediction): {true_original_result}")
                            
                            # A change is detected if the simulated result differs from the true original result
                            # BUT: PRED results should never be considered "changed" - they're using predicted scores
                            if sim_result == "PRED":
                                is_changed = False
                                print(f"DEBUG: Game {idx} - PRED selected, marking as unchanged (using predicted scores)")
                            else:
                                is_changed = sim_result != true_original_result
                                print(f"DEBUG: Game {idx} - Change detection: {sim_result} != {true_original_result} = {is_changed}")
                            
                            print(f"DEBUG: Game {original_game['Opponent']} - True Original: {true_original_result}, Simulated: {sim_result}, Changed: {is_changed}")
                            print(f"DEBUG: Session state value for {game_key}: {st.session_state[session_key].get(game_key, 'NOT SET')}")
                            print(f"DEBUG: Dropdown result: {sim_result}")
                            print(f"DEBUG: Change detection: {sim_result} != {true_original_result} = {is_changed}")
                            
                            sim_data.append({
                                'opponent': original_game['Opponent'],
                                'home_away': original_game['Home/Away'],
                                'original_result': true_original_result,
                                'simulated_result': sim_result,
                                'changed': is_changed,
                                'date': original_game['Date']
                            })
                    
                    # Simulation controls
                    st.write("---")
                    st.write("**Step 5: Save Data and Run NPI Calculations**")
                    
                    # Helpful reminder for users
                    st.markdown("""
            <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>💡 **Important:** Make sure to save the data before running the NPI calculations!</strong>
            </div>
            """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                    
                    with col1:
                        pass
                    
                    with col2:
                        if st.button("🔄 Reset All Changes", type="secondary"):
                            # Clear session state for this team
                            if session_key in st.session_state:
                                del st.session_state[session_key]
                            
                            # Reset simulation data by copying filtered data back
                            with st.spinner("Resetting data..."):
                                reset_success, reset_message = reset_simulation_data(selected_year, st.session_state['user_id'])
                            
                            if reset_success:
                                st.success("✅ Data reset successfully")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reset data. Please try again.")
                        
                    with col3:
                        if st.button("💾 Save Data", type="secondary"):
                            # Apply changes to simulation data without running NPI calculation
                            changes_made = [game for game in sim_data if game['changed']]
                            
                            if changes_made:
                                
                                with st.spinner("Saving data..."):
                                    apply_success, apply_message = apply_simulation_changes(selected_year, selected_sim_team, sim_data, st.session_state['user_id'])
                                
                                if apply_success:
                                    st.success(f"✅ {apply_message}")
                                    
                                    # Show what was saved
                                    st.subheader("Changes Saved:")
                                    for change in changes_made:
                                        if change['original_result'] == 'SCH':
                                            st.write(f"• **{change['opponent']}** ({change['date']}): Scheduled → {change['simulated_result']}")
                                        else:
                                            st.write(f"• **{change['opponent']}** ({change['date']}): {change['original_result']} → {change['simulated_result']}")
                                else:
                                    st.error(f"❌ Failed to save data: {apply_message}")
                            else:
                                st.markdown("""
                                <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                    <strong>No changes to save.</strong>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with col4:
                        if st.button("📊 Run NPI Calculations", type="secondary"):
                            # Run NPI calculator directly
                            
                            with st.spinner("Calculating NPI ratings..."):
                                success, output, error = run_npi_simulation(selected_year, st.session_state['user_id'])
                            
                            if success:
                                st.success("✅ NPI calculations completed successfully!")
                            else:
                                st.error("❌ NPI calculation failed!")
                                st.error(f"Error: {error}")
                                st.markdown("""
                                <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                    <strong>💡 **Manual Steps:**</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                st.write(f"1. Run NPI calculator manually: `python npi_calculator.py --year {selected_year} --season-only --simulation --user-id {st.session_state['user_id']}`")
                                st.write("2. Check for any error messages")
                                st.write("3. Verify the simulation data file is correct")
                    
                    with col5:
                        pass
                
                else:
                    st.warning(f"No games found for {selected_sim_team}")
                    st.markdown("""
                    <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <strong>💡 **Tip:** Select a different team or check if the team has any games in the current season.</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Step 5: View Current Simulation NPI Ratings
                st.write("---")
                st.write("**Step 6: View Projected NPI Rankings**")
                
                # Check if user-specific simulation NPI data exists
                sim_npi_file = data_dir / f"npi_ratings_{selected_year}_simulation_{st.session_state['user_id']}.csv"
                
                if sim_npi_file.exists():
                    try:
                        # Load simulation NPI data
                        sim_npi_df = pd.read_csv(sim_npi_file)
                        
                        if not sim_npi_df.empty:
                            
                            
                            # Sort by NPI rating (highest first)
                            df_sorted = sim_npi_df.sort_values('npi_rating', ascending=False).reset_index(drop=True)
                            
                            # Add ranking
                            df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
                            
                            # Select only the columns we want to display
                            columns_to_show = ['Rank', 'team', 'npi_rating', 'wins', 'losses', 'ties']
                            df_display = df_sorted[columns_to_show].copy()
                            
                            # Round NPI rating to 2 decimal places
                            df_display['npi_rating'] = df_display['npi_rating'].round(2)
                            
                            # Rename columns for cleaner display
                            df_display = df_display.rename(columns={
                                'team': 'Team',
                                'npi_rating': 'NPI Rating',
                                'wins': 'Win',
                                'losses': 'Loss', 
                                'ties': 'Tie'
                            })
                            
                            # Add margin of error note
                            st.markdown("*Note: These projected NPI ratings have a margin of error of ±0.3*")
                            
                            # Display simulation NPI data
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                            
                            # Add important note about NPI rating dependencies
                            st.error("**Important Note:** NPI ratings are largely dependent on other teams and their results. Projected results may not accurately reflect final rankings.")
                            
                            # Show comparison with original data if available (moved below the table)
                            if npi_df is not None and not npi_df.empty:
                                
                                # Find the selected team in both datasets
                                if selected_sim_team in npi_df['team'].values and selected_sim_team in sim_npi_df['team'].values:
                                    original_team = npi_df[npi_df['team'] == selected_sim_team].iloc[0]
                                    simulated_team = sim_npi_df[sim_npi_df['team'] == selected_sim_team].iloc[0]
                                    
                                    # Get original and new rankings
                                    original_rank = npi_df[npi_df['team'] == selected_sim_team].index[0] + 1
                                    new_rank = sim_npi_df[sim_npi_df['team'] == selected_sim_team].index[0] + 1
                                    rank_change = original_rank - new_rank
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Original NPI", f"{original_team['npi_rating']:.2f}")
                                    with col2:
                                        change = simulated_team['npi_rating'] - original_team['npi_rating']
                                        # More robust check for zero or near-zero changes
                                        if abs(change) < 0.01:  # Increased threshold slightly
                                            delta_display = None
                                        else:
                                            delta_display = f"{change:+.2f}"
                                        st.metric("Projected NPI", f"{simulated_team['npi_rating']:.2f}", delta=delta_display)
                                    with col3:
                                        # More robust check for zero or near-zero changes
                                        if abs(rank_change) < 0.5:
                                            delta_display = None
                                        else:
                                            delta_display = f"{rank_change:+d}"
                                        st.metric("New Rank", f"#{new_rank}", delta=delta_display)
                                
                        else:
                            st.warning("📊 Simulation NPI file exists but is empty")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading simulation NPI data: {e}")
                        
                else:
                    st.markdown(f"""
                    <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <strong>📊 **No projected NPI ratings available yet**</strong><br>
                    </div>
                    """, unsafe_allow_html=True)
                

    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created by Kevin Puorro • Data from Massey Ratings*")

if __name__ == "__main__":
    main()
