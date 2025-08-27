#!/usr/bin/env python3
"""
Filter Massey Games Script
Takes scraped Massey data and filters to only include games where both teams
exist in the team_mapping.csv file.
"""

import os
import pandas as pd
import argparse
from typing import List, Dict, Tuple


def load_team_mapping(mapping_file: str = "team_mapping.txt") -> Tuple[Dict[str, str], Dict[str, bool]]:
    try:
        # Try to find the mapping file in various locations
        # Script is in scripts/, team_mapping.csv is in project root
        possible_paths = [
            os.path.join("..", mapping_file),  # scripts/ → project root (where team_mapping.csv is)
            mapping_file,  # Current directory (fallback)
            os.path.join("..", "..", mapping_file)  # Two levels up (extra fallback)
        ]
        
        mapping_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mapping_path = path
                break
        
        if not mapping_path:
            raise FileNotFoundError(f"Team mapping file not found. Tried: {possible_paths}")
        
        print(f"Loading team mapping from: {mapping_path}")
        # Load TXT file with tab separators
        try:
            mapping_df = pd.read_csv(mapping_path, sep='\t')
        except Exception as e:
            print(f"⚠️  Tab-separated parsing failed: {e}")
            # Fallback to comma-separated if needed
            try:
                mapping_df = pd.read_csv(mapping_path)
                print("⚠️  Loaded as comma-separated instead")
            except Exception as e2:
                print(f"❌ Both parsing methods failed: {e2}")
                return {}, {}
        
        # Create mapping dictionary
        team_mapping = dict(zip(mapping_df['massey_name'], mapping_df['ncaa_name']))
        
        # Create provisional membership dictionary
        provisional_mapping = {}
        if 'provisional_membership' in mapping_df.columns:
            provisional_mapping = dict(zip(mapping_df['massey_name'], 
                                         mapping_df['provisional_membership'].fillna('').str.contains('p', case=False, na=False)))
        else:
            print("⚠️  No provisional_membership column found in team mapping")
            provisional_mapping = {team: False for team in team_mapping.keys()}
        
        print(f"Loaded {len(team_mapping)} team mappings")
        print(f"Found {sum(provisional_mapping.values())} provisional teams")
        
        return team_mapping, provisional_mapping
        
    except Exception as e:
        print(f"Error loading team mapping: {e}")
        return {}


def load_massey_games(data_dir: str, year: int, use_pred_data: bool = False) -> pd.DataFrame:
    """
    Load Massey games data for the specified year
    
    Args:
        data_dir: Directory containing the data
        year: Year to load data for
        use_pred_data: If True, load prediction data (massey_pred_games)
        
    Returns:
        DataFrame with Massey games data
    """
    try:
        # Try different possible filenames
        if use_pred_data:
            possible_files = [
                f"massey_pred_games_{year}.csv",
                f"massey_pred_games_{year}_clean.csv",
                f"massey_pred_games_{year}_clean_final.csv"
            ]
        else:
            possible_files = [
                f"massey_games_{year}.csv",
                f"massey_games_{year}_clean.csv",
                f"massey_games_{year}_clean_final.csv"
            ]
        
        massey_file = None
        for filename in possible_files:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                massey_file = file_path
                break
        
        if not massey_file:
            raise FileNotFoundError(f"No Massey games file found for {year}")
        
        print(f"Loading Massey games from: {massey_file}")
        massey_df = pd.read_csv(massey_file)
        
        # Check if game_note column exists, if not add it as empty
        if 'game_note' not in massey_df.columns:
            # Check if overtime column exists and rename it
            if 'overtime' in massey_df.columns:
                print("🔄 Renaming 'overtime' column to 'game_note'")
                massey_df = massey_df.rename(columns={'overtime': 'game_note'})
            else:
                print("⚠️  No game_note column found, adding empty column")
                massey_df['game_note'] = ""
        
        print(f"Loaded {len(massey_df)} games from {massey_file}")
        print(f"Columns: {list(massey_df.columns)}")
        
        return massey_df
        
    except Exception as e:
        print(f"Error loading Massey games: {e}")
        return pd.DataFrame()


def filter_games_by_mapping(games_df: pd.DataFrame, team_mapping: Dict[str, str], provisional_mapping: Dict[str, bool]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Filter games to create four outputs: filtered games, excluded games, provisional games, and NPI games
    
    Args:
        games_df: DataFrame with games data
        team_mapping: Dictionary mapping Massey names to NCAA names
        provisional_mapping: Dictionary mapping Massey names to provisional status
        
    Returns:
        Tuple of (filtered DataFrame, provisional DataFrame, excluded DataFrame, NPI DataFrame, list of excluded games info)
    """
    if games_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    
    # Create a copy to avoid modifying original
    filtered_df = games_df.copy()
    
    # Track different types of games
    valid_games = []
    provisional_games = []
    excluded_games = []
    npi_games = []
    
    # Filter games where both teams exist in mapping
    for idx, game in filtered_df.iterrows():
        home_team = str(game['home_team']).strip()
        away_team = str(game['away_team']).strip()
        
        # Check if both teams exist in mapping
        if home_team in team_mapping and away_team in team_mapping:
            # Check if either team is provisional
            home_provisional = provisional_mapping.get(home_team, False)
            away_provisional = provisional_mapping.get(away_team, False)
            
            if home_provisional or away_provisional:
                # This is a provisional game - add 'p' to game_note
                game_copy = game.copy()
                if pd.isna(game_copy['game_note']) or str(game_copy['game_note']).strip() == "":
                    game_copy['game_note'] = 'p'
                else:
                    game_copy['game_note'] = str(game_copy['game_note']).strip() + ', p'
                provisional_games.append(game_copy)
            else:
                # This is a regular valid game
                valid_games.append(game)
                
                # Check if this game should be included in NPI games (exclude scheduled games)
                game_note = str(game.get('game_note', '')).strip()
                if game_note != 'Sch':
                    # Include in NPI games (can have overtime but not scheduled)
                    npi_games.append(game)
        else:
            # Track excluded game
            missing_teams = []
            if home_team not in team_mapping:
                missing_teams.append(f"home: {home_team}")
            if away_team not in team_mapping:
                missing_teams.append(f"away: {away_team}")
            
            excluded_games.append({
                'date': game['date'],
                'game': f"{away_team} {game['away_score']} @ {home_team} {game['home_score']}",
                'missing_teams': ', '.join(missing_teams)
            })
    
    # Create DataFrames
    filtered_df = pd.DataFrame(valid_games)
    provisional_df = pd.DataFrame(provisional_games)
    excluded_df = pd.DataFrame(excluded_games)
    npi_df = pd.DataFrame(npi_games)
    
    return filtered_df, provisional_df, excluded_df, npi_df, excluded_games


def convert_team_names(games_df: pd.DataFrame, team_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Convert Massey team names to NCAA team names using the mapping
    
    Args:
        games_df: DataFrame with games data
        team_mapping: Dictionary mapping Massey names to NCAA names
        
    Returns:
        DataFrame with converted team names
    """
    if games_df.empty:
        return games_df
    
    # Create a copy to avoid modifying original
    converted_df = games_df.copy()
    
    # Convert team names
    converted_df['home_team'] = converted_df['home_team'].apply(
        lambda x: team_mapping.get(str(x).strip(), str(x).strip())
    )
    converted_df['away_team'] = converted_df['away_team'].apply(
        lambda x: team_mapping.get(str(x).strip(), str(x).strip())
    )
    
    return converted_df


def save_filtered_games(games_df: pd.DataFrame, output_dir: str, year: int, suffix: str = "filtered", filename_prefix: str = "massey_games_") -> str:
    """
    Save filtered games to CSV
    
    Args:
        games_df: DataFrame with filtered games
        output_dir: Directory to save the file
        year: Year for the filename
        suffix: Suffix to add to filename
        filename_prefix: Prefix for the filename (default: "massey_games_")
        
    Returns:
        Path to saved file
    """
    if games_df.empty:
        print("No games to save")
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{filename_prefix}{year}_{suffix}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Check if this is prediction data (has prediction columns)
    is_prediction_data = any(col in games_df.columns for col in ['home_pred_score', 'away_pred_score', 'home_win_prob', 'away_win_prob'])
    
    if is_prediction_data:
        # For prediction data, preserve all columns but ensure basic ones are present
        expected_columns = ['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note']
        for col in expected_columns:
            if col not in games_df.columns:
                games_df[col] = ""
        
        # Get all columns, starting with expected ones, then any additional columns
        all_columns = expected_columns + [col for col in games_df.columns if col not in expected_columns]
        games_df = games_df[all_columns]
        
        print(f"📊 Preserving all columns for prediction data")
    else:
        # For regular data, use the standard column structure
        expected_columns = ['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note']
        for col in expected_columns:
            if col not in games_df.columns:
                games_df[col] = ""
        
        # Reorder columns to match expected format
        games_df = games_df[expected_columns]
    
    # Save to CSV
    games_df.to_csv(output_path, index=False)
    print(f"Saved {len(games_df)} filtered games to: {output_path}")
    print(f"Columns saved: {list(games_df.columns)}")
    
    return output_path


def save_excluded_games(excluded_games: List[Dict], output_dir: str, year: int, filename_prefix: str = "massey_games_") -> str:
    """
    Save excluded games to CSV for analysis
    
    Args:
        excluded_games: List of dictionaries with excluded game information
        output_dir: Directory to save the file
        year: Year for the filename
        filename_prefix: Prefix for the filename (default: "massey_games_")
        
    Returns:
        Path to saved file
    """
    if not excluded_games:
        print("No excluded games to save")
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{filename_prefix}{year}_excluded.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Convert to DataFrame and save to CSV
    excluded_df = pd.DataFrame(excluded_games)
    excluded_df.to_csv(output_path, index=False)
    print(f"Saved {len(excluded_games)} excluded games to: {output_path}")
    
    return output_path


def save_provisional_games(provisional_df: pd.DataFrame, output_dir: str, year: int, filename_prefix: str = "massey_games_") -> str:
    """
    Save provisional games to CSV
    
    Args:
        provisional_df: DataFrame with provisional games data
        output_dir: Directory to save the file
        year: Year for the filename
        filename_prefix: Prefix for the filename (default: "massey_games_")
        
    Returns:
        Path to saved file
    """
    if provisional_df.empty:
        print("No provisional games to save")
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{filename_prefix}{year}_provisional.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Ensure all expected columns are present
    expected_columns = ['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note']
    for col in expected_columns:
        if col not in provisional_df.columns:
            provisional_df[col] = ""
    
    # Reorder columns to match expected format
    provisional_df = provisional_df[expected_columns]
    
    # Save to CSV
    provisional_df.to_csv(output_path, index=False)
    print(f"Saved {len(provisional_df)} provisional games to: {output_path}")
    
    return output_path


def save_simulation_games(filtered_df: pd.DataFrame, output_dir: str, year: int, filename_prefix: str = "massey_games_") -> str:
    """
    Save simulation games to CSV (identical to filtered games for simulation purposes)
    
    Args:
        filtered_df: DataFrame with filtered games data
        output_dir: Directory to save the file
        year: Year for the filename
        filename_prefix: Prefix for the filename (default: "massey_games_")
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{filename_prefix}{year}_simulation.csv"
    output_path = os.path.join(output_dir, filename)
    
    if filtered_df.empty:
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note'])
        empty_df.to_csv(output_path, index=False)
        print(f"Created empty simulation games file: {output_path} (0 games)")
        return output_path
    
    # Ensure all expected columns are present
    expected_columns = ['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note']
    for col in expected_columns:
        if col not in filtered_df.columns:
            filtered_df[col] = ""
    
    # Reorder columns to match expected format
    filtered_df = filtered_df[expected_columns]
    
    # Save to CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_df)} simulation games to: {output_path}")
    
    return output_path


def save_npi_games(npi_df: pd.DataFrame, output_dir: str, year: int, filename_prefix: str = "massey_games_") -> str:
    """
    Save NPI games to CSV (excludes scheduled games but includes overtime games)
    
    Args:
        npi_df: DataFrame with NPI games data (no scheduled games)
        output_dir: Directory to save the file
        year: Year for the filename
        filename_prefix: Prefix for the filename (default: "massey_games_")
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{filename_prefix}{year}_npi.csv"
    output_path = os.path.join(output_dir, filename)
    
    if npi_df.empty:
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note'])
        empty_df.to_csv(output_path, index=False)
        print(f"Created empty NPI games file: {output_path} (0 games)")
        return output_path
    
    # Ensure all expected columns are present
    expected_columns = ['date', 'away_team', 'away_score', 'home_team', 'home_score', 'game_note']
    for col in expected_columns:
        if col not in npi_df.columns:
            npi_df[col] = ""
    
    # Reorder columns to match expected format
    npi_df = npi_df[expected_columns]
    
    # Save to CSV
    npi_df.to_csv(output_path, index=False)
    print(f"Saved {len(npi_df)} NPI games to: {output_path}")
    
    return output_path


def main():
    """Main function to filter Massey games"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter Massey games data')
    parser.add_argument('--year', '-y', type=int, default=2025, 
                       help='Season year to process (default: 2025)')
    parser.add_argument('--suffix', '-s', type=str, default='filtered',
                       help='Suffix for output filename (default: filtered)')
    parser.add_argument('--pred', '-p', action='store_true',
                       help='Use prediction data (massey_pred_games) instead of regular data (massey_games)')
    
    args = parser.parse_args()
    
    # Configuration
    year = args.year
    suffix = args.suffix
    use_pred_data = args.pred
    
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define paths
    data_dir = os.path.join(project_root, "data", str(year))
    output_dir = data_dir  # Save filtered data in same directory
    
    print(f"🔍 Filtering Massey games for {year}")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    print("-" * 50)
    
    # Step 1: Load team mapping
    print("1️⃣ Loading team mapping...")
    team_mapping, provisional_mapping = load_team_mapping()
    if not team_mapping:
        print("❌ Failed to load team mapping. Exiting.")
        return
    
    # Step 2: Load Massey games
    print("\n2️⃣ Loading Massey games...")
    massey_df = load_massey_games(data_dir, year, use_pred_data)
    if massey_df.empty:
        print("❌ No Massey games data found. Exiting.")
        return
    
    print(f"📊 Original games: {len(massey_df)}")
    
    # Step 3: Filter games by mapping
    print("\n3️⃣ Filtering games by team mapping...")
    filtered_df, provisional_df, excluded_df, npi_df, excluded_games = filter_games_by_mapping(massey_df, team_mapping, provisional_mapping)
    
    print(f"✅ Valid games: {len(filtered_df)}")
    print(f"📋 Provisional games: {len(provisional_df)}")
    print(f"❌ Excluded games: {len(excluded_df)}")
    print(f"🏆 NPI games: {len(npi_df)}")
    
    # Step 4: Convert team names to NCAA names
    print("\n4️⃣ Converting team names to NCAA names...")
    converted_df = convert_team_names(filtered_df, team_mapping)
    converted_provisional_df = convert_team_names(provisional_df, team_mapping)
    converted_npi_df = convert_team_names(npi_df, team_mapping)
    
    # Determine filename prefix based on data type
    filename_prefix = "massey_pred_games_" if use_pred_data else "massey_games_"
    
    # Step 5: Save filtered data
    print("\n5️⃣ Saving filtered data...")
    output_file = save_filtered_games(converted_df, output_dir, year, suffix, filename_prefix)
    
    # Step 6: Save provisional games
    print("\n6️⃣ Saving provisional games...")
    provisional_output_file = save_provisional_games(converted_provisional_df, output_dir, year, filename_prefix)
    
    # Step 7: Save NPI games
    print("\n7️⃣ Saving NPI games...")
    npi_output_file = save_npi_games(converted_npi_df, output_dir, year, filename_prefix)
    
    # Step 8: Save simulation games
    print("\n8️⃣ Saving simulation games...")
    simulation_output_file = save_simulation_games(converted_df, output_dir, year, filename_prefix)
    
    # Step 9: Save excluded games
    print("\n9️⃣ Saving excluded games...")
    excluded_output_file = save_excluded_games(excluded_games, output_dir, year, filename_prefix)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 FILTERING SUMMARY")
    print("=" * 50)
    print(f"📊 Original games: {len(massey_df)}")
    print(f"✅ Valid games: {len(filtered_df)}")
    print(f"📋 Provisional games: {len(provisional_df)}")
    print(f"🏆 NPI games: {len(npi_df)}")
    print(f"❌ Excluded games: {len(excluded_df)}")
    print(f"💾 Filtered games file: {output_file}")
    print(f"💾 Provisional games file: {provisional_output_file}")
    print(f"💾 NPI games file: {npi_output_file}")
    print(f"💾 Simulation games file: {simulation_output_file}")
    print(f"💾 Excluded games file: {excluded_output_file}")
    
    # Show some excluded games for reference
    if excluded_games:
        print(f"\n❌ Sample excluded games (showing first 5):")
        for i, game in enumerate(excluded_games[:5]):
            print(f"   {i+1}. {game['date']}: {game['game']} - Missing: {game['missing_teams']}")
        
        if len(excluded_games) > 5:
            print(f"   ... and {len(excluded_games) - 5} more")
    
    print(f"\n🎯 Filtered data is ready for NPI calculation!")
    print(f"📁 File location: {output_file}")
    print(f"📋 Provisional games available at: {provisional_output_file}")
    if len(npi_df) > 0:
        print(f"🏆 NPI games (no scheduled games) available at: {npi_output_file}")
    else:
        print(f"🏆 NPI games file created at: {npi_output_file} (0 games - all games are scheduled)")
    print(f"🎯 Simulation games available at: {simulation_output_file}")


if __name__ == "__main__":
    main()
