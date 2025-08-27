#!/usr/bin/env python3
"""
Check Missing Predictions Script
Identifies games in massey_pred_games_2025_filtered.csv that are missing
prediction columns like home_win_prob, away_win_prob, etc.
"""

import pandas as pd
import os
from pathlib import Path

def check_missing_predictions():
    """Check for missing prediction data in the filtered dataset"""
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "2025"
    
    # Input file
    input_file = data_dir / "massey_pred_games_2025_filtered.csv"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"🔍 Checking for missing predictions in: {input_file}")
    print("=" * 60)
    
    # Load the data
    df = pd.read_csv(input_file)
    print(f"📊 Total games loaded: {len(df)}")
    print(f"📋 Columns found: {list(df.columns)}")
    print()
    
    # Define prediction columns to check
    prediction_columns = ['home_pred_score', 'away_pred_score', 'home_win_prob', 'away_win_prob']
    
    # Check which columns exist
    existing_pred_columns = [col for col in prediction_columns if col in df.columns]
    missing_pred_columns = [col for col in prediction_columns if col not in df.columns]
    
    print("📊 PREDICTION COLUMNS STATUS:")
    print(f"✅ Existing: {existing_pred_columns}")
    if missing_pred_columns:
        print(f"❌ Missing: {missing_pred_columns}")
    else:
        print("✅ All prediction columns present!")
    print()
    
    if not existing_pred_columns:
        print("❌ No prediction columns found in the dataset!")
        return
    
    # Check for missing values in each prediction column
    print("🔍 MISSING VALUES ANALYSIS:")
    print("-" * 40)
    
    total_games = len(df)
    missing_summary = {}
    
    for col in existing_pred_columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_games) * 100
        
        missing_summary[col] = {
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'missing_indices': df[df[col].isna()].index.tolist()
        }
        
        print(f"{col}:")
        print(f"  - Missing: {missing_count} games ({missing_pct:.1f}%)")
        print(f"  - Present: {total_games - missing_count} games")
        print()
    
    # Find games missing ANY prediction data
    print("🎯 GAMES MISSING ANY PREDICTION DATA:")
    print("-" * 40)
    
    # Create a mask for games missing any prediction column
    missing_any_mask = df[existing_pred_columns].isna().any(axis=1)
    games_missing_any = df[missing_any_mask]
    
    if len(games_missing_any) > 0:
        print(f"📊 Found {len(games_missing_any)} games missing at least one prediction value")
        print()
        
        # Show first 10 games with missing predictions
        print("📋 Sample games with missing predictions (first 10):")
        print("-" * 60)
        
        for idx, (_, game) in enumerate(games_missing_any.head(10).iterrows()):
            print(f"{idx+1}. {game['date']}: {game['away_team']} @ {game['home_team']}")
            
            # Show which specific predictions are missing
            missing_for_this_game = []
            for col in existing_pred_columns:
                if pd.isna(game[col]):
                    missing_for_this_game.append(col)
            
            print(f"   Missing: {', '.join(missing_for_this_game)}")
            print()
        
        if len(games_missing_any) > 10:
            print(f"... and {len(games_missing_any) - 10} more games with missing predictions")
        
        # Save games with missing predictions to a separate file
        output_file = data_dir / "massey_pred_games_2025_missing_predictions.csv"
        games_missing_any.to_csv(output_file, index=False)
        print(f"💾 Saved {len(games_missing_any)} games with missing predictions to: {output_file}")
        
    else:
        print("✅ All games have complete prediction data!")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS:")
    print("=" * 60)
    print(f"Total games: {total_games}")
    print(f"Games with complete predictions: {total_games - len(games_missing_any)}")
    print(f"Games missing any prediction: {len(games_missing_any)}")
    
    if existing_pred_columns:
        completion_rate = ((total_games - len(games_missing_any)) / total_games) * 100
        print(f"Data completion rate: {completion_rate:.1f}%")

if __name__ == "__main__":
    check_missing_predictions()
