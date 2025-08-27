#!/usr/bin/env python3
"""
Merge Filled Predictions Script
Merges the filled-in prediction data with the existing filtered dataset
to create a complete version with all predictions filled in.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def merge_filled_predictions():
    """Merge filled-in predictions with the filtered dataset"""
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "2025"
    
    # Input files
    filtered_file = data_dir / "massey_pred_games_2025_filtered.csv"
    filled_predictions_file = data_dir / "massey_pred_games_2025_missing_predictions.csv"
    
    # Output file
    output_file = data_dir / "massey_pred_games_2025_filtered_complete.csv"
    
    print("🔍 Merging filled predictions with filtered dataset...")
    print("=" * 60)
    
    # Check if input files exist
    if not filtered_file.exists():
        print(f"❌ Filtered file not found: {filtered_file}")
        return
    
    if not filled_predictions_file.exists():
        print(f"❌ Filled predictions file not found: {filled_predictions_file}")
        return
    
    # Load the datasets
    print("📂 Loading datasets...")
    filtered_df = pd.read_csv(filtered_file)
    filled_df = pd.read_csv(filled_predictions_file)
    
    print(f"📊 Filtered dataset: {len(filtered_df)} games")
    print(f"📊 Filled predictions: {len(filled_df)} games")
    print(f"📋 Filtered columns: {list(filtered_df.columns)}")
    print(f"📋 Filled columns: {list(filled_df.columns)}")
    print()
    
    # Convert date formats to match
    print("🔄 Converting date formats...")
    print("Filtered dates format:", filtered_df['date'].iloc[0])
    print("Filled dates format:", filled_df['date'].iloc[0])
    
    # Convert filled predictions dates from M/D/YYYY to YYYY-MM-DD
    def convert_date_format(date_str):
        try:
            # Parse M/D/YYYY format and convert to YYYY-MM-DD
            parsed_date = datetime.strptime(str(date_str), '%m/%d/%Y')
            return parsed_date.strftime('%Y-%m-%d')
        except:
            return str(date_str)
    
    filled_df['date'] = filled_df['date'].apply(convert_date_format)
    print("Converted filled dates format:", filled_df['date'].iloc[0])
    print()
    
    # Verify both datasets have the same columns
    if list(filtered_df.columns) != list(filled_df.columns):
        print("❌ Column mismatch between datasets!")
        print("Filtered columns:", list(filtered_df.columns))
        print("Filled columns:", list(filled_df.columns))
        return
    
    # Create a copy of the filtered dataset
    merged_df = filtered_df.copy()
    
    print("🔄 Updating missing predictions...")
    
    # Find games in the filtered dataset that match the filled predictions
    # We'll match by date, away_team, and home_team
    updated_count = 0
    
    for idx, filled_game in filled_df.iterrows():
        # Find matching game in the filtered dataset
        match_mask = (
            (merged_df['date'] == filled_game['date']) &
            (merged_df['away_team'] == filled_game['away_team']) &
            (merged_df['home_team'] == filled_game['home_team'])
        )
        
        if match_mask.any():
            # Update the matching row with filled predictions
            match_idx = match_mask.idxmax()
            
            # Update prediction columns
            prediction_columns = ['home_pred_score', 'away_pred_score', 'home_win_prob', 'away_win_prob']
            for col in prediction_columns:
                if col in filled_game and not pd.isna(filled_game[col]):
                    merged_df.at[match_idx, col] = filled_game[col]
            
            updated_count += 1
        else:
            print(f"⚠️  Could not find match for: {filled_game['date']} {filled_game['away_team']} @ {filled_game['home_team']}")
    
    print(f"✅ Updated {updated_count} games with filled predictions")
    print()
    
    # Verify no missing predictions remain
    print("🔍 Verifying data completeness...")
    prediction_columns = ['home_pred_score', 'away_pred_score', 'home_win_prob', 'away_win_prob']
    
    missing_before = filtered_df[prediction_columns].isna().any(axis=1).sum()
    missing_after = merged_df[prediction_columns].isna().any(axis=1).sum()
    
    print(f"📊 Missing predictions before: {missing_before} games")
    print(f"📊 Missing predictions after: {missing_after} games")
    print(f"📊 Improvement: {missing_before - missing_after} games fixed")
    
    if missing_after == 0:
        print("🎉 All games now have complete prediction data!")
    else:
        print(f"⚠️  Still {missing_after} games with missing predictions")
    
    # Save the merged dataset
    print(f"\n💾 Saving complete dataset to: {output_file}")
    merged_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved {len(merged_df)} games with complete predictions")
    print(f"📁 File location: {output_file}")
    
    # Show final statistics
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY:")
    print("=" * 60)
    print(f"Total games: {len(merged_df)}")
    print(f"Games with complete predictions: {len(merged_df) - missing_after}")
    print(f"Games missing any prediction: {missing_after}")
    
    if missing_after == 0:
        completion_rate = 100.0
    else:
        completion_rate = ((len(merged_df) - missing_after) / len(merged_df)) * 100
    
    print(f"Data completion rate: {completion_rate:.1f}%")
    
    return output_file

if __name__ == "__main__":
    merge_filled_predictions()
