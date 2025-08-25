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
        for change in sim_data:
            if change['changed']:
                print(f"Processing change: {change}")  # Debug info
                
                # Find the game in simulation data
                # Need to match based on the actual CSV structure, not the processed view
                if change['home_away'] == 'HOME':
                    # In the CSV, if team is "HOME" in the UI, they are actually listed as away_team
                    # This is due to the home/away logic fix we implemented earlier
                    mask = (sim_df['away_team'] == selected_team) & (sim_df['home_team'] == change['opponent'])
                else:
                    # In the CSV, if team is "AWAY" in the UI, they are actually listed as home_team
                    mask = (sim_df['home_team'] == selected_team) & (sim_df['away_team'] == change['opponent'])
                
                if mask.any():
                    game_idx = mask.idxmax()
                    print(f"Found game at index {game_idx}: {sim_df.loc[game_idx]}")  # Debug info
                    
                    # Update scores based on simulated result
                    if change['simulated_result'] == 'W':
                        if change['home_away'] == 'HOME':
                            # Team is HOME in UI, so they should win (higher score)
                            sim_df.loc[game_idx, 'away_score'] = 2  # away_team in CSV
                            sim_df.loc[game_idx, 'home_score'] = 1  # home_team in CSV
                        else:
                            # Team is AWAY in UI, so they should win (higher score)
                            sim_df.loc[game_idx, 'home_score'] = 2  # home_team in CSV
                            sim_df.loc[game_idx, 'away_score'] = 1  # away_team in CSV
                    elif change['simulated_result'] == 'L':
                        if change['home_away'] == 'HOME':
                            # Team is HOME in UI, so they should lose (lower score)
                            sim_df.loc[game_idx, 'away_score'] = 1  # away_team in CSV
                            sim_df.loc[game_idx, 'home_score'] = 2  # home_team in CSV
                        else:
                            # Team is AWAY in UI, so they should lose (lower score)
                            sim_df.loc[game_idx, 'home_score'] = 1  # home_team in CSV
                            sim_df.loc[game_idx, 'away_score'] = 2  # away_team in CSV
                    else: # Tie
                        sim_df.loc[game_idx, 'home_score'] = 1
                        sim_df.loc[game_idx, 'away_score'] = 1
                    
                    # Update game_note to remove SCH if it was scheduled
                    if sim_df.loc[game_idx, 'game_note'] == 'Sch':
                        sim_df.loc[game_idx, 'game_note'] = ''
                    
                    changes_applied += 1
                    print(f"Applied change {changes_applied}: {change['opponent']} -> {change['simulated_result']}")
                else:
                    print(f"WARNING: Could not find game for {selected_team} vs {change['opponent']} ({change['home_away']})")
                    print(f"Available games for {selected_team}:")
                    team_games = sim_df[(sim_df['home_team'] == selected_team) | (sim_df['away_team'] == selected_team)]
                    print(team_games[['home_team', 'away_team', 'home_score', 'away_score', 'game_note']].to_string())
        
        # Save modified simulation data
        sim_df.to_csv(sim_file, index=False)
        
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


def get_team_schedule(team_name, games_df):
    """Get schedule for a specific team"""
    if games_df is None:
        return None
    
    # Find games where the team is either home or away
    team_games = games_df[
        (games_df['home_team'] == team_name) | 
        (games_df['away_team'] == team_name)
    ].copy()
    
    if team_games.empty:
        return None
    
    # Create a more readable schedule format
    schedule_data = []
    for _, game in team_games.iterrows():
        if game['home_team'] == team_name:
            # Team is actually away (but listed as home in CSV)
            opponent = game['away_team']
            home_away = "AWAY"
            team_score = game['home_score']
            opp_score = game['away_score']
        else:
            # Team is actually home (but listed as away in CSV)
            opponent = game['home_team']
            home_away = "HOME"
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
        
        # Check for overtime (only for completed games)
        overtime = ""
        if not is_scheduled and pd.notna(game['game_note']) and str(game['game_note']).strip() != "":
            if 'o1' in str(game['game_note']).lower() or 'o2' in str(game['game_note']).lower() or 'o3' in str(game['game_note']).lower():
                overtime = "OT"
        
        schedule_data.append({
            'Date': game['date'],
            'Opponent': opponent,
            'Home/Away': home_away,
            'Team Score': team_score,
            'Opponent Score': opp_score,
            'Result': result,
            'Overtime': overtime,
            'Status': 'Scheduled' if is_scheduled else 'Completed'
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
            st.markdown("""
            <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>📅 2025 Season Note:</strong> Season has not started yet and games have not been played yet.
            </div>
            """, unsafe_allow_html=True)
    
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
    tab1, tab2, tab3 = st.tabs(["🏆 NPI Rankings", "📅 Team Schedules", "🎯 NPI Simulator"])
    
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
            
            selected_team = st.selectbox(
                "Select a team to view their schedule:",
                options=all_teams,
                index=0
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
            schedule_df = get_team_schedule(selected_team, games_df)
            
            if schedule_df is not None and not schedule_df.empty:
                st.subheader(f"📅 {selected_team} Schedule")
                
                # Format date for display
                schedule_display = schedule_df.copy()
                schedule_display['Date'] = schedule_display['Date'].dt.strftime('%m/%d/%Y')
                
                # Clean up display columns for better readability
                display_columns = ['Date', 'Opponent', 'Home/Away', 'Team Score', 'Opponent Score', 'Result', 'Overtime', 'Status']
                schedule_display = schedule_display[display_columns]
                
                # Display schedule
                st.dataframe(schedule_display, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No schedule data found for {selected_team}")
    
    # Tab 3: NPI Simulator
    with tab3:
        st.subheader(f"NPI Simulator")
        st.markdown(f"Simulate different game outcomes for the **{selected_year} season** to see how they affect NPI rankings.")
        
        # Check if we have the required data
        if games_df is None:
            st.warning("⚠️ NPI simulator requires games data. Please ensure the filtered games file exists.")
        elif npi_df is None or npi_df.empty:
            st.warning("⚠️ NPI simulator requires NPI ratings data. Please run NPI calculations first.")
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
                selected_sim_team = st.selectbox(
                    f"Choose a team ({selected_year} season):",
                    options=sim_teams,
                    index=0,
                    key="sim_team_select"
                )
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
                st.write("**Step 3: Modify Game Results**")
                
                team_games = get_team_schedule(selected_sim_team, games_df)
                
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
                            st.write(f"**{game['Opponent']}** ({game['Home/Away']}) - {game['Date']}")
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
                                st.write(f"**Original Result:** {original_result}")
                        
                        with col2:
                            # Show original score if available
                            if original_result != 'SCH':
                                st.write(f"Score: {game['Team Score']}-{game['Opponent Score']}")
                        
                        with col3:
                            # Simulation selector with session state
                            game_key = f"game_{idx}"
                            if game_key not in st.session_state[session_key]:
                                st.session_state[session_key][game_key] = original_result
                            
                            if original_result == 'SCH':
                                sim_result = st.selectbox(
                                    "Simulate Result:",
                                    options=["SCH", "W", "L", "T"],
                                    key=f"sim_{selected_year}_{selected_sim_team}_{idx}",
                                    index=["SCH", "W", "L", "T"].index(st.session_state[session_key][game_key]) if st.session_state[session_key][game_key] in ["SCH", "W", "L", "T"] else 0,
                                    label_visibility="collapsed"
                                )
                            else:
                                sim_result = st.selectbox(
                                    "Change to:",
                                    options=["SCH", "W", "L", "T"],
                                    key=f"change_{selected_year}_{selected_sim_team}_{idx}",
                                    index=["SCH", "W", "L", "T"].index(st.session_state[session_key][game_key]) if st.session_state[session_key][game_key] in ["SCH", "W", "L", "T"] else ["SCH", "W", "L", "T"].index(original_result) if original_result in ["SCH", "W", "L", "T"] else 0,
                                    label_visibility="collapsed"
                                )
                            
                            # Update session state
                            st.session_state[session_key][game_key] = sim_result
                        
                        with col4:
                            # Show change indicator
                            if original_result == 'SCH' or sim_result != original_result:
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
                        
                        # Store simulation data
                        sim_data.append({
                            'opponent': game['Opponent'],
                            'home_away': game['Home/Away'],
                            'original_result': original_result,
                            'simulated_result': sim_result,
                            'changed': original_result != sim_result,
                            'date': game['Date']
                        })
                    
                    # Simulation controls
                    st.write("---")
                    st.write("**Step 4: Save Data and Run NPI Calculations**")
                    
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
                                st.success(f"✅ {reset_message}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to reset data: {reset_message}")
                        
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
                                pass
                            else:
                                st.error("❌ NPI calculation failed!")
                                st.error(f"Error: {error}")
                                st.markdown("""
                                <div style="background-color: #830019; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                                    <strong>💡 **Manual Steps:**</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                st.write(f"1. Run NPI calculator manually: `python npi_calculator.py --year {selected_year} --season-only --simulation`")
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
                st.write("**Step 5: View Simulated NPI Rankings**")
                
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
                            st.markdown("*Note: These simulated NPI ratings have a margin of error of ±0.3*")
                            
                            # Display simulation NPI data
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                            
                            # Add important note about NPI rating dependencies
                            st.error("**Important Note:** NPI ratings are largely dependent on other teams and their results. Simulated results may not accurately reflect final rankings.")
                            
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
                                        st.metric("Simulated NPI", f"{simulated_team['npi_rating']:.2f}", delta=delta_display)
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
                        <strong>📊 **No simulated NPI ratings available yet**</strong><br>
                    </div>
                    """, unsafe_allow_html=True)
                

    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created by Kevin Puorro • Data from Massey Ratings*")

if __name__ == "__main__":
    main()
