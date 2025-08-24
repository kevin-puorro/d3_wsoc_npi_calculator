#!/usr/bin/env python3
"""
NPI Dashboard using Streamlit
Simple dashboard to view NPI ratings for all teams
"""

import streamlit as st
import pandas as pd
import subprocess
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NPI Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for all st.info elements
st.markdown("""
<style>
/* Target all st.info elements */
div[data-testid="stInfo"] {
    background-color: #830019 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

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


def run_npi_simulation(year):
    """Run NPI calculator with simulation flag and return results"""
    try:
        # Get script directory and project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        # Use absolute paths instead of changing working directory
        npi_calculator_path = script_dir / "npi_calculator.py"
        
        # Set environment variables to force UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        # Run NPI calculator with simulation flag using absolute paths and proper encoding
        result = subprocess.run([
            "python", str(npi_calculator_path), 
            "--year", str(year), 
            "--season-only", 
            "--simulation"
        ], capture_output=True, text=True, encoding='utf-8', timeout=300, 
           cwd=project_root, env=env)  # Set cwd to project root and encoding
        
        if result.returncode == 0:
            return True, result.stdout, None
        else:
            return False, None, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, None, "NPI calculation timed out (took longer than 5 minutes)"
    except Exception as e:
        return False, None, str(e)


def reset_simulation_data(year):
    """Reset simulation data by copying filtered data back to simulation file"""
    try:
        # Get script directory and project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Source file (filtered data)
        filtered_file = data_dir / f"massey_games_{year}_filtered.csv"
        
        # Destination file (simulation data)
        sim_file = data_dir / f"massey_games_{year}_simulation.csv"
        
        # Simulation NPI results file (to be cleared)
        sim_npi_file = data_dir / f"npi_ratings_{year}_simulation.csv"
        
        if not filtered_file.exists():
            return False, f"Filtered data file not found: {filtered_file}"
        
        # Read the filtered data
        filtered_df = pd.read_csv(filtered_file)
        
        # Copy to simulation file
        filtered_df.to_csv(sim_file, index=False)
        
        # Clear any existing simulation NPI results
        if sim_npi_file.exists():
            sim_npi_file.unlink()  # Delete the file
            print(f"Cleared existing simulation NPI results: {sim_npi_file}")
        
        return True, f"Reset simulation data: copied {len(filtered_df)} games from filtered data and cleared NPI results"
        
    except Exception as e:
        return False, f"Error resetting simulation data: {e}"


def apply_simulation_changes(year, selected_team, sim_data):
    """Apply simulation changes to the simulation data file"""
    try:
        # Get script directory and project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        data_dir = project_root / "data" / str(year)
        
        # Load the simulation data file
        sim_file = data_dir / f"massey_games_{year}_simulation.csv"
        
        # Safety check - verify we're working with simulation file, not original
        original_npi_file = data_dir / f"npi_ratings_{year}.csv"
        if original_npi_file.exists():
            original_size = original_npi_file.stat().st_size
            print(f"SAFETY CHECK: Original NPI file size: {original_size} bytes")
        
        if not sim_file.exists():
            return False, f"Simulation file not found: {sim_file}"
        
        print(f"SAFETY CHECK: Working with simulation file: {sim_file}")
        print(f"SAFETY CHECK: Original NPI file: {original_npi_file}")
        
        # Load current simulation data
        sim_df = pd.read_csv(sim_file)
        
        # Apply each change to the simulation data
        changes_applied = 0
        for change in sim_data:
            if change['changed']:
                # Find the game in simulation data
                if change['home_away'] == 'HOME':
                    # Team is home, find away team as opponent
                    mask = (sim_df['home_team'] == selected_team) & (sim_df['away_team'] == change['opponent'])
                else:
                    # Team is away, find home team as opponent
                    mask = (sim_df['away_team'] == selected_team) & (sim_df['home_team'] == change['opponent'])
                
                if mask.any():
                    game_idx = mask.idxmax()
                    
                    # Update scores based on simulated result
                    if change['simulated_result'] == 'W':
                        if change['home_away'] == 'HOME':
                            sim_df.loc[game_idx, 'home_score'] = 2
                            sim_df.loc[game_idx, 'away_score'] = 1
                        else:
                            sim_df.loc[game_idx, 'away_score'] = 2
                            sim_df.loc[game_idx, 'home_score'] = 1
                    elif change['simulated_result'] == 'L':
                        if change['home_away'] == 'HOME':
                            sim_df.loc[game_idx, 'home_score'] = 1
                            sim_df.loc[game_idx, 'away_score'] = 2
                        else:
                            sim_df.loc[game_idx, 'away_score'] = 1
                            sim_df.loc[game_idx, 'home_score'] = 2
                    else:  # Tie
                        sim_df.loc[game_idx, 'home_score'] = 1
                        sim_df.loc[game_idx, 'away_score'] = 1
                    
                    # Update game_note to remove SCH if it was scheduled
                    if sim_df.loc[game_idx, 'game_note'] == 'Sch':
                        sim_df.loc[game_idx, 'game_note'] = ''
                    
                    changes_applied += 1
        
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
        
        return True, f"Applied {changes_applied} changes to simulation data"
        
    except Exception as e:
        return False, f"Error applying simulation changes: {e}"


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
            # Team is home
            opponent = game['away_team']
            home_away = "HOME"
            team_score = game['home_score']
            opp_score = game['away_score']
        else:
            # Team is away
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
        
        st.markdown("---")
        
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
        st.info("📊 You can still view team schedules and individual game data.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏆 NPI Ratings", "📅 Team Schedules", "🎯 NPI Simulator"])
    
    # Tab 1: NPI Ratings
    with tab1:
        st.subheader("NPI Ratings")
        
        # Check for simulation NPI data
        if sim_npi_df is not None and not sim_npi_df.empty:
            st.success("🎯 **Simulation NPI Ratings Available!**")            
            # Sort by NPI rating (highest first)
            df_sorted = sim_npi_df.sort_values('npi_rating', ascending=False).reset_index(drop=True)
            
            # Add ranking
            df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
            
            # Select only the columns we want to display
            columns_to_show = ['Rank', 'team', 'npi_rating', 'wins', 'losses', 'ties']
            df_display = df_sorted[columns_to_show].copy()
            
            # Round NPI rating to 2 decimal places
            df_display['npi_rating'] = df_display['npi_rating'].round(2)
            
            # Display simulation NPI data
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("Original NPI Ratings")
        
        if npi_df is not None and not npi_df.empty:
            st.markdown("*Note: These estimated NPI ratings have a margin of error of ±0.3*")
            
            # Sort by NPI rating (highest first)
            df_sorted = npi_df.sort_values('npi_rating', ascending=False).reset_index(drop=True)
            
            # Add ranking
            df_sorted.insert(0, 'Rank', range(1, len(df_sorted) + 1))
            
            # Select only the columns we want to display
            columns_to_show = ['Rank', 'team', 'npi_rating', 'wins', 'losses', 'ties']
            df_display = df_sorted[columns_to_show].copy()
            
            # Round NPI rating to 2 decimal places
            df_display['npi_rating'] = df_display['npi_rating'].round(2)
            
            # Display all teams
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("📅 **No NPI ratings available for this season.**")
            st.markdown("""
            This typically happens when:
            - All games are scheduled (like 2025 season)
            - No completed games exist yet
            - NPI calculations haven't been run
            
            **You can still view team schedules in the Team Schedules tab!**
            """)
    
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
            st.info("💡 **For 2025 season:** Run NPI calculator with `--simulation` flag to create simulation NPI data.")
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
                    session_key = f"sim_data_{selected_year}_{selected_sim_team}"
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
                    st.write("**Step 4: Save and Run Simulation**")
                    
                    # Helpful reminder for users
                    st.info("💡 **Important:** Make sure to save your simulation data before running the NPI calculator!")
                    
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                    
                    with col1:
                        pass
                    
                    with col2:
                        if st.button("🔄 Reset All Changes", type="secondary"):
                            # Clear session state for this team
                            if session_key in st.session_state:
                                del st.session_state[session_key]
                            
                            # Reset simulation data by copying filtered data back
                            with st.spinner("Resetting simulation data..."):
                                reset_success, reset_message = reset_simulation_data(selected_year)
                            
                            if reset_success:
                                st.success(f"✅ {reset_message}")
                                st.info("🔄 Simulation data has been reset to original filtered data")
                                st.info("🔄 All sliders have been reset to original values")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to reset simulation data: {reset_message}")
                    
                    with col3:
                        if st.button("💾 Save Sim Data", type="secondary"):
                            # Apply changes to simulation data without running NPI calculation
                            changes_made = [game for game in sim_data if game['changed']]
                            
                            if changes_made:
                                
                                with st.spinner("Saving simulation data..."):
                                    apply_success, apply_message = apply_simulation_changes(selected_year, selected_sim_team, sim_data)
                                
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
                                    st.error(f"❌ Failed to save simulation data: {apply_message}")
                            else:
                                st.info("No changes to save.")
                    
                    with col4:
                        if st.button("📊 Run Simulation", type="secondary"):
                            # Run NPI calculator directly
                            
                            with st.spinner("Calculating NPI ratings..."):
                                success, output, error = run_npi_simulation(selected_year)
                            
                            if success:
                                pass
                            else:
                                st.error("❌ NPI calculation failed!")
                                st.error(f"Error: {error}")
                                st.info("💡 **Manual Steps:**")
                                st.write(f"1. Run NPI calculator manually: `python npi_calculator.py --year {selected_year} --season-only --simulation`")
                                st.write("2. Check for any error messages")
                                st.write("3. Verify the simulation data file is correct")
                    
                    with col5:
                        pass
                
                else:
                    st.warning(f"No games found for {selected_sim_team}")
                    st.info("💡 **Tip:** Select a different team or check if the team has any games in the current season.")
                
                # Step 5: View Current Simulation NPI Ratings
                st.write("---")
                st.write("**Step 5: View Simulated NPI Ratings**")
                
                # Check if simulation NPI data exists
                sim_npi_file = data_dir / f"npi_ratings_{selected_year}_simulation.csv"
                
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
                            
                            # Display simulation NPI data
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                            
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
                    st.info(f"📊 **No simulated NPI ratings available yet**")
                

    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created by Kevin Puorro • Data from Massey Ratings*")

if __name__ == "__main__":
    main()
