#!/usr/bin/env python3
"""
NPI Dashboard using Streamlit
Simple dashboard to view NPI ratings for all teams
"""

import streamlit as st
import pandas as pd
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
    
    # Check if we have at least one data source
    if npi_df is None and games_df is None:
        st.error(f"No data found for {selected_year}. Please ensure either NPI ratings or games data exists.")
        st.stop()
    
    # Show warning if no NPI data available
    if npi_df is None:
        st.warning(f"⚠️ No NPI ratings available for {selected_year} (this is normal for seasons with only scheduled games)")
        st.info("📊 You can still view team schedules and individual game data.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏆 NPI Ratings", "📅 Team Schedules"])
    
    # Tab 1: NPI Ratings
    with tab1:
        st.subheader("NPI Ratings")
        
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
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created by Kevin Puorro • Data from Massey Ratings*")

if __name__ == "__main__":
    main()
