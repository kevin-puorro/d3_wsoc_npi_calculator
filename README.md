# D3 Women's Soccer NPI Calculator

This project calculates National Power Index (NPI) ratings for D3 women's soccer teams using data scraped from Massey Ratings.

## GitHub Actions Setup

The project includes a GitHub Actions workflow that automatically runs the NPI data pipeline daily for the 2025 season.

### What the Workflow Does

The `daily-npi-pipeline.yml` workflow runs every day at 6:00 AM UTC and:

1. **Scrapes** fresh game data from Massey Ratings for the 2025 season
2. **Filters** the data to include only teams in your team mapping
3. **Calculates** NPI ratings based on the filtered games
4. **Commits** the updated data back to your repository
5. **Uploads** the data as GitHub artifacts for easy access

### Setup Instructions

1. **Push the workflow file**: The `.github/workflows/daily-npi-pipeline.yml` file should be committed to your repository.

2. **Enable GitHub Actions**: Go to your repository's "Actions" tab and ensure GitHub Actions are enabled.

3. **Set up repository permissions**: The workflow needs permission to commit back to your repository:
   - Go to Settings → Actions → General
   - Under "Workflow permissions", select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

4. **Verify team mapping**: Ensure your `team_mapping.txt` file is up to date with the teams you want to include.

### Manual Triggering

You can manually run the workflow anytime:
- Go to the "Actions" tab in your repository
- Click on "Daily NPI Pipeline - 2025 Season"
- Click "Run workflow" button

### Workflow Schedule

- **Automatic**: Runs daily at 6:00 AM UTC (2:00 AM EST, 1:00 AM CST)
- **Manual**: Can be triggered anytime via the GitHub Actions interface

### Output Files

The workflow generates these files in the `data/2025/` directory:
- `massey_games_2025.csv` - Raw scraped data
- `massey_games_2025_filtered.csv` - Filtered games for NPI calculation
- `massey_games_2025_npi.csv` - Games used in NPI calculations
- `npi_ratings_2025.csv` - Final NPI ratings for all teams

### Troubleshooting

If the workflow fails:

1. Check the Actions tab for error logs
2. Verify your `requirements.txt` includes all necessary dependencies
3. Ensure your scripts can run independently with the `--year 2025` argument
4. Check that the `data/2025/` directory structure exists

### Customization

To modify the schedule or add other seasons:
- Edit the `cron` expression in the workflow file
- Add additional `--year` parameters for other seasons
- Modify the commit message or artifact retention settings

## Local Development

To run the pipeline locally:

```bash
cd scripts
python massey_games_scraper.py --year 2025
python filter_massey_games.py --year 2025
python npi_calculator.py --year 2025
```

## Requirements

- Python 3.9+
- pandas
- numpy
- requests
- streamlit (for dashboard)
