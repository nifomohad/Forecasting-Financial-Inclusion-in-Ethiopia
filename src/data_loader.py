# src/data_loader.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class EthiopiaFIData:
    """
    Flexible class to load, explore, and enrich the Ethiopia Financial Inclusion unified dataset.
    Supports both .csv and .xlsx formats for main dataset and reference codes.
    Designed for reuse across notebooks.
    """

    def __init__(self,
                 main_path: str = "data/raw/ethiopia_fi_unified_data.xlsx",
                 ref_path: str = "data/raw/reference_codes.xlsx"):
        """
        Initialize by loading the datasets.
        Automatically detects .csv or .xlsx and loads accordingly.
        """
        self.main_path = Path(main_path)
        self.ref_path = Path(ref_path)

        if not self.main_path.exists():
            raise FileNotFoundError(
                f"Main dataset not found: {self.main_path.resolve()}")
        if not self.ref_path.exists():
            raise FileNotFoundError(
                f"Reference codes not found: {self.ref_path.resolve()}")

        # Flexible main dataset loading
        if self.main_path.suffix.lower() in ['.xlsx', '.xls']:
            print(f"Loading main Excel file: {self.main_path.resolve()}")
            self.df = pd.read_excel(self.main_path)
        else:
            print(f"Loading main CSV file: {self.main_path.resolve()}")
            self.df = pd.read_csv(self.main_path)

        # Flexible reference loading
        if self.ref_path.suffix.lower() in ['.xlsx', '.xls']:
            print(f"Loading reference Excel file: {self.ref_path.resolve()}")
            self.ref_codes = pd.read_excel(self.ref_path)
        else:
            print(f"Loading reference CSV file: {self.ref_path.resolve()}")
            self.ref_codes = pd.read_csv(self.ref_path)

        # Standardize date columns
        date_cols = ['observation_date', 'event_date', 'date']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        print(
            f"Loaded main dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Loaded reference codes: {self.ref_codes.shape}")

    def get_record_counts(self) -> Dict[str, Any]:
        """Count records by key categorical fields."""
        counts = {
            "by_record_type": self.df['record_type'].value_counts().to_dict(),
            "by_pillar": self.df['pillar'].value_counts(dropna=False).to_dict(),
            "by_source_type": self.df.get('source_type', pd.Series()).value_counts(dropna=False).to_dict(),
            "by_confidence": self.df.get('confidence', pd.Series()).value_counts(dropna=False).to_dict(),
        }
        return counts

    def get_temporal_range(self) -> Dict[str, Any]:
        """Identify temporal range of observations and events."""
        date_col = self._get_date_column()
        if date_col is None:
            return {"error": "No date column found"}

        obs_dates = self.df[self.df['record_type'] == 'observation'][date_col]
        event_dates = self.df[self.df['record_type'] == 'event'][date_col]

        return {
            "overall_min": self.df[date_col].min(),
            "overall_max": self.df[date_col].max(),
            "observations_min": obs_dates.min() if not obs_dates.empty else None,
            "observations_max": obs_dates.max() if not obs_dates.empty else None,
            "events_min": event_dates.min() if not event_dates.empty else None,
            "events_max": event_dates.max() if not event_dates.empty else None,
        }

    def list_unique_indicators(self) -> pd.DataFrame:
        """List unique indicators with their coverage."""
        indicator_col = 'indicator_code' if 'indicator_code' in self.df.columns else 'indicator' if 'indicator' in self.df.columns else None
        if indicator_col is None:
            print("No indicator or indicator_code column found.")
            return pd.DataFrame()

        coverage = self.df[indicator_col].value_counts().reset_index()
        coverage.columns = [indicator_col, 'count']
        return coverage

    def get_events_summary(self) -> pd.DataFrame:
        """Summary of cataloged events (sorted by date, safe column selection)."""
        events = self.df[self.df['record_type'] == 'event'].copy()
        if events.empty:
            print("No events found.")
            return pd.DataFrame()

        date_col = self._get_date_column()
        if date_col and date_col in events.columns:
            events = events.sort_values(date_col)

        # Safe column selection
        possible_cols = ['category', date_col or 'event_date', 'description',
                         'event_description', 'notes', 'text', 'source_name', 'event_name']
        available_cols = [c for c in possible_cols if c in events.columns]

        if not available_cols:
            print("No recognizable columns for events summary. Showing all columns.")
            return events.head(20)

        return events[available_cols].head(20)

    def get_impact_links_summary(self) -> pd.DataFrame:
        """Summary of existing impact links (safe column selection)."""
        links = self.df[self.df['record_type'] == 'impact_link']
        if links.empty:
            print("No impact links found.")
            return pd.DataFrame()

        possible_cols = ['parent_id', 'pillar', 'related_indicator',
                         'impact_direction', 'impact_magnitude', 'lag_months', 'evidence_basis']
        available_cols = [c for c in possible_cols if c in links.columns]

        if not available_cols:
            return links.head(20)

        return links[available_cols].head(20)

    def plot_temporal_coverage(self, freq: str = 'Y'):
        """Plot observations by time period and indicator (robust to invalid dates)."""
        date_col = self._get_date_column()
        if date_col is None:
            print("No date column for plotting")
            return

        obs = self.df[self.df['record_type'] == 'observation'].copy()
        if obs.empty:
            print("No observations to plot")
            return

        # Drop invalid dates
        obs = obs.dropna(subset=[date_col])
        if obs.empty:
            print("No observations with valid dates for plotting")
            return

        indicator_col = 'indicator_code' if 'indicator_code' in obs.columns else 'indicator' if 'indicator' in obs.columns else None
        if indicator_col is None:
            print("No indicator_code or indicator column for plotting")
            return

        # Re-coerce dates
        obs[date_col] = pd.to_datetime(obs[date_col], errors='coerce')

        grouped = obs.groupby(
            [pd.Grouper(key=date_col, freq=freq), indicator_col]).size()
        if grouped.empty:
            print("No data to plot after grouping")
            return

        ax = grouped.unstack().plot(kind='bar', stacked=True, figsize=(14, 7))
        plt.title('Temporal Coverage of Observations by Indicator')
        plt.ylabel('Number of Observations')
        plt.xlabel('Time Period')
        plt.legend(title=indicator_col.capitalize(),
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def add_records(self, new_records: list[dict]):
        """Add new records for enrichment."""
        if not new_records:
            print("No records to add.")
            return

        new_df = pd.DataFrame(new_records)
        # Align columns
        for col in self.df.columns:
            if col not in new_df.columns:
                new_df[col] = pd.NA

        self.df = pd.concat(
            [self.df, new_df[self.df.columns]], ignore_index=True)
        print(
            f"Added {len(new_records)} new records. Total rows: {len(self.df)}")

    def save_enriched(self, output_path: str = "data/processed/ethiopia_fi_unified_data_enriched.csv"):
        """Save the enriched dataset to data/processed/."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"Enriched dataset saved to: {output_path.resolve()}")

    def _get_date_column(self) -> Optional[str]:
        for col in ['observation_date', 'event_date', 'date']:
            if col in self.df.columns:
                return col
        return None
