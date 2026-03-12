import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import wesanderson
import seaborn as sns


class DataPreprocessing:
    def __init__(
        self,
        path="classes/data",
        file_name="online_retail_II.xlsx",
        freq='M',  # 'M'/'2M' for monthly aggregation, 'W'/'2W' for weekly
        period=3,
        save=False,
        half_data=False
    ):
        self.path = path
        self.file_name = file_name
        self.save = save
        self.freq = freq
        self.period = period
        self.half_data = half_data
        self.seed = np.random.seed(42)
        self.df = None
        self.agg_df = None
        self.full_agg = None

    def read_data(self):
        df_1 = pd.read_excel(f"{self.path}/{self.file_name}", sheet_name=0)
        df_2 = pd.read_excel(f"{self.path}/{self.file_name}", sheet_name=1)

        df = pd.concat([df_1, df_2], axis=0)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={
            "Customer ID": "CustomerID",
            "Invoice": "InvoiceNo"
            }, inplace=True)
        self.df = df

    def prepare_data(self):
        '''
        Parse InvoiceDate as datetime.
        '''
        # self.df = pd.read_excel(f"{self.path}/{self.file_name}")
        self.df['InvoiceDate'] = pd.to_datetime(
            self.df['InvoiceDate']
            ).dt.normalize()
        
        # Randomly sample 50% of the data
        if self.half_data:        
            self.df = self.df.sample(frac=0.5, random_state=42).reset_index(drop=True)

    def _assign_timestamp(self):
        '''
        Assign a timestamp column based on the chosen frequency (monthly, bimonthly, weekly, biweekly).
        '''
        if self.freq == 'M':
            # Monthly: yyyy-mm
            self.df['timestamp'] = self.df['InvoiceDate'].dt.to_period('M').astype(str)
        elif self.freq == '2M':
            # Bimonthly: group months in pairs (Jan-Feb, Mar-Apr, etc.)
            month = self.df['InvoiceDate'].dt.month
            year = self.df['InvoiceDate'].dt.year
            # Calculate bimonthly period index (0 for Jan-Feb, 1 for Mar-Apr, etc.)
            bi_month = ((month - 1) // 2) + 1
            self.df['timestamp'] = year.astype(str) + '-B' + bi_month.astype(str)
        elif self.freq == 'W':
            # Weekly: week starting date yyyy-mm-dd (Monday)
            self.df['timestamp'] = self.df['InvoiceDate'] - pd.to_timedelta(self.df['InvoiceDate'].dt.weekday, unit='d')
            self.df['timestamp'] = self.df['timestamp'].dt.strftime('%Y-%m-%d')
        elif self.freq == '2W':
            # Biweekly: every 2 weeks from the min date
            start_date = self.df['InvoiceDate'].min()
            days_since_start = (self.df['InvoiceDate'] - start_date).dt.days
            biweek_index = (days_since_start // 14).astype(int)
            self.df['timestamp'] = biweek_index.apply(lambda x: (start_date + pd.Timedelta(days=14*x)).strftime('%Y-%m-%d'))
        else:
            raise ValueError("freq must be 'M', '2M', 'W', or '2W'")

    def clean_data(self):
        '''
        Clean the data by removing rows with missing CustomerID, test products, and negative revenue.
        '''
        self.df = self.df.dropna(subset=['CustomerID'])
        self.df['Revenue'] = self.df['Quantity'] * self.df['Price']

        self._assign_timestamp()

        # Dataset contains some manual adjustment made by retailer
        # We decide to remove those
        self.df = self.df[self.df['Description'] != 'This is a test product.']
        self.df = self.df[self.df['Revenue'] >= 0]

        # Assert there are no month having negative revenue
        assert len(self.df[self.df['Revenue'] < 0]) == 0

    def remove_outliers(self):
        # Remove outliers above 99th percentile
        threshold = self.df['Revenue'].quantile(0.975)
        self.df = self.df[self.df['Revenue'] <= threshold]

    def log_transform(self):
        # Since Revenue is right skewed, apply log transformation
        self.df['Revenue'] = np.log1p(self.df['Revenue'])

    def aggregate_by_timestamp(self):
        '''
        Aggregate data by customer and timestamp, computing revenue, quantity, and other features.
        '''
        self.agg_df = self.df.groupby(['CustomerID', 'timestamp']).agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Country': 'first'
        }).reset_index()
        self.agg_df['avg_unit_price'] = np.where(
            self.agg_df['Quantity'] != 0,
            self.agg_df['Revenue'] / self.agg_df['Quantity'],
            0
        )
        invoice_counts = (
            self.df.groupby(['CustomerID', 'timestamp'])['InvoiceNo']
            .nunique()
            .reset_index()
            .rename(columns={'InvoiceNo': 'num_invoices'})
        )
        self.agg_df = self.agg_df.merge(
            invoice_counts, on=['CustomerID', 'timestamp'], how='left')
        self.agg_df['avg_rev_per_invoice'] = self.agg_df['Revenue'] / self.agg_df['num_invoices']
        self.agg_df['quantity_per_invoice'] = self.agg_df['Quantity'] / self.agg_df['num_invoices']

    def fill_missing_months(self):
        '''
        Fill in missing customer-period combinations with zeros for dormant periods.
        '''
        # Input all months that are null values to determine dormant state
        all_customers = self.agg_df['CustomerID'].unique()
        all_periods = self.agg_df['timestamp'].unique()

        # Create a MultiIndex of all possible customer-month pairs
        full_index = pd.MultiIndex.from_product(
            [all_customers, all_periods], names=['CustomerID', 'timestamp'])
        full_df = pd.DataFrame(index=full_index).reset_index()
        self.full_agg = pd.merge(
            full_df, self.agg_df, on=['CustomerID', 'timestamp'], how='left')
        self.full_agg = self.full_agg.sort_values(
            ['CustomerID', 'timestamp']).reset_index(drop=True)
        
        # Manage NaNs
        cols_to_fill = [
            'Revenue', 'Quantity', 'num_invoices',
            'avg_unit_price', 'avg_rev_per_invoice',
            'avg_rev_per_quantity', 'quantity_per_invoice'
        ]
        for col in cols_to_fill:
            if col in self.full_agg.columns:
                self.full_agg[col] = self.full_agg[col].fillna(0)

        self.full_agg['Country'] = self.full_agg.groupby(
            "CustomerID")["Country"].transform(lambda x: x.ffill().bfill())

    def add_purchase_dates(self):
        '''
        Add columns for first and last purchase timestamps for each customer.
        '''
        # Find the first purchase period for each customer
        first_purchase = (
            self.full_agg[self.full_agg['Quantity'] > 0]
            .groupby('CustomerID')['timestamp']
            .min()
            .rename('first_purchase_ts')
        )
        self.full_agg = self.full_agg.merge(first_purchase, on='CustomerID', how='left')
        self.full_agg = self.full_agg.sort_values(['CustomerID', 'timestamp'])

        # Create a column with timestamp only where a purchase happened, else NaT
        self.full_agg['last_purchase_ts'] = np.where(
            self.full_agg['Quantity'] > 0,
            pd.to_datetime(self.full_agg['timestamp']),
            pd.NaT
        )

        # Forward fill the last purchase period within each customer
        self.full_agg['last_purchase_ts'] = self.full_agg.groupby('CustomerID')['last_purchase_ts'].ffill()
        self.full_agg['timestamp_ts'] = pd.to_datetime(self.full_agg['timestamp'], errors='coerce')
        self.full_agg['last_purchase_ts'] = pd.to_datetime(self.full_agg['last_purchase_ts'], errors='coerce')
        self.full_agg['first_purchase_ts'] = pd.to_datetime(self.full_agg['first_purchase_ts'], errors='coerce')

    def add_rolling_features(self):
        '''
        Add rolling features: average revenue, activity frequency, and revenue volatility over past periods.
        '''
        # Rolling average revenue (past x periods, excluding current)
        self.full_agg[f'avg_revenue_past_{self.period}T'] = (
            self.full_agg.groupby('CustomerID')['Revenue']
            .shift(1)
            .rolling(window=self.period, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Activity frequency (number of months with purchase in past x periods)
        self.full_agg[f'activity_freq_past_{self.period}T'] = (
            self.full_agg.groupby('CustomerID')['Quantity']
            .shift(1)
            .rolling(window=self.period, min_periods=1)
            .apply(lambda x: (x > 0).sum(), raw=True)
            .reset_index(level=0, drop=True)
        )

        # Volatility (std dev of revenue in past x periods)
        self.full_agg[f'revenue_volatility_past_{self.period}T'] = (
            self.full_agg.groupby('CustomerID')['Revenue']
            .shift(1)
            .rolling(window=self.period, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    @staticmethod
    def time_diff(later, earlier):
        '''
        Calculate the number of months between two dates.
        '''
        if pd.isnull(later) or pd.isnull(earlier):
            return None
        return (later.year - earlier.year) * 12 + (later.month - earlier.month)

    def engagement_state(self, row):
        '''
        Determine the engagement state (active, inactive, dormant) for a given user at a certain timestamp.
        '''
        # Before first purchase
        if row['timestamp_ts'] < row['first_purchase_ts']:
            return 'dormant'
        # Active if purchased this period
        elif row['Quantity'] > 0:
            return 'active'
        # Calculate months/weeks since last purchase
        if pd.isnull(row['last_purchase_ts']):
            return 'dormant'
        if self.freq == 'M':
            months_since_last = self.time_diff(
                row['timestamp_ts'],
                row['last_purchase_ts']
                )
            if months_since_last is None:
                return 'dormant'
            if months_since_last > 12:
                return 'dormant'
            elif 0 < months_since_last <= 12:
                return 'inactive'
            else:
                return 'dormant'
        elif self.freq == 'W':
            weeks_since_last = (
                row['timestamp_ts'] - row['last_purchase_ts']
                ).days // 7
            if weeks_since_last is None:
                return 'dormant'
            if weeks_since_last > 52:
                return 'dormant'
            elif 0 < weeks_since_last <= 52:
                return 'inactive'
            else:
                return 'dormant'

    def add_engagement_states(self):
        '''
        Add engagement state columns for current and next period.
        '''
        self.full_agg['engagement_t0'] = self.full_agg.apply(
            self.engagement_state, axis=1)
        self.full_agg['engagement_t1'] = (
            self.full_agg.groupby('CustomerID')['engagement_t0'].shift(-1)
        )

    def save_to_csv(self, filename=f'online_retail_II_states'):
        '''
        Save the processed DataFrame to a CSV file.
        '''
        self.full_agg.to_csv(f"{filename}_{self.freq}.csv", index=False)

    def return_df(self):
        '''
        Return the processed DataFrame.
        '''
        return self.full_agg

    def plot_engagement_distribution(self):
        ''' Plot the distribution of engagement states as a bar chart. '''
        counts = self.full_agg['engagement_t0'].value_counts()
        colors_1 = wesanderson.film_palette('Hotel Chevalier')[:3]
        colors_2 = wesanderson.film_palette('Castello Cavalcanti')
        color_map = {
            'active': colors_1[0],    # green
            'inactive': colors_1[1],  # yellow
            'dormant': colors_2[4]    # red
        }
        # Prepare DataFrame for seaborn
        df = pd.DataFrame({
            'State': [str(label).capitalize() for label in counts.index],
            'Count': counts.values
        })
        palette = {str(label).capitalize(): color_map.get(label, 'gray') for label in counts.index}

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.barplot(
            data=df,
            x='State',
            y='Count',
            hue='State',         # Assign x variable to hue
            palette=palette,
            legend=False         # Hide redundant legend
        )
        plt.xlabel('')
        plt.ylabel('Number of entities', fontsize=16, color='#4f4e4e')
        plt.xticks(fontsize=14, color='#4f4e4e', rotation=0)
        plt.yticks(fontsize=14, color='#4f4e4e')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        sns.despine(left=True)
        os.makedirs('classes/plots', exist_ok=True)
        plt.savefig(f"classes/plots/{self.freq}_T{self.period}_states.png", dpi=300)
        plt.close()

    def preprocess_data(self):
        '''
        Run the full preprocessing pipeline: load, clean, aggregate, fill, add features, and plot.
        '''
        self.read_data()
        self.prepare_data()
        self.clean_data()
        self.remove_outliers()
        self.log_transform()
        self.aggregate_by_timestamp()
        self.fill_missing_months()
        self.add_purchase_dates()
        self.add_rolling_features()
        self.add_engagement_states()
        if self.save:
            self.save_to_csv()
        self.plot_engagement_distribution()


class Dataset:
    def __init__(self, df, freq, period, test_mode=False):
        self.freq = freq
        self.period = period
        self.df = df
        self.test_mode = test_mode
        self.seed = np.random.seed(42)
        self.test_size = 100
        self.state_mapping = {'active': 0, 'dormant': 1, 'inactive': 2}
        self.feature_cols = [
            'avg_unit_price', 'num_invoices',
            'avg_rev_per_invoice', 'quantity_per_invoice',
            f'avg_revenue_past_{self.period}T',
            f'activity_freq_past_{self.period}T',
            f'revenue_volatility_past_{self.period}T',
            'months_since_last_purchase'
        ]
        self.value_col = 'Revenue'
        self.train_df = None
        self.test_df = None
        self.load_and_prepare_data()

        if self.test_mode:
            self.create_test_df()

    def _periods_diff(self, row):
        '''
        Calculate the number of periods since the last purchase for a row.
        '''
        if pd.isnull(row['last_purchase_ts']):
            return -1
        ts = row['timestamp_ts']
        last = row['last_purchase_ts']
        if self.freq == 'M':
            # Number of months
            return (ts.year - last.year) * 12 + (ts.month - last.month)
        elif self.freq == '2M':
            # Number of 2-month periods
            months = (ts.year - last.year) * 12 + (ts.month - last.month)
            return months // 2
        elif self.freq == 'W':
            # Number of weeks
            return (ts - last).days // 7
        elif self.freq == '2W':
            # Number of 2-week periods
            return (ts - last).days // 14
        else:
            raise ValueError("Unsupported frequency: choose from 'M', '2M', 'W', '2W'.")

    def load_and_prepare_data(self):
        """
        Loads and prepares data for modeling, including encoding and feature engineering.
        - If a DataFrame is provided (df), uses it
        - Otherwise, raises a ValueError.
        """
        if isinstance(self.df, pd.DataFrame):
            self.df = self.df.copy()
        else:
            raise ValueError("Input must be a pandas DataFrame or a valid CSV file path as a string.")

        # Fill NaN in engagement_t1 (last row for each user) with t0
        self.df['engagement_t1'] = self.df['engagement_t1'].fillna(self.df['engagement_t0'])
        # Encode states
        self.df['engagement_t0_code'] = self.df['engagement_t0'].map(self.state_mapping)
        self.df['engagement_t1_code'] = self.df['engagement_t1'].map(self.state_mapping)
        
        # Fill NaNs in features
        for col in [
            f'avg_revenue_past_{self.period}T',
            f'activity_freq_past_{self.period}T',
            f'revenue_volatility_past_{self.period}T']:
            self.df[col] = self.df[col].fillna(0)
        # Ensure datetime columns
        self.df['timestamp_ts'] = pd.to_datetime(self.df['timestamp_ts'])
        self.df['last_purchase_ts'] = pd.to_datetime(self.df['last_purchase_ts'])
        # Calculate months since last purchase
        self.df['periods_since_last_purchase'] = self.df.apply(self._periods_diff, axis=1)

    def split_train_test(self, train_ratio=0.8):
        '''
        Split the data into train and test sets by customer.
        '''
        train_list, test_list = [], []
        for customer_id, group in self.df.groupby('CustomerID'):
            n = len(group)
            train_end = int(n * train_ratio)
            train = group.iloc[:train_end]
            test = group.iloc[train_end:]
            train_list.append(train)
            test_list.append(test)
        self.train_df = pd.concat(train_list)
        self.test_df = pd.concat(test_list)

        return self.train_df, self.test_df

    def create_test_df(self):
        '''
        Create a test DataFrame with a subset of customers for testing purposes.
        '''
        # Subset by number of customers
        unique_customers = self.df['CustomerID'].unique()[:self.test_size]
        self.df = self.df[self.df['CustomerID'].isin(unique_customers)]

        # Fill NaN in engagement_t1 (last row for each user) with t0
        self.df['engagement_t1'] = self.df['engagement_t1'].fillna(self.df['engagement_t0'])
        # Encode states
        self.df['engagement_t0_code'] = self.df['engagement_t0'].map(self.state_mapping)
        self.df['engagement_t1_code'] = self.df['engagement_t1'].map(self.state_mapping)
        
        # Fill NaNs in features
        for col in [
            f'avg_revenue_past_{self.period}T',
            f'activity_freq_past_{self.period}T',
            f'revenue_volatility_past_{self.period}T']:
            self.df[col] = self.df[col].fillna(0)
        
        # Ensure datetime columns
        self.df['timestamp_ts'] = pd.to_datetime(self.df['timestamp_ts'])
        self.df['last_purchase_ts'] = pd.to_datetime(self.df['last_purchase_ts'])
        # Calculate periods since last purchase
        self.df['periods_since_last_purchase'] = self.df.apply(self._periods_diff, axis=1)
