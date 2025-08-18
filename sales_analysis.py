"""
Complete, robust sales data analysis script
- Auto-detects date and financial columns
- Safe conversions and missing-column handling
- Well-structured plotting with proper spacing
- Saves individual chart PNGs and combined overview
Author: Enhanced version
Date: 2025-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

def load_csv_with_encoding(filename):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    for enc in encodings:
        try:
            df = pd.read_csv(filename, encoding=enc)
            print(f"✓ Loaded '{filename}' with encoding {enc}")
            return df
        except FileNotFoundError:
            raise
        except Exception:
            continue
    raise ValueError("Could not read CSV with common encodings.")

def detect_date_column(df):
    # Prefer common names first
    candidates = [c for c in df.columns if c.lower() in ('order date','order_date','orderdate','date','ship date','ship_date','shipdate')]
    if candidates:
        return candidates[0]
    # Otherwise try to parse each object column and see which yields most non-na datetimes
    best_col, best_count = None, -1
    for col in df.columns:
        if df[col].dtype == object or np.issubdtype(df[col].dtype, np.number):
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce', dayfirst=False)
            non_na = parsed.notna().sum()
            if non_na > best_count and non_na > 0:
                best_col, best_count = col, non_na
    return best_col

def detect_amount_columns(df):
    pattern_keywords = ['sales','amount','revenue','price','profit','total']
    amount_cols = [c for c in df.columns if any(k in c.lower() for k in pattern_keywords)]
    # Heuristics: numeric columns that look like money but missing keywords
    if not amount_cols:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # pick columns with large ranges as candidate sales/profit
        for c in numeric:
            rng = df[c].max() - df[c].min() if df[c].notna().any() else 0
            if rng > 100:  # arbitrary threshold
                amount_cols.append(c)
        amount_cols = list(dict.fromkeys(amount_cols))  # unique preserve order
    return amount_cols

def safe_divide(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b

def save_individual_chart(fig, title, chart_type):
    """Save individual chart as PNG"""
    try:
        filename = f"chart_{chart_type.lower().replace(' ', '_').replace('-', '_')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved individual chart: {filename}")
    except Exception as e:
        print(f"⚠️ Could not save {title} chart: {e}")

def create_individual_charts(df, outputs, monthly_sales):
    """Create and save individual charts"""
    print("\nCreating individual chart files...")
    
    # Create charts directory
    if not os.path.exists('individual_charts'):
        os.makedirs('individual_charts')
    
    # Chart 1: Sales by Category
    if 'Category' in df.columns and 'Sales' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
        bars = category_sales.plot(kind='barh', color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(category_sales)], ax=ax)
        ax.set_title('Total Sales by Category', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sales ($)', fontsize=12)
        ax.set_ylabel('Category', fontsize=12)
        for i, v in enumerate(category_sales.values):
            ax.text(v + max(category_sales.values)*0.01, i, f'${v:,.0f}', va='center', fontsize=10)
        plt.tight_layout()
        fig.savefig('individual_charts/01_sales_by_category.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 2: Sales by Region
    if 'Region' in df.columns and 'Sales' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        region_sales = df.groupby('Region')['Sales'].sum()
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366'][:len(region_sales)]
        wedges, texts, autotexts = ax.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', 
                                         colors=colors, startangle=90, textprops={'fontsize': 12})
        ax.set_title('Sales Distribution by Region', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        fig.savefig('individual_charts/02_sales_by_region.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 3: Monthly Sales Trend
    if monthly_sales is not None and 'Sales' in monthly_sales.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        monthly_sales['Year_Month'] = monthly_sales['Year'].astype(str) + "-" + monthly_sales['Month'].astype(str).str.zfill(2)
        x_labels = monthly_sales['Year_Month']
        y = monthly_sales['Sales']
        ax.plot(range(len(x_labels)), y, marker='o', linewidth=3, markersize=8, color='#45B7D1')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_title('Monthly Sales Trend', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year-Month', fontsize=12)
        ax.set_ylabel('Sales ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        # Add value labels on points
        for i, v in enumerate(y):
            ax.annotate(f'${v:,.0f}', (i, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        plt.tight_layout()
        fig.savefig('individual_charts/03_monthly_sales_trend.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 4: Profit by Segment
    if 'Segment' in df.columns and 'Profit' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        segment_profit = df.groupby('Segment')['Profit'].sum()
        bars = ax.bar(segment_profit.index, segment_profit.values, 
                     color=['#45B7D1', '#96CEB4', '#FFEAA7', '#FF6B6B'][:len(segment_profit)])
        ax.set_title('Total Profit by Customer Segment', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Customer Segment', fontsize=12)
        ax.set_ylabel('Profit ($)', fontsize=12)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max(segment_profit.values)*0.02, 
                   f'${h:,.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig.savefig('individual_charts/04_profit_by_segment.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 5: Sales vs Profit Scatter
    if 'Sales' in df.columns and 'Profit' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(df['Sales'], df['Profit'], alpha=0.6, c='#45B7D1', s=50)
        ax.set_title('Sales vs Profit Correlation', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sales ($)', fontsize=12)
        ax.set_ylabel('Profit ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        corr = df['Sales'].corr(df['Profit'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), fontsize=12)
        plt.tight_layout()
        fig.savefig('individual_charts/05_sales_vs_profit.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 6: Top Sub-Categories
    subcat_col = next((c for c in df.columns if c.lower().replace(' ','') in ('subcategory','sub-category','sub_category')), None)
    if subcat_col and 'Sales' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        top_sub = df.groupby(subcat_col)['Sales'].sum().sort_values(ascending=True).tail(10)
        bars = top_sub.plot(kind='barh', ax=ax, color='#96CEB4')
        ax.set_title('Top 10 Sub-Categories by Sales', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sales ($)', fontsize=12)
        ax.set_ylabel('Sub-Category', fontsize=12)
        for i, v in enumerate(top_sub.values):
            ax.text(v + max(top_sub.values)*0.01, i, f'${v:,.0f}', va='center', fontsize=9)
        plt.tight_layout()
        fig.savefig('individual_charts/06_top_subcategories.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 7: Quarterly Sales
    if 'Quarter' in df.columns and 'Sales' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        q_sales = df.groupby('Quarter')['Sales'].sum()
        labels = [f"Q{int(i)}" for i in q_sales.index]
        bars = ax.bar(labels, q_sales.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(labels)])
        ax.set_title('Quarterly Sales Performance', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Sales ($)', fontsize=12)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max(q_sales.values)*0.02, 
                   f'${h:,.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig.savefig('individual_charts/07_quarterly_sales.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # Chart 8: Shipping Mode Distribution
    if 'Ship Mode' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ship_counts = df['Ship Mode'].value_counts()
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366'][:len(ship_counts)]
        wedges, texts, autotexts = ax.pie(ship_counts.values, labels=ship_counts.index, autopct='%1.1f%%', 
                                         colors=colors, startangle=90, pctdistance=0.85, textprops={'fontsize': 12})
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        ax.set_title('Shipping Mode Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        fig.savefig('individual_charts/08_shipping_mode.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    print("✓ All individual charts saved in 'individual_charts/' directory")

def main(filename='stores_sales_forecasting.csv'):
    print("="*60)
    print("COMPREHENSIVE SALES DATA ANALYSIS (ENHANCED VERSION)")
    print("="*60)

    # 1. Load
    try:
        df = load_csv_with_encoding(filename)
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found. Put it in the working directory.")
        sys.exit(1)
    except ValueError as e:
        print("❌", str(e))
        sys.exit(1)

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Basic preview
    print("Columns and dtypes:")
    for c in df.columns:
        print(f" - {c} : {df[c].dtype}")

    # 2. Detect and convert date column
    date_col = detect_date_column(df)
    if date_col is None:
        print("⚠️ No date-like column detected. Time-based analysis will be skipped.")
    else:
        print(f"Detected date column: '{date_col}' -> converting to datetime...")
        # Handle numeric (Excel serial) and object strings
        if np.issubdtype(df[date_col].dtype, np.number):
            # Try excel origin conversion
            df[date_col] = pd.to_datetime(df[date_col], origin='1899-12-30', unit='D', errors='coerce')
        else:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce')
        if df[date_col].isna().all():
            print(f"⚠️ Conversion produced all NaT for '{date_col}'. Time analysis will be skipped.")
            date_col = None
        else:
            # create time features
            df['Order Date'] = df[date_col]  # normalized name for further use
            df['Year'] = df['Order Date'].dt.year
            df['Month'] = df['Order Date'].dt.month
            df['Month_Name'] = df['Order Date'].dt.month_name()
            df['Quarter'] = df['Order Date'].dt.quarter
            df['Day_of_Week'] = df['Order Date'].dt.day_name()

    # 3. Detect financial columns
    amount_candidates = detect_amount_columns(df)
    # We prefer columns named 'Sales' and 'Profit' if present
    sales_col = None
    profit_col = None
    for c in df.columns:
        lc = c.lower()
        if 'sales' == lc or lc.endswith(' sales') or ' sales' in lc:
            sales_col = c
        if 'profit' == lc or lc.endswith(' profit') or ' profit' in lc:
            profit_col = c

    # fallback picks
    if not sales_col:
        sales_col = amount_candidates[0] if amount_candidates else None
    if not profit_col:
        # choose next candidate different from sales
        profit_col = amount_candidates[1] if len(amount_candidates) > 1 else None

    if not sales_col:
        print("⚠️ Could not detect a Sales column. Many analyses will be skipped.")
    else:
        df['Sales'] = pd.to_numeric(df[sales_col], errors='coerce')

    if not profit_col:
        print("⚠️ Could not detect a Profit column. Profit-based analyses will be skipped.")
    else:
        df['Profit'] = pd.to_numeric(df[profit_col], errors='coerce')

    # Ensure Quantity exists or create a fallback
    if 'Quantity' not in df.columns:
        numeric_candidates = df.select_dtypes(include=[np.integer, np.number]).columns.tolist()
        q_col = next((c for c in numeric_candidates if 'qty' in c.lower() or 'quantity' in c.lower()), None)
        if q_col:
            df['Quantity'] = pd.to_numeric(df[q_col], errors='coerce').fillna(0).astype(int)
            print(f"Using '{q_col}' as Quantity.")
        else:
            df['Quantity'] = 0
            print("No quantity-like column found; setting Quantity = 0 for all rows.")

    # Create safe Profit_Margin if both Sales and Profit exist
    if 'Profit' in df.columns and 'Sales' in df.columns:
        df['Profit_Margin'] = df.apply(
            lambda row: safe_divide(row['Profit'], row['Sales']) * 100 if pd.notna(row['Sales']) else np.nan,
            axis=1
        )
    else:
        df['Profit_Margin'] = np.nan

    # Show missing value summary (top)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values (top 10):")
        print(missing[missing > 0].sort_values(ascending=False).head(10))
    else:
        print("\n✓ No missing values detected.")

    # 4. Basic stats (guarded)
    print("\nKEY METRICS:")
    if 'Sales' in df.columns:
        print(f"- Average Order Value: ${df['Sales'].mean():.2f}")
        print(f"- Median Order Value: ${df['Sales'].median():.2f}")
    else:
        print("- Sales data not available")

    if 'Profit' in df.columns:
        print(f"- Average Profit per Order: ${df['Profit'].mean():.2f}")
    else:
        print("- Profit data not available")

    if 'Profit_Margin' in df.columns and df['Profit_Margin'].notna().any():
        print(f"- Average Profit Margin: {df['Profit_Margin'].mean():.2f}%")
    else:
        print("- Profit margin not available")

    # Unique counts if columns present
    if 'Customer ID' in df.columns:
        print(f"- Total Customers: {df['Customer ID'].nunique():,}")
    if 'Product ID' in df.columns:
        print(f"- Total Products: {df['Product ID'].nunique():,}")
    if 'Category' in df.columns:
        print(f"- Number of Categories: {df['Category'].nunique()}")
    if 'Region' in df.columns:
        print(f"- Number of Regions: {df['Region'].nunique()}")

    # 5. Grouped analyses (only if columns exist)
    print("\nDETAILED ANALYSIS:")

    outputs = {}  # store dataframes to export later

    # Sales by Category
    if 'Category' in df.columns and 'Sales' in df.columns:
        cat = df.groupby('Category').agg(
            Total_Sales=('Sales','sum'),
            Avg_Sales=('Sales','mean'),
            Order_Count=('Sales','count'),
            Total_Profit=('Profit','sum') if 'Profit' in df.columns else pd.NamedAgg(column='Sales', aggfunc='sum'),
            Avg_Profit=('Profit','mean') if 'Profit' in df.columns else pd.NamedAgg(column='Sales', aggfunc='mean'),
            Total_Quantity=('Quantity','sum')
        ).round(2).sort_values('Total_Sales', ascending=False)
        print("\nTop Categories by Sales:")
        print(cat.head())
        outputs['Category Analysis'] = cat
    else:
        print(" - Skipped category analysis (missing Category or Sales column)")

    # Sales by Region
    if 'Region' in df.columns and 'Sales' in df.columns:
        region = df.groupby('Region').agg(
            Total_Sales=('Sales','sum'),
            Avg_Sales=('Sales','mean'),
            Total_Profit=('Profit','sum') if 'Profit' in df.columns else np.nan,
            Avg_Profit=('Profit','mean') if 'Profit' in df.columns else np.nan,
            Unique_Customers=('Customer ID','nunique') if 'Customer ID' in df.columns else np.nan
        ).round(2).sort_values('Total_Sales', ascending=False)
        print("\nSales by Region:")
        print(region.head())
        outputs['Region Analysis'] = region
    else:
        print(" - Skipped region analysis (missing Region or Sales)")

    # Monthly Sales Trend (requires Order Date)
    if date_col:
        monthly_sales = df.groupby(['Year','Month']).agg(
            Sales=('Sales','sum') if 'Sales' in df.columns else pd.NamedAgg(column=df.columns[0], aggfunc='count'),
            Profit=('Profit','sum') if 'Profit' in df.columns else np.nan,
            Orders=('Order Date','count')
        ).round(2)
        # Create a clean month index for plotting (Year-Month)
        monthly_sales = monthly_sales.reset_index().sort_values(['Year','Month'])
        print("\nMonthly sales snapshot:")
        print(monthly_sales.head())
        outputs['Monthly Sales'] = monthly_sales
    else:
        monthly_sales = None
        print(" - Skipped monthly trend (no valid date column)")

    # Segment analysis
    if 'Segment' in df.columns and 'Sales' in df.columns:
        seg = df.groupby('Segment').agg(
            Total_Sales=('Sales','sum'),
            Avg_Sales=('Sales','mean'),
            Total_Profit=('Profit','sum') if 'Profit' in df.columns else np.nan,
            Avg_Profit=('Profit','mean') if 'Profit' in df.columns else np.nan,
            Customer_Count=('Customer ID','nunique') if 'Customer ID' in df.columns else np.nan
        ).round(2).sort_values('Total_Sales', ascending=False)
        print("\nSegment analysis:")
        print(seg)
        outputs['Segment Analysis'] = seg
    else:
        print(" - Skipped segment analysis")

    # Top products by sales
    if 'Product Name' in df.columns and 'Sales' in df.columns:
        top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
        print("\nTop 10 Products by Sales:")
        print(top_products)
        outputs['Top Products'] = top_products
    else:
        print(" - Skipped top products")

    # Shipping mode
    if 'Ship Mode' in df.columns:
        shipping = df.groupby('Ship Mode').agg(
            Total_Sales=('Sales','sum') if 'Sales' in df.columns else 'size',
            Avg_Sales=('Sales','mean') if 'Sales' in df.columns else np.nan,
            Order_Count=('Order Date','count') if 'Order Date' in df.columns else 'size'
        ).round(2).sort_values('Total_Sales', ascending=False)
        print("\nShipping mode summary:")
        print(shipping)
        outputs['Shipping Mode'] = shipping
    else:
        print(" - No Ship Mode column")

    # Profitability analysis per product
    if 'Product Name' in df.columns:
        aggreg = {
            'Sales':'sum'
        }
        if 'Profit' in df.columns:
            aggreg['Profit'] = 'sum'
            aggreg['Profit_Margin'] = 'mean'
        aggreg['Quantity'] = 'sum'
        profit_analysis = df.groupby('Product Name').agg(aggreg).round(2)
        profit_analysis = profit_analysis.sort_values('Profit' if 'Profit' in profit_analysis.columns else 'Sales', ascending=False)
        print("\nProduct profitability snapshot (top 5):")
        print(profit_analysis.head(5))
        outputs['Product Profitability'] = profit_analysis
    else:
        print(" - No Product Name column for profitability analysis")

    # State-wise
    if 'State' in df.columns and 'Sales' in df.columns:
        state_perf = df.groupby('State').agg(
            Total_Sales=('Sales','sum'),
            Total_Profit=('Profit','sum') if 'Profit' in df.columns else np.nan,
            Order_Count=('Order Date','count') if 'Order Date' in df.columns else 'size'
        ).round(2).sort_values('Total_Sales', ascending=False)
        outputs['State Performance'] = state_perf
        print("\nTop states by sales:")
        print(state_perf.head(10))
    else:
        print(" - Skipped state-wise performance")

    # 6. Create individual chart files first
    create_individual_charts(df, outputs, monthly_sales)

    # 7. Create comprehensive dashboard with proper spacing
    print("\nCREATING COMPREHENSIVE DASHBOARD...")
    
    # Calculate how many charts we can create
    chart_count = 0
    chart_funcs = []
    
    # Define chart creation functions
    def create_category_chart(ax):
        if 'Category' in df.columns and 'Sales' in df.columns:
            category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
            bars = category_sales.plot(kind='barh', color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(category_sales)], ax=ax)
            ax.set_title('Sales by Category', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Sales ($)', fontsize=11)
            ax.set_ylabel('Category', fontsize=11)
            ax.tick_params(axis='both', labelsize=10)
            for i, v in enumerate(category_sales.values):
                ax.text(v + max(category_sales.values)*0.01, i, f'${v/1000:.0f}K', va='center', fontsize=9)
            return True
        return False

    def create_region_chart(ax):
        if 'Region' in df.columns and 'Sales' in df.columns:
            region_sales = df.groupby('Region')['Sales'].sum()
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366'][:len(region_sales)]
            wedges, texts, autotexts = ax.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', 
                                             colors=colors, startangle=90, textprops={'fontsize': 10})
            ax.set_title('Sales by Region', fontsize=14, fontweight='bold', pad=15)
            return True
        return False

    def create_monthly_trend_chart(ax):
        if monthly_sales is not None and 'Sales' in monthly_sales.columns:
            monthly_sales['Year_Month'] = monthly_sales['Year'].astype(str) + "-" + monthly_sales['Month'].astype(str).str.zfill(2)
            x_labels = monthly_sales['Year_Month']
            y = monthly_sales['Sales']
            ax.plot(range(len(x_labels)), y, marker='o', linewidth=2.5, markersize=5, color='#45B7D1')
            # Show fewer labels to avoid crowding
            step = max(1, len(x_labels)//6)
            ax.set_xticks(range(0, len(x_labels), step))
            ax.set_xticklabels([x_labels.iloc[i] for i in range(0, len(x_labels), step)], rotation=45, fontsize=9)
            ax.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Year-Month', fontsize=11)
            ax.set_ylabel('Sales ($)', fontsize=11)
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(True, alpha=0.3)
            return True
        return False

    def create_segment_profit_chart(ax):
        if 'Segment' in df.columns and 'Profit' in df.columns:
            segment_profit = df.groupby('Segment')['Profit'].sum()
            bars = ax.bar(segment_profit.index, segment_profit.values, 
                         color=['#45B7D1', '#96CEB4', '#FFEAA7', '#FF6B6B'][:len(segment_profit)])
            ax.set_title('Profit by Segment', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Segment', fontsize=11)
            ax.set_ylabel('Profit ($)', fontsize=11)
            ax.tick_params(axis='x', rotation=30, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + max(segment_profit.values)*0.02, 
                       f'${h/1000:.0f}K', ha='center', va='bottom', fontsize=9)
            return True
        return False

    def create_scatter_chart(ax):
        if 'Sales' in df.columns and 'Profit' in df.columns:
            sample_size = min(1000, len(df))  # Limit points for clarity
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            scatter = ax.scatter(sample_df['Sales'], sample_df['Profit'], alpha=0.6, c='#45B7D1', s=25)
            ax.set_title('Sales vs Profit', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Sales ($)', fontsize=11)
            ax.set_ylabel('Profit ($)', fontsize=11)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, alpha=0.3)
            corr = df['Sales'].corr(df['Profit'])
            ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round", facecolor='white', alpha=0.8), fontsize=10)
            return True
        return False

    def create_subcategory_chart(ax):
        subcat_col = next((c for c in df.columns if c.lower().replace(' ','') in ('subcategory','sub-category','sub_category')), None)
        if subcat_col and 'Sales' in df.columns:
            top_sub = df.groupby(subcat_col)['Sales'].sum().sort_values(ascending=True).tail(6)  # Top 6 for better spacing
            bars = top_sub.plot(kind='barh', ax=ax, color='#96CEB4')
            ax.set_title('Top Sub-Categories', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Sales ($)', fontsize=11)
            ax.tick_params(axis='both', labelsize=9)
            for i, v in enumerate(top_sub.values):
                ax.text(v + max(top_sub.values)*0.01, i, f'${v/1000:.0f}K', va='center', fontsize=8)
            return True
        return False

    def create_quarterly_chart(ax):
        if 'Quarter' in df.columns and 'Sales' in df.columns:
            q_sales = df.groupby('Quarter')['Sales'].sum()
            labels = [f"Q{int(i)}" for i in q_sales.index]
            bars = ax.bar(labels, q_sales.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(labels)])
            ax.set_title('Quarterly Sales', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Quarter', fontsize=11)
            ax.set_ylabel('Sales ($)', fontsize=11)
            ax.tick_params(axis='both', labelsize=10)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + max(q_sales.values)*0.02, 
                       f'${h/1000:.0f}K', ha='center', va='bottom', fontsize=9)
            return True
        return False

    def create_shipping_chart(ax):
        if 'Ship Mode' in df.columns:
            ship_counts = df['Ship Mode'].value_counts()
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366'][:len(ship_counts)]
            wedges, texts, autotexts = ax.pie(ship_counts.values, labels=ship_counts.index, autopct='%1.1f%%', 
                                             colors=colors, startangle=90, pctdistance=0.85, textprops={'fontsize': 10})
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            ax.set_title('Shipping Mode', fontsize=14, fontweight='bold', pad=15)
            return True
        return False

    # Collect available charts
    available_charts = [
        create_category_chart,
        create_region_chart,
        create_monthly_trend_chart,
        create_segment_profit_chart,
        create_scatter_chart,
        create_subcategory_chart,
        create_quarterly_chart,
        create_shipping_chart
    ]

    # Create the comprehensive dashboard with much larger size and better spacing
    fig = plt.figure(figsize=(24, 20))  # Increased size significantly
    fig.suptitle('Sales Analytics Dashboard', fontsize=22, fontweight='bold', y=0.97)
    
    # Use 4x2 layout with better positioning
    positions = [
        (4, 2, 1), (4, 2, 2),  # Row 1
        (4, 2, 3), (4, 2, 4),  # Row 2
        (4, 2, 5), (4, 2, 6),  # Row 3
        (4, 2, 7), (4, 2, 8),  # Row 4
    ]
    
    chart_idx = 0
    created_axes = []
    
    for chart_func in available_charts:
        if chart_idx >= len(positions):
            break
        
        ax = plt.subplot(*positions[chart_idx])
        if chart_func(ax):
            created_axes.append(ax)
            chart_idx += 1
        else:
            # Remove empty subplot
            fig.delaxes(ax)
    
    # Much more aggressive spacing adjustments
    plt.subplots_adjust(
        left=0.08,      # Left margin
        bottom=0.06,    # Bottom margin  
        right=0.96,     # Right margin
        top=0.94,       # Top margin (leave space for title)
        wspace=0.35,    # Width spacing between subplots
        hspace=0.55     # Height spacing between subplots (increased significantly)
    )
    
    try:
        plt.savefig('sales_dashboard_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("✓ Comprehensive dashboard saved as 'sales_dashboard_comprehensive.png'")
    except Exception as e:
        print("⚠️ Could not save dashboard:", e)
    
    plt.show()

    # 8. Advanced insights printouts
    print("\nADVANCED INSIGHTS:")
    if 'Product Profitability' in outputs:
        profit_df = outputs['Product Profitability']
        if 'Profit' in profit_df.columns:
            print("\nTop 5 most profitable products:")
            print(profit_df.sort_values('Profit', ascending=False).head(5))
        # loss-making products
        if 'Profit' in profit_df.columns and (profit_df['Profit'] < 0).any():
            print("\nTop 5 loss-making products:")
            print(profit_df[profit_df['Profit'] < 0].sort_values('Profit').head(5))

    # Top customers
    if 'Customer Name' in df.columns and 'Sales' in df.columns:
        cust = df.groupby('Customer Name').agg(Total_Spent=('Sales','sum'), Total_Profit_Generated=('Profit','sum') if 'Profit' in df.columns else np.nan, Order_Count=('Order Date','count') if 'Order Date' in df.columns else 'size').round(2)
        cust_sorted = cust.sort_values('Total_Spent', ascending=False)
        print("\nTop 10 customers by sales:")
        print(cust_sorted.head(10))
        outputs['Top Customers'] = cust_sorted

    # 9. Executive summary
    total_sales = df['Sales'].sum() if 'Sales' in df.columns else np.nan
    total_profit = df['Profit'].sum() if 'Profit' in df.columns else np.nan
    overall_margin = (safe_divide(total_profit, total_sales) * 100) if pd.notna(total_sales) else np.nan

    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    print(f"Total Sales Revenue: ${total_sales:,.2f}" if pd.notna(total_sales) else "Total Sales: N/A")
    print(f"Total Profit: ${total_profit:,.2f}" if pd.notna(total_profit) else "Total Profit: N/A")
    print(f"Overall Profit Margin: {overall_margin:.2f}%" if pd.notna(overall_margin) else "Overall Margin: N/A")
    if 'Category Analysis' in outputs:
        best_cat = outputs['Category Analysis'].idxmax()['Total_Sales']
        print(f"Best Category: {best_cat} (${outputs['Category Analysis'].loc[best_cat,'Total_Sales']:,.2f})")
    if 'Region Analysis' in outputs:
        best_reg = outputs['Region Analysis'].idxmax()['Total_Sales']
        print(f"Best Region: {best_reg} (${outputs['Region Analysis'].loc[best_reg,'Total_Sales']:,.2f})")
    if 'Top Customers' in outputs:
        best_cust = outputs['Top Customers'].index[0]
        print(f"Best Customer: {best_cust} (${outputs['Top Customers'].iloc[0]['Total_Spent']:,.2f})")
    if monthly_sales is not None:
        best_month = monthly_sales.loc[monthly_sales['Sales'].idxmax()]
        print(f"Best Month: {best_month['Year']}-{int(best_month['Month']):02d} (${best_month['Sales']:,.2f})")

    # 10. Save only the consolidated Excel report (NO individual CSV files)
    try:
        with pd.ExcelWriter('sales_analysis_results.xlsx', engine='openpyxl') as writer:
            for sheet, df_out in outputs.items():
                # convert Series to df
                if isinstance(df_out, pd.Series):
                    df_out.to_frame(name='Value').to_excel(writer, sheet_name=sheet[:31])
                else:
                    df_out.to_excel(writer, sheet_name=sheet[:31])
            # Also save top customers if present
            if 'Top Customers' in outputs:
                outputs['Top Customers'].head(200).to_excel(writer, sheet_name='Top Customers')
        print("✓ Detailed results saved to 'sales_analysis_results.xlsx'")
        print("✓ NO individual CSV files created - everything consolidated in Excel!")
    except Exception as e:
        print("⚠️ Could not save Excel file (openpyxl missing?). Error:", e)
        print("Please install openpyxl: pip install openpyxl")

    print("\n✅ Analysis Complete!")
    print(f"📊 Generated files:")
    print(f"   - sales_dashboard_comprehensive.png (Main dashboard)")
    print(f"   - individual_charts/ directory (8 separate PNG files)")
    print(f"   - sales_analysis_results.xlsx (Data analysis)")

if __name__ == "__main__":
    # optionally allow passing filename as first CLI arg
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else 'stores_sales_forecasting.csv'
    main(fname)