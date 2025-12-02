import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine


# ---------------------------------------------------------
# LOGGING CONFIG
# ---------------------------------------------------------
logging.basicConfig(
    filename="data_log/get_vendor_summary.log",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------------------------------------
# DB CONNECTION
# ---------------------------------------------------------
connection_url = f"mysql+pymysql://{'root'}:{'8520147'}@{'localhost'}:{3306}/{'tech_classes'}"
engine = create_engine(connection_url)


# ---------------------------------------------------------
# INGEST FUNCTION
# ---------------------------------------------------------
def ingest_db(df, table_name, engine, if_exists_value='replace'):
    """Insert a DataFrame into MySQL."""
    df.to_sql(table_name, con=engine, if_exists=if_exists_value, index=False)


# ---------------------------------------------------------
# CREATE SUMMARY FUNCTION
# ---------------------------------------------------------
def create_vendor_summary(engine):
    """Merge tables and compute vendor summary."""
    
    vendor_sales_summary = pd.read_sql(
        """
        WITH freight_summary AS (
            SELECT 
                VendorNumber,
                SUM(Freight) AS freight_cost
            FROM vendor_invoice
            GROUP BY VendorNumber
        ),
        purchase_summary AS (
            SELECT 
                p.VendorNumber,
                p.VendorName,
                p.Brand,
                p.Description,
                p.PurchasePrice,
                pp.Price AS actual_price,
                pp.Volume,
                SUM(p.Quantity) AS total_purchase_quantity,
                SUM(p.Dollars) AS total_purchase_dollars
            FROM purchases p
            JOIN purchase_prices pp
                ON p.Brand = pp.Brand
            WHERE p.PurchasePrice > 0
            GROUP BY 
                p.VendorNumber, 
                p.VendorName, 
                p.Brand, 
                p.Description, 
                p.PurchasePrice, 
                pp.Price, 
                pp.Volume
        ),
        sales_summary AS (
            SELECT 
                VendorNo,
                Brand,
                SUM(SalesQuantity) AS total_sales_quantity,
                SUM(SalesDollars) AS total_sales_dollars,
                SUM(SalesPrice) AS total_sales_price,
                SUM(ExciseTax) AS total_excise_tax       
            FROM sales
            GROUP BY VendorNo, Brand
        )
        SELECT 
            ps.VendorNumber,
            ps.VendorName,
            ps.Brand,
            ps.Description,
            ps.PurchasePrice,
            ps.actual_price,
            ps.Volume,
            ps.total_purchase_quantity,
            ps.total_purchase_dollars,
            ss.total_sales_quantity,
            ss.total_sales_dollars,
            ss.total_sales_price,
            ss.total_excise_tax,
            fs.freight_cost
        FROM purchase_summary ps
        LEFT JOIN sales_summary ss
            ON ps.VendorNumber = ss.VendorNo
            AND ps.Brand = ss.Brand
        LEFT JOIN freight_summary fs
            ON ps.VendorNumber = fs.VendorNumber
        ORDER BY ps.total_purchase_dollars DESC;
        """,
        con=engine
    )
    
    return vendor_sales_summary


# ---------------------------------------------------------
# CLEANING FUNCTION
# ---------------------------------------------------------
def clean_data(df):
    """Clean and enhance the vendor summary data."""
    
    df['Volume'] = df['Volume'].astype('float64')

    df.fillna(0, inplace=True)

    df['VendorName'] = df['VendorName'].str.strip()
    df['Description'] = df['Description'].str.strip()

    df['gross_profit'] = df['total_sales_dollars'] - df['total_purchase_dollars']
    df['profit_margin'] = (df['gross_profit'] / df['total_sales_dollars']) * 100
    df['stock_turn_over'] = df['total_sales_quantity'] / df['total_purchase_quantity']
    df['sales_purchase_ratio'] = df['total_sales_dollars'] / df['total_purchase_dollars']

    # ---- FIX SQL insertion error ----
    df.replace([np.inf, -np.inf], 0, inplace=True) #if we got 'inf' value , then we convert them into zero
    df.fillna(0, inplace=True)                    # if we got 'NaN' value , then we convert them into zero

    return df


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == '__main__':
    
    logging.info('Creating vendor summary table...')
    summary_df = create_vendor_summary(engine)
    logging.info(summary_df.head())

    logging.info('Cleaning data...')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    # -----------------------------------------------------
    # Force numeric types to prevent MySQL "escape_float" error
    # -----------------------------------------------------
    clean_df = clean_df.astype({
        "total_purchase_quantity": "float64",
        "total_purchase_dollars": "float64",
        "total_sales_quantity": "float64",
        "total_sales_dollars": "float64",
        "total_sales_price": "float64",
        "total_excise_tax": "float64",
        "freight_cost": "float64",
        "Volume": "float64",
        "gross_profit": "float64",
        "profit_margin": "float64",
        "stock_turn_over": "float64",
        "sales_purchase_ratio": "float64"
    })

    # -----------------------------------------------------
    # Insert into database
    # -----------------------------------------------------
    logging.info('Inserting cleaned data into database...')
    ingest_db(clean_df, 'vendor_sales_summary', engine, if_exists_value='replace')

    logging.info('COMPLETED')

