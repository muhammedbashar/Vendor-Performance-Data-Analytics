import pandas as pd
import numpy as np
import time 
from sqlalchemy import create_engine
from sqlalchemy import text

# creating connection to MySQL
engine = create_engine("mysql+pymysql://root:8520147@localhost:3306/tech_classes")


# checking tables present in the MySQL database
tables = pd.read_sql("SHOW TABLES;", con=engine)

print(tables)

 -- >for t in tables.iloc[:,0]:   # for look in our tables first column
     -- >print('-'*10, f'{t}','-'*10)
     -- >query = f"SELECT COUNT(*) AS Total_rows FROM {t}"
     -- >count_df = pd.read_sql(query,con=engine)
     -- >print('Rows:',count_df.iloc[0,0]) # for get the output in LEFT most side
    
     -- >print(pd.read_sql(f'SELECT * FROM {t} LIMIT 5', con=engine)) # extract each tabel first 5 columns to get to know about the every column in each tables

# lest explore each table deeply (mandetory) 

purchases = pd.read_sql(f'SELECT * FROM purchases WHERE VendorNumber = 4466',con=engine)
#print(purchases)

purchase_prices = pd.read_sql("SELECT * FROM purchase_prices WHERE VendorNumber = 4466",con=engine)
#print(purchase_prices)

vendor_invoice = pd.read_sql("SELECT * FROM vendor_invoice WHERE VendorNumber = 4466",con=engine)
#print(vendor_invoice)

sales = pd.read_sql("SELECT * FROM sales WHERE VendorNo = 4466",con=engine)
#print(sales) 

# group by 
print(purchases.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum())

#print(vendor_invoice.columns)
#print(vendor_invoice['PONumber'].nunique()) # total unique 'ponumber' rows

#print(f'\n{sales.columns}')
print(sales.groupby('Brand')[['SalesDollars','SalesPrice','SalesQuantity']].sum())

""" 
    -> The purchases tables contains actual purchase data, include the date of purchase, products(brands) purchased by vendors, the amount paid (in dollars), and the quantity purchased
    -> The purchase price column is derived from purchase_priced table, which provides product-wise actual and purchase prices. The combination of vendor and brand is unique in this table.
    -> The vendor_invoice table aggregates data from the purchases table, summarizing quantity and dollar amounts, along with an additional columns for freight.
        This table maintains uniqueness based on vendor and PO number.
    -> The sales table captures actual sales trancsations, detailing the brands purchased by vendors, the quantity sold, the selling price, and the revenue earned.

"""
""" 
    As the data that we need for analysis is distributed in different tables, we need to create a summary table containing:
        => purchase transactions made by vendors
        => sales transaction data
        => freight costs for each vendor  (>>>>Freight cost typically includes charges for transportation, fuel, handling, insurance, loading/unloading, and documentation fees)
        => actual product prices from vendors
"""

print()

# check all columns for vendor_invoice table
print(vendor_invoice.columns)
print()
frieght_summary = pd.read_sql("""SELECT VendorNumber,SUM(Freight) as Frieght_cost 
                                 FROM vendor_invoice 
                                 GROUP BY VendorNumber""",con=engine)
print(frieght_summary)
print()

# purchase details
print(f"purchase_price table columns are :\n{purchase_prices.columns}") # extract all column names in 'purchase_prices' tables
print(f"purcahse table columns are :\n {purchases.columns}")    # extract all column names in 'purchase' tables

print(pd.read_sql("""
            SELECT 
                p.VendorNumber,
                p.VendorName,
                p.Brand,
                p.PurchasePrice,
                pp.Volume,
                pp.Price AS actual_price,
                SUM(p.Quantity) AS total_purchase_quantity,
                SUM(p.Dollars) AS total_purchase_dollars
            FROM purchases AS p
            JOIN purchase_prices AS pp
                ON p.Brand = pp.Brand
            WHERE p.PurchasePrice > 0
            GROUP BY 
                p.VendorNumber,
                p.VendorName,
                p.Brand,
                p.PurchasePrice,
                pp.Volume,
                pp.Price
            ORDER BY total_purchase_dollars DESC;
            """,con=engine))


# sales details
print(sales.columns)
print()
print(pd.read_sql("""SELECT 
                  VendorNo,
                  Brand,
                  SUM(SalesDollars) as total_sales_dollars,
                  SUM(SalesPrice) as total_sales_price,
                  SUM(SalesQuantity) as total_sales_quantity,
                  SUM(ExciseTax) as total_excise_tax
                  FROM sales
                  GROUP BY VendorNo,Brand
                  ORDER BY total_sales_dollars
                  """,con=engine))

print()



# join every table and make one fully structured table
vendor_sales_summary = pd.read_sql("""
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
""", con=engine)

print(vendor_sales_summary)
""" 
this query genereates a vender wise sales and purchase summary, which is valuable for :
    PERFOMANCE OPTIMISATION :   
        => the query involves heavy joins and aggregations on large datasets like sales and purchases.
        => storing the pre-aggregated results avoids repeated expensive computations.
        => Helps in analysing sales, purchases, and pricing for different vendors and brands.
        => future benefits of storing this data for faster dashboarding & reporting.
        => instead of running expensive queries each time, dashboards can fetch data quickly from 'vendor_sales_summary'.
"""



print(vendor_sales_summary.dtypes)
print()
print(vendor_sales_summary.isnull().sum())
print()
print(vendor_sales_summary['VendorName'].unique())
print()
print(vendor_sales_summary['Description'].unique())
print()

vendor_sales_summary['Volume'] = vendor_sales_summary['Volume'].astype('float64')  # convert datatype into flaat value
vendor_sales_summary.fillna(0, inplace=True) #change all null value to zero
vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()  # trim all insufficient spaces

#check again to make sure that we cleaned right way
print(vendor_sales_summary.isnull().sum())
print()
print(vendor_sales_summary['VendorName'].unique())
print()
print(vendor_sales_summary['Volume'].dtype)
print()


# lets create some new columns 
# 1- column name 'gross_profit'
vendor_sales_summary['gross_profit'] = vendor_sales_summary['total_sales_dollars'] - vendor_sales_summary['total_purchase_dollars']
print(vendor_sales_summary['gross_profit'].min()) # to check our minimum gross profit and did we create perfectly a new columns called 'gross_profit'

# 2- column name 'profit_margin'
vendor_sales_summary['profit_margin'] = (vendor_sales_summary['gross_profit'] / vendor_sales_summary['total_sales_dollars'])*100

# 3- columns name 'stock_turn_over'
vendor_sales_summary['stock_turn_over'] = vendor_sales_summary['total_sales_quantity'] / vendor_sales_summary['total_purchase_quantity']

# 4- columns name 'sales_purchase_ratio'
vendor_sales_summary['sales_purchase_ratio'] = vendor_sales_summary['total_sales_dollars'] / vendor_sales_summary['total_purchase_dollars']



# lets create an empty new whole table and then insert the whole columns yet we got into that table 
# like in sql, we do truncate we insert values into it

# -> create an empty table
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS vendor_sales_summaryy (
            vendor_number INT,
            vendor_name VARCHAR(100),
            brand VARCHAR(100),
            description VARCHAR(255),
            purchase_price DECIMAL(10,2),
            actual_price DECIMAL(10,2),
            volume INT,
            total_purchase_quantity INT,
            total_purchase_dollars DECIMAL(15,2),
            total_sales_quantity INT,
            total_sales_dollars DECIMAL(15,2),
            total_sales_price DECIMAL(15,2),
            total_excise_tax DECIMAL(15,2),
            freight_cost DECIMAL(15,2),
            gross_profit DECIMAL(15,2),
            profit_margin DECIMAL(10,4),
            stock_turn_over DECIMAL(10,4),
            sales_to_purchase_ratio DECIMAL(10,4),
            PRIMARY KEY (vendor_number, brand)
        )
    """))



print()
print(pd.read_sql("""SELECT * FROM vendor_sales_summaryy""",con=engine))

# ---- FIX SQL insertion error ----
vendor_sales_summary.replace([np.inf, -np.inf], 0, inplace=True) #if we got 'inf' value , then we convert them into zero
vendor_sales_summary.fillna(0, inplace=True)                    # if we got 'NaN' value , then we convert them into zero

vendor_sales_summary.to_sql(
    'vendor_sales_summaryy',
    con=engine,
    if_exists='append',
    index=False
)
print()
print(pd.read_sql("""SELECT * FROM vendor_sales_summaryy""",con=engine))
