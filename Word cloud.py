import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the Excel file
file_path = "your_file.xlsx"  # Update with your file path
df = pd.read_excel(file_path)

# Ensure required columns exist
required_columns = {"comp_name", "cleaned_sent", "sentiments"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing required columns. Expected: {required_columns}")

# Generate word clouds for each company
companies = df["comp_name"].unique()
for company in companies:
    company_text = " ".join(df[df["comp_name"] == company]["cleaned_sent"].dropna())
    if company_text:  # Avoid empty text cases
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(company_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {company}")
        plt.show()

# Generate word clouds for each sentiment category
sentiments = df["sentiments"].unique()
for sentiment in sentiments:
    sentiment_text = " ".join(df[df["sentiments"] == sentiment]["cleaned_sent"].dropna())
    if sentiment_text:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(sentiment_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Sentiment: {sentiment}")
        plt.show()
      




from sqlalchemy import create_engine, Table, MetaData
from datetime import datetime

# Database connection
DATABASE_URL = "oracle+cx_oracle://username:password@host:port/service_name"
engine = create_engine(DATABASE_URL)

# Reflect the existing table
metadata = MetaData()
metadata.reflect(bind=engine)

# Reference the existing table
executive_briefings = metadata.tables["executive_briefings"]  # Use actual table name

# Insert data
insert_data = [
    {
        "comp_name": "TechCorp", 
        "pub_date": datetime.now().strftime("%Y-%m-%d"),  # Store date as string (VARCHAR2)
        "executive_names": "John Doe", 
        "briefings": "Company expansion plan discussion.",  # CLOB data
        "inserted_by": "Admin"
    },
    {
        "comp_name": "InnovateX", 
        "pub_date": datetime.now().strftime("%Y-%m-%d"),  
        "executive_names": "Jane Smith", 
        "briefings": "AI-driven innovation strategies.", 
        "inserted_by": "Admin"
    },
]

# Execute insert
with engine.connect() as connection:
    connection.execute(executive_briefings.insert(), insert_data)
    connection.commit()

print("Data inserted successfully!")
