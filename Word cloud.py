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
      
