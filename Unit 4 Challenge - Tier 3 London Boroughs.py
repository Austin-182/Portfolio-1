#!/usr/bin/env python
# coding: utf-8

# # Springboard Data Science Career Track Unit 4 Challenge - Tier 3 Complete
# 
# ## Objectives
# Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time. 
# 
# In this notebook, you're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the *slightly* messier work that data scientists do with actual datasets!
# 
# Here’s the mystery we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***
# 
# 
# A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London [(here's some info for the curious)](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.
# 
# ***This is the Tier 3 notebook, which means it's not filled in at all: we'll just give you the skeleton of a project, the brief and the data. It's up to you to play around with it and see what you can find out! Good luck! If you struggle, feel free to look at easier tiers for help; but try to dip in and out of them, as the more independent work you do, the better it is for your learning!***
# 
# This challenge will make use of only what you learned in the following DataCamp courses: 
# - Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
# - Data Types for Data Science
# - Python Data Science Toolbox (Part One) 
# - pandas Foundations
# - Manipulating DataFrames with pandas
# - Merging DataFrames with pandas
# 
# Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following: 
# - **pandas**
#     - **data ingestion and inspection** (pandas Foundations, Module One) 
#     - **exploratory data analysis** (pandas Foundations, Module Two)
#     - **tidying and cleaning** (Manipulating DataFrames with pandas, Module Three) 
#     - **transforming DataFrames** (Manipulating DataFrames with pandas, Module One)
#     - **subsetting DataFrames with lists** (Manipulating DataFrames with pandas, Module One) 
#     - **filtering DataFrames** (Manipulating DataFrames with pandas, Module One) 
#     - **grouping data** (Manipulating DataFrames with pandas, Module Four) 
#     - **melting data** (Manipulating DataFrames with pandas, Module Three) 
#     - **advanced indexing** (Manipulating DataFrames with pandas, Module Four) 
# - **matplotlib** (Intermediate Python for Data Science, Module One)
# - **fundamental data types** (Data Types for Data Science, Module One) 
# - **dictionaries** (Intermediate Python for Data Science, Module Two)
# - **handling dates and times** (Data Types for Data Science, Module Four)
# - **function definition** (Python Data Science Toolbox - Part One, Module One)
# - **default arguments, variable length, and scope** (Python Data Science Toolbox - Part One, Module Two) 
# - **lambda functions and error handling** (Python Data Science Toolbox - Part One, Module Four) 

# ## The Data Science Pipeline
# 
# This is Tier Three, so we'll get you started. But after that, it's all in your hands! When you feel done with your investigations, look back over what you've accomplished, and prepare a quick presentation of your findings for the next mentor meeting. 
# 
# Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as  [David Spiegelhalter](https://www.youtube.com/watch?v=oUs1uvsz0Ok) reminds us, there is no substitute for simply **taking a really, really good look at the data.** Sometimes, this is all we need to answer our question.
# 
# Data Science projects generally adhere to the four stages of Data Science Pipeline:
# 1. Sourcing and loading 
# 2. Cleaning, transforming, and visualizing 
# 3. Modeling 
# 4. Evaluating and concluding 
# 

# ### 1. Sourcing and Loading 
# 
# Any Data Science project kicks off by importing  ***pandas***. The documentation of this wonderful library can be found [here](https://pandas.pydata.org/). As you've seen, pandas is conveniently connected to the [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) libraries. 
# 
# ***Hint:*** This part of the data science pipeline will test those skills you acquired in the pandas Foundations course, Module One. 

# #### 1.1. Importing Libraries

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# #### 1.2.  Loading the data
# Your data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal for London-oriented datasets. 

# In[2]:


# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices = "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)


# ### 2. Cleaning, transforming, and visualizing
# This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.
# 
# The end goal of data cleaning is to have tidy data. When data is tidy: 
# 
# 1. Each variable has a column.
# 2. Each observation forms a row.
# 
# Keep the end goal in mind as you move through this process, every step will take you closer. 
# 
# 
# 
# ***Hint:*** This part of the data science pipeline should test those skills you acquired in: 
# - Intermediate Python for data science, all modules.
# - pandas Foundations, all modules. 
# - Manipulating DataFrames with pandas, all modules.
# - Data Types for Data Science, Module Four.
# - Python Data Science Toolbox - Part One, all modules

# **2.1. Exploring your data** 
# 
# Think about your pandas functions for checking out a dataframe. 

# In[3]:


properties.head()


# **2.2. Cleaning the data**
# 
# You might find you need to transpose your dataframe, check out what its row indexes are, and reset the index. You  also might find you need to assign the values of the first row to your column headings  . (Hint: recall the .columns feature of DataFrames, as well as the iloc[] method).
# 
# Don't be afraid to use StackOverflow for help  with this.

# In[4]:


# Transpose properties
t_properties = properties.transpose()
t_properties.head()


# **2.3. Cleaning the data (part 2)**
# 
# You might we have to **rename** a couple columns. How do you do this? The clue's pretty bold...

# In[5]:


# Reset properties index
t_properties = t_properties.reset_index()


# In[6]:


# Move the rows 
t_properties.columns = t_properties.iloc[0]
t_properties.head()


# In[7]:


# Correct the axis
t_properties= t_properties.drop([0],axis=0)


# In[8]:


t_properties.head()


# In[9]:


# Fix the names of two columns
t_properties.rename(columns={"Unnamed: 0": "Borough Name", pd.NaT: "City ID"}, inplace=True)
t_properties.head()


# **2.4.Transforming the data**
# 
# Remember what Wes McKinney said about tidy data? 
# 
# You might need to **melt** your DataFrame here. 

# In[10]:


# Extract date columns 
date_columns = t_properties.columns[2:]

# Convert year format 
formatted_dates = [pd.to_datetime(date).strftime('%Y-%b') for date in date_columns]
t_properties.columns = ['Borough Name', 'City ID'] + formatted_dates


t_properties_melted = t_properties.melt(id_vars=['Borough Name', 'City ID'], 
                                       value_vars=formatted_dates,
                                       var_name='Date',
                                       value_name='Average Price')


t_properties.head()



# In[11]:


columns_level_0 = list(t_properties.columns[:2]) + [date for date in formatted_dates if pd.notna(date)]
columns_level_1 = [''] * 2 + ['Ave-Price'] * (len(t_properties.columns) - 2)

t_properties.columns = pd.MultiIndex.from_tuples(list(zip(columns_level_0, columns_level_1)))

t_properties.head()


# Remember to make sure your column data types are all correct. Average prices, for example, should be floating point numbers... 

# **2.5. Cleaning the data (part 3)**
# 
# Do we have an equal number of observations in the ID, Average Price, Month, and London Borough columns? Remember that there are only 32 London Boroughs. How many entries do you have in that column? 
# 
# Check out the contents of the London Borough column, and if you find null values, get rid of them however you see fit. 

# In[12]:


# Valid boroughs
valid_boroughs = [
    'Barking & Dagenham', 'Barnet', 'Bexley', 'Brent', 'Bromley',
    'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich', 'Hackney',
    'Hammersmith & Fulham', 'Haringey', 'Harrow', 'Havering', 'Hillingdon',
    'Hounslow', 'Islington', 'Kensington & Chelsea', 'Kingston upon Thames',
    'Lambeth', 'Lewisham', 'Merton', 'Newham', 'Redbridge', 'Richmond upon Thames',
    'Southwark', 'Sutton', 'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 
    'Westminster'
]

t_properties_filtered = t_properties_melted[t_properties_melted['Borough Name'].isin(valid_boroughs)]


# In[119]:


# Double checking to see if valid_boroughs works
missing_ids = t_properties_filtered['City ID'].isnull().sum()
missing_prices = t_properties_filtered['Average Price'].isnull().sum()

print(missing_ids)
print(missing_prices)


# **2.6. Visualizing the data**
# 
# To visualize the data, why not subset on a particular London Borough? Maybe do a line plot of Month against Average Price?

# In[112]:


t_properties_filtered.head(10)


# In[113]:


camden_prices = t_properties_filtered[t_properties_filtered['Borough Name'] == 'Camden']
ax = camden_prices.plot(kind='line', x='Date', y='Average Price', figsize=(15,10))
ax.set_ylabel('Price')
ax.set_title('Camden Average Price (£) Over the past 30 years')
plt.grid(True)
ax.legend()
plt.show()


# In[114]:


# Group by the Date and compute the average for all boroughs
average_prices = t_properties_filtered.groupby('Date')['Average Price'].mean()

# Now, plot the data
ax = average_prices.plot(kind='line', figsize=(15,10))
ax.set_ylabel('Average Pricen (£) Across Boroughs')
ax.set_title('Average House Prices Across All Boroughs Over Time')
plt.grid(True)
ax.legend()
plt.show()


# In[115]:


overall_avg = t_properties_filtered.groupby('Date')['Average Price'].mean()
highest_avg = t_properties_filtered.groupby('Borough Name')['Average Price'].mean().idxmax()
lowest_avg = t_properties_filtered.groupby('Borough Name')['Average Price'].mean().idxmin()

# Build the plot
ax = overall_avg.plot(figsize=(15,10), label='Overall Average')
t_properties_filtered[t_properties_filtered['Borough Name'] == highest_avg].groupby('Date')['Average Price'].mean().plot(ax=ax, label=highest_avg + ' (Highest)')
t_properties_filtered[t_properties_filtered['Borough Name'] == lowest_avg].groupby('Date')['Average Price'].mean().plot(ax=ax, label=lowest_avg + ' (Lowest)')

# Plot the Data
ax.set_ylabel('Average Price (£) in Millions')
ax.set_title('Average House Prices Across Time')
plt.grid(True)
ax.legend()
plt.show()


# To limit the number of data points you have, you might want to extract the year from every month value your *Month* column. 
# 
# To this end, you *could* apply a ***lambda function***. Your logic could work as follows:
# 1. look through the `Month` column
# 2. extract the year from each individual value in that column 
# 3. store that corresponding year as separate column. 
# 
# Whether you go ahead with this is up to you. Just so long as you answer our initial brief: which boroughs of London have seen the greatest house price increase, on average, over the past two decades? 

# In[92]:


# Convert 'Date' column to datetime type & extract the year
df_copy = t_properties_filtered.assign(Date=lambda x: pd.to_datetime(x['Date']), Year=lambda x: x['Date'].dt.year)

# Calculate average price per year 
annual_avg = df_copy.groupby(['Borough Name', 'Year'])['Average Price'].mean()

# Calculate price difference over the past two decades
two_decades_ago = df_copy['Year'].max() - 20
price_increase = (annual_avg.xs(key=df_copy['Year'].max(), level='Year') - 
                  annual_avg.xs(key=two_decades_ago, level='Year')).sort_values(ascending=False)

print(price_increase)


# In[120]:


price_increase_sorted = price_increase.sort_values(ascending=True)

# Create the plot
plt.title('Increase in Average House Prices(£) Over Two Decades')
price_increase_sorted.plot(kind='bar')
plt.show()


# In[117]:


import matplotlib.pyplot as plt

# Select top 20 price increases
top_20_increase = price_increase.sort_values(ascending=False).head(20)

# Set figure size
plt.figure(figsize=(15,10))

# Plotting the data
top_20_increase.sort_values().plot(kind='bar')
plt.title('Top 20 Increases in Average House Prices (£) From 1998 - 2018')
plt.grid(axis='x')

# Show the plot
plt.grid(True)
plt.show()



# ### 4. Conclusion
# What can you conclude? Type out your conclusion below. 
# 
# Look back at your notebook. Think about how you might summarize what you have done, and prepare a quick presentation on it to your mentor at your next meeting. 
# 
# We hope you enjoyed this practical project. It should have consolidated your data hygiene and pandas skills by looking at a real-world problem involving just the kind of dataset you might encounter as a budding data scientist. Congratulations, and looking forward to seeing you at the next step in the course! 

# In[ ]:


#In my recent project, I encountered substantial aspects of data analysis. I grappled with Pandas and faced challenges 
#in recalling the overall syntax for various libraries I had recently learned. Renaming columns and transforming data 
#using Pandas functions presented notable difficulties.


#I also encountered hurdles in data visualization using Matplotlib. Customizing plots and implementing fundamental 
#features like labeling, incremental ticks, and basic mathematical operations proved to be intricate. I also recognized
#that achieving an accurate and readable graph requires a delicate balance in coding and presentation. It necessitates 
#ensuring that information is easily digestible, clear, and precise.

#My project revolved around calculating ratios and examining house price trends over time. While I successfully
#computed price ratios, I encountered obstacles in DataFrame manipulation, particularly in indexing and grouping data.
#Once again, the challenge was primarily centered around syntax and my ability to recall appropriate methods and 
#functions. The project has expanded my knowledge of various statistics concerning London's boroughs over the past 
#three decades.

#Overall, this project provided a perspective on the extent of my ongoing learning journey and the areas I need to 
#continually develop to progress from a novice level. Initially, I aimed to complete the project within the suggested 
#3-6 hours as per the guidelines, yet it took approximately 14 hours to finish. This experience proved both humbling 
#and immensely educational. The inherent linearity of code execution, where mistakes impact subsequent code segments,
#became even more evident and led to several initial restarts. This project has only increased my curiosity, passion, 
#and hunger to learn more and do more with data manipulation. 



# In[ ]:




