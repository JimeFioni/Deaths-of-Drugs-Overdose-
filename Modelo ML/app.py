import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import os

# Load the dataset
df = pd.read_csv('datasets/DRUG_OVER-2023.csv')

# Convert 'Year' and 'Month' columns to datetime format
df['Date'] = df['Year'].astype(str) + '-' + df['Month']
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%B')

# Filter data for the period from January 2019 to December 2023
df = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2023-08-12')]

# Filter the dataset for 'Number of Drug Overdose Deaths'
df_forecast_deaths = df[df['Indicator'] == 'Número de muertes por sobredosis de opioides']

# Ensure the data is in the correct order
sorted_df = df_forecast_deaths.sort_values(by='Date')

# Handle missing values by filling with the mean
sorted_df['Data Value'] = sorted_df['Data Value'].fillna(sorted_df['Data Value'].mean())

# Set the Date as the index
sorted_df.set_index('Date', inplace=True)

# Load the trained model
model_path = os.path.join('Models', 'trained_model.pkl')
model = joblib.load(model_path)

# Streamlit app layout
st.title('Predicción de Muertes por Sobredosis de Drogas para 2024~2025')

# Subheading based on functionalities
st.subheader('Esta aplicación te permite:  \n* Visualizar el histórico mensual de muertes por sobredosis de drogas en USA.  \n* Analizar el comportamiento de los datos mediante el modelo SARIMA (1, 1, 1) x (1, 1, 1, 12).  \n* Generar un pronóstico para el año 2024~2025. \n* Ten en cuenta que el pronóstico es para fines informativos y no debe utilizarse para la toma de decisiones.')


# Sidebar navigation
page = st.sidebar.selectbox('Page', ['PostCovid Forecasting', 'Reporte PostCovid'])

# Page content
import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Function to load and preprocess data
def load_data():
  # Replace with actual file path if needed
  file_path = 'datasets/DRUG_OVER-2023.csv'
  data = pd.read_csv(file_path, encoding='ascii')
  data['Date'] = pd.to_datetime(data['Date'])
  data.set_index('Date', inplace=True)
  return data.groupby(pd.Grouper(freq='M')).sum()['Data Value']

# Load data on app launch
monthly_data = load_data()


# Display historical data with chart
fig, ax = plt.subplots()
ax.plot(monthly_data.index, monthly_data)
ax.set_xlabel('Date')
ax.set_ylabel('Número de muertes por sobredosis de opioides')
st.pyplot(fig)

# Model parameters section
st.subheader('Parametros del Modelo SARIMA')
st.text(' - Order: (1, 1, 1)')
st.text(' - Seasonal Order: (1, 1, 1, 12)')

# Forecast section
st.subheader('Forecast para 2024~2025')

# Button to trigger forecast calculation
if st.button('Generate Forecast'):
  with st.spinner('Forecasting...'):  
    # Fit model (consider caching for performance)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model = SARIMAX(monthly_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Generate forecast
    forecast = model_fit.forecast(steps=12)
    
    # Plot forecast
    fig, ax = plt.subplots()
    ax.plot(monthly_data.index, monthly_data, label='Historical Monthly Data')
    ax.plot(forecast.index, forecast, label='Forecasted Monthly Data for 2024', color='black')
    ax.set_title('Forecast of Drug Overdose Deaths for 2024~2025')
    ax.set_xlabel('Date')
    ax.set_ylabel('Número de muertes por sobredosis de opioides')
    ax.legend()
    st.pyplot(fig)
    
    # Display disclaimer (consider adding confidence intervals)
    st.warning('This forecast is for informational purposes only and should not be used for decision-making.')





# POSTCOVID REPORT
elif page == 'Reporte PostCovid':
    st.header('Reporte PostCovid')

    # Total number of deaths by year
    deaths_by_year = df.groupby(df['Date'].dt.year)['Data Value'].sum()
    st.subheader('Número de muertes por año')
    st.bar_chart(deaths_by_year)

    # Total number of deaths by month
    deaths_by_month = df.groupby(df['Date'].dt.strftime('%B'))['Data Value'].sum().sort_values()
    st.subheader('Número de muertes por mes durante 2023')
    st.bar_chart(deaths_by_month)

    # Heatmap for deaths by year and month
    deaths_pivot = df.pivot_table(index=df['Date'].dt.month, columns=df['Date'].dt.year, values='Data Value', aggfunc='sum')
    st.subheader('Heatmap Número de muertes por año y por MES')
    sns.heatmap(deaths_pivot, cmap='viridis', annot=True, fmt='g')
    st.pyplot()

# Box plot for deaths by month
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df['Month'] = df['Date'].dt.strftime('%B')
    st.subheader('Box Plot of Deaths by Month')
    sns.boxplot(x='Month', y='Data Value', data=df, palette='viridis')
    plt.xlabel('Month')
    plt.ylabel('Number of Deaths')
    plt.title('Distribution of Deaths by Month in 2023')
    plt.xticks(rotation=45)
    st.pyplot()

    # Geographical perspective (if data allows)
    if 'Department' in df.columns:  # Check if department data exists
        deaths_by_department = df.groupby('Department')['Data Value'].sum().sort_values(ascending=False)
        st.subheader('Top 5 Departments with Highest Deaths')
        st.bar_chart(deaths_by_department.head(5))
        # Consider adding a map visualization if location data is available

    # Age group perspective (if data allows)
    if 'Age Group' in df.columns:  # Check if age group data exists
        deaths_by_age_group = df.groupby('Age Group')['Data Value'].sum().sort_values(ascending=False)
        st.subheader('Deaths by Age Group')
        st.pie_chart(deaths_by_age_group, labels=deaths_by_age_group.index, autopct='%1.1f%%')

    # Additional perspectives (consider adding more based on your data):
    # - Trend analysis (e.g., compare pre-Covid vs. post-Covid periods)
    # - Correlation analysis (if other relevant data exists)
    # - User-defined filtering based on date range, department, etc.

    # **Informative message instead of warning:**
    st.write('This report provides insights into drug overdose deaths. Further analysis might be needed for a comprehensive understanding.')


