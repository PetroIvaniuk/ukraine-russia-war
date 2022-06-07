import streamlit as st
import json
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
from dateutil.relativedelta import relativedelta


url = 'https://raw.githubusercontent.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset/main/data/russia_losses_equipment.json'
response = requests.get(url)
df = pd.DataFrame(response.json())

columns_sum = ['military auto', 'fuel tank', 'vehicles and fuel tanks']
columns_drop = columns_sum + ['mobile SRBM system', 'greatest losses direction']

df['date'] = pd.to_datetime(df['date'])
df['vehicle and fuel tank'] = df[columns_sum].sum(axis=1).astype(int)
df = df.drop(columns_drop, axis=1)

df_daily = df.copy().set_index(['date', 'day'])
df_daily = df_daily.diff().fillna(df_daily).fillna(0).astype(int).reset_index()

df_sum = df_daily.sum()
df_sum_last_7_days = df_daily.tail(7).sum()
losses_total = df_sum.sum()
date_last = df_daily.iloc[-1]['date'].date()
day_last = df_daily.iloc[-1]['day']
columns = df_daily.columns[2:]

st.set_page_config(page_title='War-Losses', page_icon="🇺🇦", layout="wide")

with st.container():
    _, col001, _ = st.columns((1.75, 2, 1.75))
    with col001:
        st.title('russian Equipment Losses')

    _, col0020, _ = st.columns((5, 1, 5))
    _, col0021, _ = st.columns((1.5, 1, 1.5))

    with col0020:
        st.markdown('### Day {}'.format(day_last))
    with col0021:
        st.markdown('### Total Equipment Losses: {}'.format(losses_total))

    _, col101, col102, col103, col104, col105, col106, _ = st.columns((1.25, 1, 1, 1, 1, 1, 1, 1.25))
    _, col107, col108, col109, col110, col111, col112, _ = st.columns((1.25, 1, 1, 1, 1, 1, 1, 1.25))

    columns_lsit = [col101, col102, col103, col104, col105, col106, 
                    col107, col108, col109, col110, col111, col112,]

    for i, col in enumerate(columns_lsit):
        col.metric(columns[i], int(df_sum[columns[i]]), int(df_sum_last_7_days[columns[i]]))
    
    _, col0022, _ = st.columns((5, 1, 5))
    with col0022:
        st.markdown(' ⬆ last week losses ')

    _, col003, _ = st.columns((1.5, 1, 1.5))
    with col003:
        st.markdown('### Losses by the Equipment Type')

    _, col004, _ = st.columns((0.5, 2, 0.5))
    with col004:
        selected_equipment = st.selectbox(
            label='Select an Equipment:', 
            options=columns, 
            index=0
        )

    _, col005, _ = st.columns((0.3, 2, 0.3))
    with col005:
        fig = make_subplots(2, 1, subplot_titles=("Daily Losses", "Total Losses"))
        fig.add_trace(go.Bar(x=df_daily['date'], y=df_daily[selected_equipment]), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df[selected_equipment], mode='lines+markers'), row=2, col=1)

        fig.update_layout(
            xaxis=dict(
                range=[date_last - relativedelta(months=+2), date_last],
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="last month", step="month", stepmode="backward"),
                        dict(count=2, label="last 2 months", step="month", stepmode="backward"),
                        dict(count=3, label="last 3 months", step="month", stepmode="backward"),
                        dict(label="all time", step="all")
                        ])
                ),
                type="date",
            ),
            xaxis2=dict(
                rangeslider=dict(visible=True, thickness=0.05),
                type="date",
            ),
        )

        fig.update_layout(
            height=800,
            showlegend=False,
        )

        fig.update_xaxes(matches='x')

        st.plotly_chart(fig, use_container_width=True)


    _, col006, _ = st.columns((2.25, 1, 2.25))
    with col006:
        st.markdown('### Sources & About')

    _, col007, _ = st.columns((1, 2, 1))
    with col007:
        st.markdown(
            """
            Dedicated to the Armed Forces of Ukraine!

            The application is simple dashbord that describes russian Equipment Losses during the 2022 russian invasion of Ukraine.
            The data includes official information from [Armed Forces of Ukraine](https://www.zsu.gov.ua/en) 
            and [Ministry of Defence of Ukraine](https://www.mil.gov.ua/en/). The data will be updated daily till Ukraine win.

            **Acronyms**
            - MRL - Multiple Rocket Launcher,
            - APC - Armored Personnel Carrier,
            - SRBM - Short-range ballistic missile,
            - POW - Prisoner of War,
            - drones: 
              - UAV - Unmanned Aerial Vehicle, 
              - RPA - Remotely Piloted Vehicle.

            **Data Sources**
            - [2022 Ukraine Russia War Dataset](https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset) - JSON format on GitHub [![GitHub Repo stars](https://img.shields.io/github/stars/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset?style=social)](https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset)
            - [2022 Ukraine Russia War Dataset](https://doi.org/10.34740/KAGGLE/DS/1967621) - CSV format on Kaggle.
            - [2022 Ukraine Russia War Equipment Losses Oryx Dataset](https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx) - Ukraine and russia Equipment Losses based on Oryx data.
            
            **Contacts**

            [![GitHub followers](https://img.shields.io/github/followers/PetroIvaniuk?style=social)](https://github.com/PetroIvaniuk)
            [![](https://img.shields.io/badge/Linkedin-Connect-informational)](https://www.linkedin.com/in/petro-ivaniuk-68a89432/)
            [![Twitter Follow](https://img.shields.io/twitter/follow/PetroIvanyuk?style=social)](https://twitter.com/PetroIvanyuk)


            **© Petro Ivaniuk, 2022**
            """)
