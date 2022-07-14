import streamlit as st
import json
import requests
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def preprocess_dataframe_equipment(url):
    columns_to_sum = ['military auto', 'fuel tank', 'vehicles and fuel tanks']
    columns_to_drop = columns_to_sum + ['mobile SRBM system', 'greatest losses direction']

    df = pd.DataFrame(requests.get(url).json())
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['day'].astype(int)
    df['vehicle and fuel tank'] = df[columns_to_sum].sum(axis=1).astype(int)
    df = df.drop(columns_to_drop, axis=1)
    return df

def create_dataframe_heatmap(df, columns_list):
    '''
    Converts input dataset into dataset ready to use for heatmap chart
    '''
    df_week = df.copy()
    df_week['week_event'] = df_week['date'].dt.week
    df_week['week_label'] = df_week['date'].dt.to_period('W-SUN').apply(lambda r: r.end_time).dt.strftime('%b')+' '+\
                            df_week['date'].dt.to_period('W-SUN').apply(lambda r: r.end_time).dt.strftime('%d')

    list_df = []
    for column in columns_list:
        df_temp = df_week[[column, 'week_event']].copy()
        df_temp['name'] = column
        df_temp_pivot = pd.pivot_table(df_temp, values=column, index='name', columns='week_event', aggfunc=np.sum)
        list_df.append(df_temp_pivot)
    df_output = pd.concat(list_df, axis=0)

    x_labels_list = list(df_week['week_label'].unique())
    y_labels_list = list(df_output.index)

    return df_output, x_labels_list, y_labels_list

def plot_heat_map(values_input, values_heat_map, x_label, y_label):
    '''
    Plot heatmap chart
    '''
    fig = ff.create_annotated_heatmap(
        values_input[::-1],
        x=x_label,
        y=y_label[::-1], 
        annotation_text=values_heat_map[::-1], 
        text=values_heat_map[::-1],
        colorscale='Greys', 
        hoverinfo='text'
        )
    fig.update_layout(
        title_text='<b>Weekly losses</b>',
        title_x=0.5,
        )
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_line_plot(df, df_daily, columns_list, index_selected, date_last):
    '''
    Plot bar and line charts together
    '''
    fig = make_subplots(2, 1, subplot_titles=("<b>Daily Losses</b>", "<b>Total Losses</b>"))
    fig.add_trace(
        go.Bar(
            x=df_daily['date'],
            y=df_daily[columns_list[index_selected]],
            text=df_daily[columns_list[index_selected]]
            ), 
        row=1, 
        col=1)
    fig.add_trace(
        go.Scatter(x=df['date'], 
        y=df[columns_list[index_selected]], 
        mode='lines+markers'), 
        row=2, 
        col=1
        )
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


period_last = 7

url1 = 'https://raw.githubusercontent.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset/main/data/russia_losses_equipment.json'
url2 = 'https://raw.githubusercontent.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset/main/data/russia_losses_personnel.json'

df_personnel = pd.DataFrame(requests.get(url2).json())
df_personnel_daily = df_personnel[['date', 'personnel']].copy().set_index('date')
df_personnel_daily = df_personnel_daily.diff().fillna(df_personnel_daily).fillna(0).astype(int).reset_index()

total_losses_personnel = df_personnel.iloc[-1]['personnel']
total_losses_personnel_period = df_personnel_daily.tail(period_last).sum()['personnel']

df_equipment = preprocess_dataframe_equipment(url1)
df_equipment_daily = df_equipment.copy().set_index(['date', 'day'])
df_equipment_daily = df_equipment_daily.diff().fillna(df_equipment_daily).fillna(0).astype(int).reset_index()

df_sum = df_equipment_daily.loc[:, df_equipment_daily.columns!='day'].sum()
df_sum_period = df_equipment_daily.loc[:, df_equipment_daily.columns!='day'].tail(period_last).sum()
total_losses = df_sum.sum()
total_losses_peroid = df_sum_period.sum()

date_last = df_equipment_daily.iloc[-1]['date'].date()
day_last = df_equipment_daily.iloc[-1]['day']
columns_equipment_list = df_equipment_daily.loc[:, ~df_equipment_daily.columns.isin(['date', 'day'])].columns


st.set_page_config(page_title='War-Losses', page_icon="🇺🇦", layout="wide")

# metrics part
with st.container():
    _, col001, _ = st.columns((1.8, 1.9, 1.8))
    with col001:
        st.title('russian Equipment Losses')

    _, col0020, _ = st.columns((3.75, 1, 3.75))
    _, col0021, _ = st.columns((1.35, 1, 1.35))
    _, col0022, _ = st.columns((1.95, 1, 1.95))

    with col0020:
        st.markdown('#### {} Day of War'.format(day_last))
    with col0021:
        st.markdown('#### Total Equipment Losses: {} ⬆{}'.format(total_losses, total_losses_peroid))
    with col0022:
        st.markdown('#### The Death Toll: {} ⬆{}'.format(total_losses_personnel, total_losses_personnel_period))

    _, col101, col102, col103, col104, col105, col106, _ = st.columns((1.25, 1, 1, 1, 1, 1, 1, 1.25))
    _, col107, col108, col109, col110, col111, col112, _ = st.columns((1.25, 1, 1, 1, 1, 1, 1, 1.25))

    columns_metric = [col101, col102, col103, col104, col105, col106, 
                      col107, col108, col109, col110, col111, col112,]

    for i, col in enumerate(columns_metric):
        col.metric(
            columns_equipment_list[i],
            int(df_sum[columns_equipment_list[i]]),
            int(df_sum_period[columns_equipment_list[i]]
                ))
    
    _, col0022, _ = st.columns((4, 1, 4))
    with col0022:
        st.markdown('⬆ last week losses')

    _, col0023, _ = st.columns((3, 1, 3))
    with col0023:
        st.markdown(' Data updated on {}'.format(date_last))

# plots part
with st.container():
    _, col003, _ = st.columns((1.5, 1, 1.5))
    with col003:
        st.markdown('### Losses by the Equipment Type')

    _, col004, _ = st.columns((1, 2, 1))
    with col004:
        index_selected_equipment = st.selectbox(
            label="Select an Equipment:", 
            options=range(len(columns_equipment_list)), 
            format_func=lambda x: columns_equipment_list[x],
            index=6
            )

    # plot bar and line chharts
    plot_bar_line_plot(df_equipment, df_equipment_daily, columns_equipment_list, index_selected_equipment, date_last)

    # plot heatmap
    df_heat_map, x_labels_list, y_labels_list = create_dataframe_heatmap(df_equipment_daily, columns_equipment_list)
    values_heat_map = df_heat_map.values
    values_input = np.zeros(values_heat_map.shape)
    values_input[index_selected_equipment]=values_heat_map[index_selected_equipment]
    plot_heat_map(values_input, values_heat_map, x_labels_list, y_labels_list)

# sources and about part
with st.container():
    _, col006, _ = st.columns((2.25, 1, 2.25))
    with col006:
        st.markdown('### Sources & About')

    _, col007, _ = st.columns((1, 2, 1))
    with col007:
        st.markdown(
            """
            Dedicated to the Armed Forces of Ukraine!

            The application is a simple dashboard that describes russian Equipment Losses during the 2022 russian invasion of Ukraine.
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
