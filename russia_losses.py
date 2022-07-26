import streamlit as st
import json
import requests
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def preprocess_dataframe_equipment(df):
    '''
    Preprocess equipment losses dataset
    '''
    columns_to_sum = ['military auto', 'fuel tank', 'vehicles and fuel tanks']
    columns_to_drop = columns_to_sum + ['mobile SRBM system', 'greatest losses direction']
    columns_to_rename = {
        'aircraft': 'Aircrafts',
        'helicopter': 'Helicopters',
        'tank': 'Tanks',
        'APC': 'Armoured Personnel Carriers',
        'field artillery': 'Artillery Systems',
        'MRL': 'Multiple Rocket Launchers',
        'drone': 'Unmanned Aerial Vehicles',
        'naval ship': 'Warships, Boats',
        'anti-aircraft warfare': 'Anti-aircraft Warfare Systems',
        'special equipment': 'Special Equipment',
        'cruise missiles': 'Cruise Missiles',
        'vehicle and fuel tank': 'Vehicle and Fuel Tank',
    }
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['vehicle and fuel tank'] = df[columns_to_sum].sum(axis=1).astype(int)
    df = df.drop(columns_to_drop, axis=1)
    df = df.rename(columns=columns_to_rename)
    return df

def create_dataframe_direction_bar(df, direction_rename_dict):
    '''
    Creats direction dataset for bar chart
    '''
    map_direction_oblast = {
        'Popasna':'Luhansk',
        'Slobozhanskyi':'Kharkiv',
        'Kharkiv':'Kharkiv',
        'Mykolaiv':'Mykolaiv',
        'Lyman':'Donetsk',
        'Novopavlivsk':'Kharkiv',
        'Donetsk':'Donetsk',
        'Sievierodonetsk':'Luhansk',
        'Kryvyi Rih':'Dnipropetrovsk',
        'Izyum':'Kharkiv',
        'Kurakhove':'Donetsk',
        'Zaporizhzhia':'Zaporizhzhia',
        'Kramatorsk':'Donetsk',
        'Avdiivka':'Donetsk',
        'Sloviansk':'Donetsk',
        'Bakhmut':'Donetsk',
    }
    df = df.copy()
    df['direction'] = df['greatest losses direction'].str.split(',|and')
    df_direction = df['direction'].explode().str.strip().replace(direction_rename_dict)\
                                  .value_counts(ascending=True).reset_index()\
                                  .rename(columns={
                                    'index':'direction',
                                    'direction':'occurrence'
                                    })
    df_direction['oblast'] = df_direction['direction'].replace(map_direction_oblast)
    df_direction['direction'] = df_direction['direction'] + '   '
    df_direction['text_hover'] = df_direction['occurrence'].astype(str) + ' days'
    return df_direction

def create_dataframe_direction_heatmap(df, direction_rename_dict):
    '''
    Creats direction dataset for heatmap chart
    '''
    df = df.copy()
    df['direction'] = df['greatest losses direction'].str.split(',|and')
    df_direction = df[df['direction'].notna()][['date', 'direction']].set_index('date')['direction'].explode()\
                                                                     .reset_index()
    df_direction['direction'] = df_direction['direction'].str.strip().replace(direction_rename_dict)
    df_direction['value'] = 1
    df_pivot = df_direction.pivot(index='date', columns='direction', values='value').T.copy()
    return df_pivot

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

def plot_bar(df):
    '''
    Plot bar chart
    '''
    fig = px.bar(
        df,
        y='direction',
        x='occurrence',
        color='oblast',
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_name='text_hover',
        text_auto=True,
    )
    annotation_list = [
        dict(font=dict(size=14),
             x=0.01,
             y=1.04,
             showarrow=False,
             text="NUMBER OF DAYS OF GREATEST LOSSES",
             textangle=0,
             xanchor='left',
             yref="paper",
             xref="paper"),
         dict(font=dict(size=14),
             x=0,
             y=1.04,
             showarrow=False,
             text="DIRECTION",
             textangle=0,
             xanchor='right',
             yref="paper",
             xref="paper")
    ]
    fig.update_layout(
        height=800,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis_showticklabels=False,
        xaxis_visible=False,
        yaxis_title=None,
        title_text='<b>Directions of personnel losses</b>',
        title_x=0.5,
        font_size=16,
        annotations=annotation_list,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_direction(df, date_last):
    '''
    Plot heatmap chart of directions with greatest personnel losses
    '''
    ticks_number = df.columns.shape[0]//2
    fig = px.imshow(df, color_continuous_scale=["blue", "green"])
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        yaxis_title=None,
        xaxis_title=None,
        coloraxis_showscale=False,
        xaxis_nticks=ticks_number,
        title_text='<b>Timeline of personnel losses</b>',
        title_x=0.5,
        font_size=16,
    )
    fig.update_layout(
        xaxis=dict(
            range=[date_last - relativedelta(months=+3), date_last],
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="last 3 months", step="month", stepmode="backward"),
                    dict(label="all time", step="all")
                    ])
            ),
            type="date",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_losses(values_input, values_heat_map, x_label, y_label):
    '''
    Plot heatmap chart of equipment losses
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
        title_text='<b>Weekly Losses</b>',
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

direction_rename_dict = {
    'Slobozhanskyi':'Kharkiv'
    }

df_personnel = pd.DataFrame(requests.get(url2).json())
df_personnel_daily = df_personnel[['date', 'personnel']].copy().set_index('date')
df_personnel_daily = df_personnel_daily.diff().fillna(df_personnel_daily).fillna(0).astype(int).reset_index()

total_losses_personnel = df_personnel.iloc[-1]['personnel']
total_losses_personnel_period = df_personnel_daily.tail(period_last).sum()['personnel']

df = pd.DataFrame(requests.get(url1).json())
df['day'] = df['day'].astype(int)

df_equipment = preprocess_dataframe_equipment(df)
df_equipment_daily = df_equipment.copy().set_index(['date', 'day'])
df_equipment_daily = df_equipment_daily.diff().fillna(df_equipment_daily).fillna(0).astype(int).reset_index()

df_direction_bar = create_dataframe_direction_bar(df, direction_rename_dict)
df_direction_heatmap = create_dataframe_direction_heatmap(df, direction_rename_dict)

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

    _, col101, col102, col103, col104, col105, col106, _ = st.columns((1, 1, 1, 1, 1, 1, 1, 1))
    _, col107, col108, col109, col110, col111, col112, _ = st.columns((1, 1, 1, 1, 1, 1, 1, 1))

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

    # plot bar and line charts
    plot_bar_line_plot(df_equipment, df_equipment_daily, columns_equipment_list, index_selected_equipment, date_last)

    # plot heatmap of equipment losses
    df_heat_map, x_labels_list, y_labels_list = create_dataframe_heatmap(df_equipment_daily, columns_equipment_list)
    values_heat_map = df_heat_map.values
    values_input = np.zeros(values_heat_map.shape)
    values_input[index_selected_equipment]=values_heat_map[index_selected_equipment]
    plot_heatmap_losses(values_input, values_heat_map, x_labels_list, y_labels_list)

    # plot heatmap of greatest direction
    _, col005, _ = st.columns((1.15, 3, 1.15))
    with col005:
        st.markdown('### Directions with greatest losses of russian personnel, since 2022-04-25')
    plot_heatmap_direction(df_direction_heatmap, date_last)

    # plot bar chart
    _, col006, _ = st.columns((1, 2, 1))
    with col006:
        plot_bar(df_direction_bar)


# sources and about part
with st.container():
    _, col007, _ = st.columns((2.5, 1, 2.5))
    with col007:
        st.markdown('### Sources & About')

    _, col008, _ = st.columns((1, 2, 1))
    with col008:
        st.markdown(
            """
            Dedicated to the Armed Forces of Ukraine!

            The application is a simple dashboard that describes russian Equipment Losses during the 2022 russian invasion of Ukraine.
            The data includes official information from [Armed Forces of Ukraine](https://www.zsu.gov.ua/en) and
            [Ministry of Defence of Ukraine](https://www.mil.gov.ua/en/). The data will be updated daily till Ukraine win.

            **Tracking**
            - Aircrafts
            - Helicopters
            - Tanks
            - Armoured Personnel Carriers
            - Artillery Systems
            - Multiple Rocket Launchers
            - Unmanned Aerial Vehicles
            - Warships, Boats
            - Anti-aircraft Warfare Systems
            - Special Equipment
            - Cruise Missiles
            - Vehicle and Fuel Tank

            **Possible Acronyms**
            - MRL - Multiple Rocket Launcher
            - APC - Armored Personnel Carrier
            - SRBM - Short-range Ballistic Missile
            - POW - Prisoner of War
            - UAV - Unmanned Aerial Vehicle
            - RPA - Remotely Piloted Vehicle

            **Data Sources**
            - [2022 Ukraine Russia War Dataset](https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset) -
                russia Losses, JSON format on GitHub.
                [![GitHub Repo stars](https://img.shields.io/github/stars/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset?style=social)](https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset)
            - [2022 Ukraine Russia War Dataset](https://doi.org/10.34740/KAGGLE/DS/1967621) - russia Losses, CSV format on Kaggle.

            **Data Sources (Additional)**
            - [2022 Ukraine Russia War, Losses, Oryx + Images](https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx) -
                Ukraine and russia Equipment Losses based on Oryx data. The dataset includes images of all losses.
            - [rassian navi dataset](https://www.kaggle.com/datasets/piterfm/russian-navy) -
                All Surface Combatants, Submarines, Littoral Warfare Ships, Rescue, and Auxiliary Ships. The dataset includes all warship losses.
            
            **Contacts**
            
            [![GitHub followers](https://img.shields.io/github/followers/PetroIvaniuk?style=social)](https://github.com/PetroIvaniuk)
            [![](https://img.shields.io/badge/Linkedin-Connect-informational)](https://www.linkedin.com/in/petro-ivaniuk-68a89432/)
            [![Twitter Follow](https://img.shields.io/twitter/follow/PetroIvanyuk?style=social)](https://twitter.com/PetroIvanyuk)


            **© Petro Ivaniuk, 2022**
            """)
