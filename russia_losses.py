import streamlit as st
import json
import requests
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def preprocess_df_equipment(df):
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

def create_df_direction(df, direction_rename_dict):
    '''
    Creates dataframe direction
    '''
    df = df.copy()
    df['direction'] = df['greatest losses direction'].str.split(',|and')
    df_direction = df[['date', 'direction']].explode('direction')
    df_direction = df_direction[df_direction['direction'].notna()].copy()
    df_direction['direction'] = df_direction['direction'].str.strip().replace(direction_rename_dict)
    return df_direction

def preprocess_df_direction_amount(df, df_init):
    '''
    Prepocess dataframe direction for bar plot and map
    '''
    columns_to_rename = {
        'index':'direction',
        'direction':'direction_count'
    }
    df = df.copy()
    df_direction = df['direction'].value_counts(ascending=True).reset_index()\
                                  .rename(columns=columns_to_rename)
    df_result = df_direction.merge(df_init, how='left', left_on='direction', right_on='direction')
    df_result['direction'] = df_result['direction'] + '   '
    df_result['text_hover'] = df_result['direction_count'].astype(str) + ' days'
    df_result['radius'] = df_result['direction_count'].apply(lambda x: (x+25)**2)
    return df_result

def preprocess_df_direction_pivot(df):
    '''
    Prepocess dataframe for heatmap plot
    '''
    df = df.copy()
    df['value'] = 1
    df_result = df.pivot(index='date', columns='direction', values='value').T.copy()
    return df_result

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
    Plot bar chart of directions with greatest personnel losses
    '''
    fig = px.bar(
        df,
        y='direction',
        x='direction_count',
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
        # title_text='<b>Directions of personnel losses</b>',
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
        # title_text='<b>Timeline of greatest losses</b>',
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

def plot_map(df):
    '''
    Plot map of directions with greatest personnel losses
    '''
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=5,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position="coordinate_ua",
        get_radius="radius",
        get_fill_color=[255, 140, 0],
        get_line_color=[255, 0, 0],
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer], 
        map_style='light',
        height=600,
        initial_view_state=pdk.ViewState(
            latitude=49.4302723, 
            longitude=32.0533029, 
            zoom=5.5,
            min_zoom=4,
            max_zoom=9,
            bearing=0, 
            pitch=0), 
        tooltip={"text": "{direction} direction: {text_hover}"}
        )
    )

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
        col=1
        )
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
path_data_geo = 'data/geo_init.json'
path_page_2 = 'data/page2.txt'
path_page_3 = 'data/page3.txt'

direction_rename_dict = {
    'Slobozhanskyi':'Kharkiv',
    'Novopavlivsk':'Novopavlivske'
    }

with open(path_data_geo) as f:
    data_geo = json.load(f)

with open(path_page_2,'r') as f:
    page_2_contents = f.read()

with open(path_page_3,'r') as f:
    page_3_contents = f.read()

df_geo = pd.DataFrame.from_dict(data_geo, orient='index')

df_personnel = pd.DataFrame(requests.get(url2).json())
df_personnel_daily = df_personnel[['date', 'personnel']].copy().set_index('date')
df_personnel_daily = df_personnel_daily.diff().fillna(df_personnel_daily).fillna(0).astype(int).reset_index()

total_losses_personnel = df_personnel.iloc[-1]['personnel']
total_losses_personnel_period = df_personnel_daily.tail(period_last).sum()['personnel']

df = pd.DataFrame(requests.get(url1).json())
df['day'] = df['day'].astype(int)

df_equipment = preprocess_df_equipment(df)
df_equipment_daily = df_equipment.copy().set_index(['date', 'day'])
df_equipment_daily = df_equipment_daily.diff().fillna(df_equipment_daily).fillna(0).astype(int).reset_index()

df_direction = create_df_direction(df, direction_rename_dict)
df_direction_amount = preprocess_df_direction_amount(df_direction, df_geo)
df_direction_pivot = preprocess_df_direction_pivot(df_direction)

df_sum = df_equipment_daily.loc[:, df_equipment_daily.columns!='day'].sum(numeric_only=True)
df_sum_period = df_equipment_daily.loc[:, df_equipment_daily.columns!='day'].tail(period_last).sum(numeric_only=True)
total_losses = df_sum.sum()
total_losses_peroid = df_sum_period.sum()

date_last = df_equipment_daily.iloc[-1]['date'].date()
date_last_str = '**Last Data Update:** {}.'.format(date_last)

day_last = df_equipment_daily.iloc[-1]['day']
day_last_str = '#### {} Day of War'.format(day_last)

columns_equipment_list = df_equipment_daily.loc[:, ~df_equipment_daily.columns.isin(['date', 'day'])].columns


st.set_page_config(page_title='War-Losses', page_icon="🇺🇦", layout="wide")

_, col01, _ = st.columns((3.75, 1, 3.75))
with col01:
    st.markdown(day_last_str)

_, col02, _ = st.columns((1.8, 1.9, 1.8))
with col02:
    st.title('russian Equipment Losses')

tab1, tab2, tab3 = st.tabs(["Equipment Losses", "Directions with Greatest Losses", "Sources & About"])

with tab1:
    with st.container():
        _, col1001, _ = st.columns((1.35, 1, 1.35))
        _, col1002, _ = st.columns((1.95, 1, 1.95))

        with col1001:
            st.markdown('#### Total Equipment Losses: {} ⬆{}'.format(total_losses, total_losses_peroid))
        with col1002:
            st.markdown('#### The Death Toll: {} ⬆{}'.format(total_losses_personnel, total_losses_personnel_period))

        _, col101, col102, col103, col104, col105, col106, _ = st.columns((1, 1, 1, 1, 1, 1, 1, 1))
        _, col107, col108, col109, col110, col111, col112, _ = st.columns((1, 1, 1, 1, 1, 1, 1, 1))

        columns_metric = [col101, col102, col103, col104, col105, col106, 
                          col107, col108, col109, col110, col111, col112,]

        for i, col in enumerate(columns_metric):
            col.metric(
                columns_equipment_list[i],
                int(df_sum[columns_equipment_list[i]]),
                int(df_sum_period[columns_equipment_list[i]])
                )
        
        _, col1003, _ = st.columns((4.25, 1, 4.25))
        with col1003:
            st.markdown('**Last Week Losses:** ⬆.')

        _, col1004, _ = st.columns((3, 1, 3))
        with col1004:
            st.markdown(date_last_str)

        _, col1005, _ = st.columns((1.8, 1, 1.8))
        with col1005:
            st.markdown('#### Losses by the Equipment Type')

        _, col006, _ = st.columns((1, 2, 1))
        with col006:
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

with tab2:
    with st.container():
        _, col2001, _ = st.columns((1, 2, 1))
        with col2001:
            st.markdown('### Directions with Greatest Losses')
            st.markdown(page_2_contents)
            st.markdown(date_last_str)
            st.markdown('### Timeline')

        # plot heatmap of directions with greatest losses
        plot_heatmap_direction(df_direction_pivot, date_last)

        # plot map of directions with greatest losses
        _, col2002, _ = st.columns((1, 2, 1))
        with col2002:
            col2002.markdown('### Map')
        plot_map(df_direction_amount)

        # plot bar chart of directions with greatest losses
        _, col2003, _ = st.columns((1, 2, 1))
        with col2003:
            col2003.markdown('### Days with Greatest Losses Distributions')
            plot_bar(df_direction_amount)

with tab3:
    with st.container():
        _, col3001, _ = st.columns((1, 2, 1))
        with col3001:
            st.markdown('### Sources & About')
            st.markdown(page_3_contents)
