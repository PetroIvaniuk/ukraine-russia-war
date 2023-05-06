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


def convert_data2daily(df):
    df = df.copy().set_index(['date', 'day'])
    df = df.diff().fillna(df).fillna(0).astype(int).reset_index()
    return df

def create_equipment_model_dict(df, column_equipment):
    '''
    Creates equipment-equipment model dictionary
    '''
    df = df[(df['equipment_ua']==column_equipment)&
            (~df['model'].str.lower().str.contains('unknown'))].copy()
    equipment_model_dict = df.groupby(['equipment_oryx'])['model'].apply(lambda x: sorted(list(set(x)))).to_dict()
    return equipment_model_dict

def initial_preprocessing(df, columns_rename_dict):
    '''
    Preprocess inital dataset
    '''
    columns_to_sum = ['military auto', 'fuel tank', 'vehicles and fuel tanks']
    columns_to_drop = ['mobile SRBM system', 'greatest losses direction']

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['vehicle and fuel tank'] = df[columns_to_sum].sum(axis=1).astype(int)
    df = df.drop(columns_to_sum+columns_to_drop, axis=1)
    df = df.rename(columns=columns_rename_dict)
    return df

def create_direction_dataframe(df, direction_rename_dict):
    '''
    Creates dataframe direction
    '''
    df = df.copy()
    df['direction'] = df['greatest losses direction'].str.split(',|and')
    df_direction = df[['date', 'direction']].explode('direction')
    df_direction = df_direction[df_direction['date']>='2022-04-25'].copy()
    df_direction['direction'] = df_direction['direction'].str.strip().replace(direction_rename_dict)

    last_date_valid = df_direction[df_direction['direction'].apply(type)==str]['date'].tail(1).iloc[0]
    df_direction = df_direction[df_direction['date']<=last_date_valid].copy()
    df_direction['date'] = pd.to_datetime(df_direction['date'])
    return df_direction

def preprocess_direction_amount(df, df_geo):
    '''
    Prepocess dataframe direction for bar plot and map
    '''
    df = df.copy()
    df_direction = df['direction'].value_counts(ascending=True).reset_index()
    df_result = df_direction.merge(df_geo, how='left', on='direction')
    df_result['direction'] = df_result['direction'] + '   '
    df_result['text_hover'] = df_result['count'].astype(str) + ' days'
    df_result['radius'] = df_result['count'].apply(lambda x: (x/np.pi)**0.5*1000)
    return df_result

def preprocess_direction_pivot(df):
    '''
    Prepocess dataframe for heatmap plot
    '''
    df = df.copy()
    df['value'] = 1
    df_result = df.pivot(index='date', columns='direction', values='value').T.copy()
    return df_result

def create_weekly_dataframe(df, columns_list):
    '''
    Converts input dataset into dataset ready to use for heatmap chart
    '''
    df_week = df.copy()
    df_week['week_event'] = df_week['date'].dt.isocalendar().week
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

def create_treemap_dataframe(df):
    '''
    Converts input dataset ready to use fro treemap chart
    '''
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['week'] = 'week ' + df['date'].dt.isocalendar().week.astype(str)
    df_group = df.groupby(['year', 'month', 'week'])['personnel'].sum().reset_index()
    return df_group

def plot_bar(df):
    '''
    Plot bar chart of directions with greatest personnel losses
    '''
    fig = px.bar(
        df,
        y='direction',
        x='count',
        color='oblast',
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_name='text_hover',
        text_auto=True,
    )
    annotation_list = [
        dict(font=dict(size=18),
             x=0.01,
             y=1.04,
             showarrow=False,
             text="NUMBER OF DAYS OF GREATEST LOSSES",
             textangle=0,
             xanchor='left',
             yref="paper",
             xref="paper"),
         dict(font=dict(size=18),
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
        title_text='<b>Places of Greatest Losses</b>',
        title_x=0.25,
        title=dict(font=dict(size=24)),
        font_size=18,
        yaxis=dict(tickfont=dict(size=16)),
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
        title_text='<b>Timeline of Places of Greatest Losses</b>',
        title_x=0.4,
        title=dict(font=dict(size=24)),
        font_size=16,
        yaxis=dict(tickfont=dict(size=16)),
        xaxis=dict(tickfont=dict(size=14)),
    )
    fig.update_layout(
        xaxis=dict(
            range=[date_last - relativedelta(months=+3), date_last],
            rangeselector=dict(
                buttons=list([
                    dict(count=3, label="last 3 months", step="month", stepmode="backward"),
                    dict(count=6, label="last 6 months", step="month", stepmode="backward"),
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
            latitude=48.9302723,
            longitude=32.0533029, 
            zoom=6,
            min_zoom=4,
            max_zoom=9,
            bearing=0, 
            pitch=0
            ),
        tooltip={"text": "{direction} direction: {text_hover}"}
        )
    )

def plot_heatmap_weekly_losses(values_input, values_heat_map, x_label, y_label, title_label):
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
        title_text='<b>{} Year - Weekly Losses</b>'.format(title_label),
        title_x=0.4,
        title=dict(font=dict(size=24)),
        height=800,
        font=dict(size=16),
        yaxis=dict(tickfont=dict(size=18)),
        xaxis=dict(tickfont=dict(size=18), ticklabelposition='inside'),
        )
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_line_plot(df, df_daily, columns_selected, date_last):
    '''
    Plot bar and line charts together
    '''
    fig = make_subplots(2, 1, subplot_titles=("<b>Daily Losses</b>", "<b>Total Losses</b>"))
    fig.add_trace(
        go.Bar(
            x=df_daily['date'],
            y=df_daily[columns_selected],
            text=df_daily[columns_selected]
            ), 
        row=1, 
        col=1
        )
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df[columns_selected],
            mode='lines+markers'),
        row=2, 
        col=1
        )
    fig.update_layout(
        xaxis=dict(
            range=[date_last - relativedelta(months=+2), date_last],
            rangeselector=dict(
                borderwidth=1,
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

def plot_treemap(df):
    fig = px.treemap(
        df,
        path=[px.Constant("Personnel Losses"), 'year', 'month', 'week'],
        values='personnel',
        color='personnel',
        color_continuous_scale='blues',
        maxdepth=3
    )
    fig.update_layout(
        height=800,
        margin = dict(t=50, l=25, r=25, b=25)
    )
    st.plotly_chart(fig, use_container_width=True)


path_config = 'config.json'
with open(path_config) as f:
    config = json.load(f)

with open(config['data_geo']) as f:
    data_geo = json.load(f)

content_dict = {}
for page in config['data_content'].keys():
    with open(config['data_content'][page], 'r') as f:
        content_dict[page] = f.read()

period_last = config['period_last']
df_geo = pd.DataFrame.from_dict(data_geo, orient='index')
df_equipment_oryx = pd.DataFrame(requests.get(config['url']['oryx']).json())
df_personnel = pd.DataFrame(requests.get(config['url']['personnel']).json())
df_equipment = pd.DataFrame(requests.get(config['url']['equipment']).json())

df_equipment_proc = initial_preprocessing(df_equipment, config['rename_columns'])
df_daily_equipment = convert_data2daily(df_equipment_proc)
df_daily_personnel = convert_data2daily(df_personnel[['date', 'day', 'personnel']])

df_sum = df_daily_equipment.loc[:, df_daily_equipment.columns!='day'].sum(numeric_only=True)
df_sum_period = df_daily_equipment.loc[:, df_daily_equipment.columns!='day'].tail(period_last).sum(numeric_only=True)

total_losses_personnel = df_personnel['personnel'].tail(1).iloc[0]
total_losses_personnel_period = total_losses_personnel - df_personnel['personnel'].tail(period_last).iloc[0]
total_losses = df_sum.sum()
total_losses_peroid = df_sum_period.sum()

df_direction = create_direction_dataframe(df_equipment, config['rename_direction'])
df_direction_amount = preprocess_direction_amount(df_direction, df_geo)
df_direction_pivot = preprocess_direction_pivot(df_direction)

date_last = df_daily_equipment.iloc[-1]['date'].date()
day_last = df_daily_equipment.iloc[-1]['day']
date_last_direction = df_direction['date'].tail(1).iloc[0]

date_last_str = '**Last Data Update:** {}.'.format(date_last)
day_last_str = '#### {} Day of War'.format(day_last)
date_last_direction_str = '**Last Data Update:** {}.'.format(date_last_direction)
metric_help_str = '⬆ - Last {} Day Losses'.format(period_last)

columns_losses_list = sorted(df_daily_equipment.loc[:, ~df_daily_equipment.columns.isin(['date', 'day'])].columns)
columns_equipment_list = sorted(df_daily_equipment.loc[:, ~df_daily_equipment.columns.isin(['date', 'day', 'Personnel'])].columns)

st.set_page_config(page_title='War-Losses', page_icon="🇺🇦", layout="wide")

_, col01, _ = st.columns((3.85, 1, 3.85))
with col01:
    st.markdown(day_last_str)

_, col02, _ = st.columns((2.1, 1, 2.1))
with col02:
    st.title('russian losses')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Equipment", "Personnel", "Directions with Greatest Losses", "App Changelog", "About"])

with tab1:
    with st.container():
        _, col1000, _ = st.columns((1, 2, 1))
        with col1000:
            st.markdown('### Equipment Losses during the 2022 russian invasion of Ukraine')

        _, col113, col114, _ = st.columns((3, 1, 1, 3))
        with col113:
            col113.metric(
                'Total Equipment Losses',
                int(total_losses),
                int(total_losses_peroid),
                help=metric_help_str,
                )

        with col114:
            col114.metric(
                'The Death Toll',
                int(total_losses_personnel),
                int(total_losses_personnel_period),
                help=metric_help_str,
                )

        _, col101, col102, col103, col104, _ = st.columns((2, 1, 1, 1, 1, 2))
        _, col105, col106, col107, col108, _ = st.columns((2, 1, 1, 1, 1, 2))
        _, col109, col110, col111, col112, _ = st.columns((2, 1, 1, 1, 1, 2))

        columns_metric = [
            col101, col102, col103, col104, col105, col106,
            col107, col108, col109, col110, col111, col112,
        ]

        for i, col in enumerate(columns_metric):
            col.metric(
                columns_equipment_list[i],
                int(df_sum[columns_equipment_list[i]]),
                int(df_sum_period[columns_equipment_list[i]]),
                help=metric_help_str,
                )

        _, col1004, _ = st.columns((3, 1, 3))
        with col1004:
            st.markdown(date_last_str)

        _, col1005, _ = st.columns((1, 2, 1))
        with col1005:
            st.markdown('#### Losses by Type')

        _, col1006, _ = st.columns((1, 2, 1))
        with col1006:
            index_selected_equipment = st.selectbox(
                label="Select Losses Type:", 
                options=range(len(columns_losses_list)),
                format_func=lambda x: columns_losses_list[x],
                index=9,
                )

        column_selected_equipment = columns_losses_list[index_selected_equipment]
        equipment_model_dict = create_equipment_model_dict(df_equipment_oryx, column_selected_equipment)

        _, col1007, _ = st.columns((1, 2, 1))
        with col1007:
            if len(equipment_model_dict)!=0:
                st.markdown('#### List of {} of which photo evidence is available'.format(column_selected_equipment))
                for equipment, equipment_list in equipment_model_dict.items():
                    with st.expander(equipment):
                        for model in equipment_list:
                            st.markdown(model)
            else:
                st.info('There is no additional information about {}.'.format(column_selected_equipment))

        # plot bar and line charts
        plot_bar_line_plot(df_equipment_proc, df_daily_equipment, column_selected_equipment, date_last)

        # plot heatmap of equipment losses
        years_list = sorted(df_daily_equipment['date'].dt.year.unique())
        for year in years_list:
            df_daily_temp = df_daily_equipment[df_daily_equipment['date'].dt.year==year].copy()
            df_weekly, x_labels_list, y_labels_list = create_weekly_dataframe(df_daily_temp, columns_losses_list)

            values_weekly = df_weekly.values
            values_input = np.zeros(df_weekly.shape)
            values_input[index_selected_equipment]=values_weekly[index_selected_equipment]
            plot_heatmap_weekly_losses(values_input, values_weekly, x_labels_list, y_labels_list, year)

with tab2:
    with st.container():
        _, col201, _ = st.columns((1, 2, 1))
        with col201:
            st.markdown('### Personnel Losses during the 2022 russian invasion of Ukraine')

        df_daily_personnel_chart = df_daily_personnel[df_daily_personnel['day']>8].copy()
        plot_bar_line_plot(df_personnel, df_daily_personnel_chart, 'personnel', date_last)
        df_daily_personnel_group = create_treemap_dataframe(df_daily_personnel)
        plot_treemap(df_daily_personnel_group)

with tab3:
    with st.container():
        st.markdown('### Directions with Greatest Losses')

        col301, col302= st.columns((1, 1), gap='large')
        with col301:
            st.markdown(content_dict['page_2'])
            st.markdown(date_last_direction_str)
        with col302:
            plot_bar(df_direction_amount)

        # plot heatmap of directions with greatest losses
        plot_heatmap_direction(df_direction_pivot, date_last_direction)
        # plot map of directions with greatest losses
        plot_map(df_direction_amount)

with tab4:
    with st.container():
        _, col401, _ = st.columns((1, 2, 1))
        with col401:
            st.markdown(content_dict['page_3'])

with tab5:
    with st.container():
        _, col501, _ = st.columns((1, 2, 1))
        with col501:
            st.markdown(content_dict['page_4'])