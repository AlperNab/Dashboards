# Food-Analysis-Dashboard

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
from dash import dcc,html,Dash
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go   
import pandas as pd
import numpy as np
import dash_pivottable
import dash_bootstrap_components as dbc


app = dash.Dash(__name__)
server = app.server

app.title= 'Food Delievery Dashboard'

df=pd.read_csv('onlinedeliverydata.csv')
user=['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications']
occ=['Occupation','Meal(P1)']

y=html.Div([
    html.Div([html.H3("The total who are willing to buy again vs those who are not")],style={'text-align':'center','display':'inline-block','width':'100%','font-weight':'900','font-size':'150%'}),
    html.Div([
    html.Div([html.H3("Yes",style={'width':'100%','font-weight':'900','font-size':'150%','font-family': '"Fantasy", Times, serif'},id="tooltip-target1"),dbc.Tooltip(df[df['Output']=='Yes']['Output'].count(),style={'display':'inline-block','width':'200%','font-weight':'900','font-size':'150%'},target="tooltip-target1")]),
    html.Div([html.H3("No",style={'width':'100%','font-weight':'900','font-size':'150%','font-family': '"Fantasy", Times, serif'},id="tooltip-target2"),dbc.Tooltip(df[df['Output']=='No']['Output'].count(),style={'display':'inline-block','width':'200%','font-weight':'900','font-size':'150%'},target="tooltip-target2")]),
],style={'margin-left':'35%','padding': '20px','border': '2px solid black','border-radius': '25px','width':'25%','height':'25%','background-color':'#28282B', 'text-color':'#FFFFFF','text-align':'center', 'float':'center'}),
    html.Div([
    html.H3('Data heatmap related to outcome',style={'text-align':'center','display':'inline-block','width':'100%','font-weight':'900','font-size':'150%'}),
    dcc.RadioItems(id='heat1',
    options=[
        {'label': 'Output', 'value': 'Output'}]),

    dcc.Graph(id="graph12"),
    
]),    
html.Div([  
dcc.Dropdown(id='drop',
        options=[
        {'label': i, 'value': i} for i in user],
    value=['Gender', 'Marital Status', 'Occupation',  'Educational Qualifications'],
    multi=True,
    clearable=False,
    searchable=True,),
html.Div(dcc.Graph(id="figure_data",style={'height':'25'})),
],style={'float':'left','display':'inline-block','width':'50%'}),
html.Div([ 
    dcc.RadioItems(id='radio',
    options=[
        {'label': 'Age', 'value': 'Age'},
        {'label': 'Family size ', 'value': 'Family size'}],value='Age',style={'font-size':25}),
html.Div(dcc.Graph(id="figure_data1")),
],style={'float':'right','display':'inline-block','width':'50%','height':'35%'}),
    
html.Div([html.H3(" What do you want to check the output of buying again with ",style={'text-align':'center','display':'inline-block','width':'100%','font-weight':'900','font-size':'150%'}), dcc.Dropdown(id='outputoutput',
            options=[
            {'label': 'Meal(P1)', 'value': 'Meal(P1)'},
            {'label': 'Meal(P2)', 'value': 'Meal(P2)'},
            {'label': 'Perference(P1)', 'value': 'Perference(P1)'},
            {'label': 'Perference(P2)', 'value': 'Perference(P2)'},
            {'label': 'Medium (P1)', 'value': 'Medium (P1)'},
            {'label': 'Medium (P2)', 'value': 'Medium (P2)'},
            ]),
html.Div(dcc.Graph(id="outputoutputfig")),
]),
]),

pivottable=html.Div(dash_pivottable.PivotTable(
            id='component_id',
           data=[df.columns.values.tolist()] + df.values.tolist(),
            cols=['Easy Payment option'],
            rows=['Bad past experience'],
            vals=["Educational Qualifications"],
            rendererName="Table Heatmap",
            aggregatorName="Count"
                   ))

datatable = dbc.Container([
    dcc.Markdown('DataTable', style={'textAlign':'center'}),

    dbc.Label("Show number of rows"),
    row_drop := dcc.Dropdown(value=10, clearable=False, style={'width':'35%'},
                             options=[10, 25, 50, 100]),

    my_table := dash_table.DataTable(
        columns=[
            {'name': i, 'id': i} for i in df.columns],

        data=df.to_dict('records'),
        # filter_action='native',
        page_size=10,

        style_data={
            'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        }
    ),
    dbc.Row([
        dbc.Col([
            continent_drop := dcc.Dropdown([x for x in sorted(df['Educational Qualifications'].unique())],clearable=True)
        ], width=3),
        dbc.Col([
            pop_slider := dcc.Slider(18 ,33, 1, marks={'33':'33'}, value=10,
                                   tooltip={"placement": "bottom", "always_visible": True})
        ], width=3),
        dbc.Col([
            lifeExp_slider := dcc.Slider(1, 6, 1, marks={'10':'10'}, value=2,
                                   tooltip={"placement": "bottom", "always_visible": True})
        ], width=3),

    ], justify="between"),

])



app.layout =html.Div([
html.Div([html.H4([html.Img(src='/assets/1.png',style={'float':'center','width':'4%','height':'4%'}),html.H2('Food Delivery Analysis Dashboard',style={'float':'right','font-size':'200%'})])],style={'float':'center','width':'98%','height':'100px','text-color':'#000000','background-color':'#28282B','margin-top':'-20px','padding': '20px','border': '2px solid black','border-radius': '25px',}), 
    
html.Div([
    html.H1(''),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Let\'s Break it For you', value='tab-1'),
        dcc.Tab(label='Go And Dive', value='tab-2'),
        dcc.Tab(label='Go And Dive Once More', value='tab-3'),
        
    ]),
    html.Div(id='tabs_output'),
]),
])

 
@app.callback(
Output("tabs_output", "children"), 
Input("tabs", "value"))
def tabs(value):
    if value == 'tab-1':
        return y
    elif value == 'tab-2':
        return pivottable
    elif value == 'tab-3':
        return datatable

    
@app.callback(
Output("figure_data","figure"),
Input("drop", "value"))
def question9(drop):
    cust_fig = px.sunburst(df, path=drop)
    cust_fig.update_traces(textinfo='label+percent entry')
    return cust_fig


@app.callback(
Output("figure_data1","figure"),
Input("radio", "value"))

def question9(radio):
    if radio=='Age':
        fig_hist=px.histogram(df,x='Age')
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif radio=='Family size':
        fig_hist=px.histogram(df,x='Family size')
        fig_hist.update_layout(font_size=19)
        return fig_hist


@app.callback(
Output("outputoutputfig", "figure"),
Input("outputoutput", "value"))
def question9(value1):
    if value1=='Meal(P1)':
        fig_hist=px.pie(df, names="Output", color="Meal(P1)")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Meal(P2)':
        fig_hist=px.pie(df, names="Output", color="Meal(P2)")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Perference(P1)':
        fig_hist=px.pie(df, names="Output", color="Perference(P1)")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Perference(P2)':
        fig_hist=px.pie(df, names="Output", color="Perference(P2)")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Medium (P1)':
        fig_hist=px.pie(df, names="Output", color="Medium (P1)")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Medium (P2)':
        fig_hist=px.pie(df, names="Output", color="Medium (P2)")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    else:
        fig_hist=px.pie(df, names="Output")
        fig_hist.update_layout(font_size=19)
        return fig_hist

        
@app.callback(
Output("graph12", "figure"), 
Input("heat1", "value"))
# Input("heat2", "value")])

def heatmap(value1):
    col=['Age','Gender','Marital Status','Occupation','Monthly Income']
    if value1==None:
        fig = go.Figure(data=go.Heatmap(x=col,y=col,z=df.corr(),xgap=2,ygap=2, colorscale='Viridis'))
        return fig
    else:
        fig = go.Figure(data=go.Heatmap(x=col,y=col,z=df.corr(),xgap=2,ygap=2, colorscale='Viridis'))
        return fig


@app.callback(
    [Output(my_table, 'data'),
    Output(my_table, 'page_size')],
    [Input(continent_drop, 'value'),
    Input(pop_slider, 'value'),
    Input(lifeExp_slider, 'value'),
    Input(row_drop, 'value')]
)
def update_dropdown_options(cont_v, pop_v, life_v, row_v):
    dff = df.copy()

    if cont_v:
        dff = dff[dff['Educational Qualifications']==cont_v]
 
    dff = dff[(dff['Age'] >= pop_v) & (dff['Age'] < 33)]
    dff = dff[(dff['Family size'] >= life_v) & (dff['Family size'] < 6)]

    return dff.to_dict('records'), row_v

if __name__ == '__main__':
        app.run_server()

        
    
# In[ ]:
