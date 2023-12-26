import dash
from dash import dcc,html,Dash
from dash import dash_table
import base64
import datetime
import io
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go   
import pandas as pd
import numpy as np
import dash_pivottable
import dash_bootstrap_components as dbc
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import export_graphviz
from graphviz import Source
import pickle
import cloudpickle


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


app.title= 'Bank Customers Churn Dashboard'



df=pd.read_csv('Bank_Churn_Modelling.csv')
# df.drop(['RowNumber','CustomerId'], axis=1, inplace=True)
# df.drop('Surname',axis=1,inplace=True)

# df_copy1=df.copy()
# en=OneHotEncoder(drop='first')
# geographyTransformed=en.fit_transform(df[['Geography','Gender']]).toarray()
# geographyTransformed=pd.DataFrame(geographyTransformed,columns=en.get_feature_names_out(input_features=en.feature_names_in_))
# df_copy1=pd.concat([df_copy1,geographyTransformed],axis=1).drop(['Geography','Gender'],axis=1)
# df_copy=df_copy1.copy()
# scaler=MinMaxScaler()
# df_copy=pd.DataFrame(scaler.fit_transform(df_copy),columns=scaler.get_feature_names_out(input_features=scaler.feature_names_in_))

# X=df_copy.drop('Exited',axis=1)
# y=df_copy['Exited']
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
# xgb=XGBClassifier(n_estimators=13, max_depth=10, learning_rate=1.920, objective='binary:logistic')
# xgb.fit(X_train,y_train)
# preds=xgb.predict(X_test)
# # with open("model.pkl", "wb") as f:
# #     pickle.dump(xgb, f)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
        # Assume that the user uploaded a CSV file
            df1 = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
        # Assume that the user uploaded an excel file
            df1 = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    
    en=pickle.load(open('en.pkl','rb'))
    scaler=pickle.load(open('scaler.pkl','rb'))
    df1.drop(['RowNumber','CustomerId'], axis=1, inplace=True)
    df1.drop('Surname',axis=1,inplace=True)
    geographyTransformed=en.transform(df1[['Geography','Gender']]).toarray()
    geographyTransformed=pd.DataFrame(geographyTransformed,columns=en.get_feature_names_out(input_features=en.feature_names_in_))
    df_copy2=pd.concat([df1,geographyTransformed],axis=1).drop(['Geography','Gender'],axis=1)
    df_copy3=df_copy2.copy()
    df_copy3=pd.DataFrame(scaler.transform(df_copy3),columns=scaler.get_feature_names_out(input_features=scaler.feature_names_in_))
    # df_copy3.drop('Exited',axis=1,inplace=True)
    model=pickle.load(open("model.pkl", "rb"))
    return html.Div(model.predict(df_copy3)[0])
                    

y=html.Div([
    html.Div([html.H3("How many no. of Exited Customers vs Not Exited Customers")],style={'text-align':'center','display':'inline-block','width':'100%','font-weight':'900','font-size':'150%','text-color':'#FFFFFF'}),
    html.Div([
    html.Div([html.H3("Exited",style={'width':'100%','font-weight':'900','font-size':'150%','font-family': '"Fantasy", Times, serif'},id="tooltip-target1"),dbc.Tooltip(df[df['Exited']==1].count().iloc[0],style={'display':'inline-block','width':'200%','font-weight':'900','font-size':'150%'},target="tooltip-target1")]),
    html.Div([html.H3("Not Exited",style={'width':'100%','font-weight':'900','font-size':'150%','font-family': '"Fantasy", Times, serif'},id="tooltip-target2"),dbc.Tooltip(df[df['Exited']==0].count().iloc[0],style={'display':'inline-block','width':'200%','font-weight':'900','font-size':'150%'},target="tooltip-target2")]),
],style={'margin-bottom':'20px','margin-left':'35%','padding': '20px','border': '2px solid black','border-radius': '25px','width':'25%','height':'25%','background-color':'#FF5733', 'text-color':'#00FFFF','text-align':'center', 'float':'center'}),

html.Div([dcc.Dropdown(id='dropdown1',
        options=[
        {'label': i, 'value': i} for i in ['Gender','Geography']],
    #value=['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'],
    value=['Gender','Geography'],
    multi=True,
    clearable=False,
    searchable=True,),
          html.Div([dcc.RadioItems(id='radio',
    options=[
        {'label': i, 'value': i} for i in ['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']],
    value='CreditScore',style={'font-size':25})]),
html.Div(dcc.Graph(id="sunburstgraph1",style={'height':'25'})),
],style={'display':'inline-block','width':'100%'}),



html.Div([html.H3(" What do you want to check the exited customers with",style={'text-align':'center','display':'inline-block','width':'100%','font-weight':'900','font-size':'150%'}), 
          dcc.Dropdown(id='drop_down2',
            options=[
                  {'label': i, 'value': i} for i in ['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
            ]),
html.Div(dcc.Graph(id="piechart1"))]),
]),

predict= html.Div([
dcc.Upload(
        id='upload-data',
        children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
        ]),
        style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
         },
        # Allow multiple files to be uploaded
        multiple=False
),

html.Div(id='output-data-upload'),
])


predict_choice= html.Div([
    html.Div([
    dbc.Row([
    dbc.Col(dbc.Label("RowNumber", html_for="RowNumber", width=2)),
    dbc.Col(dbc.Input(id='RowNumber', type='text',placeholder='Please Enter Random Number', n_submit=0)),   
            ]),

            dbc.Row([
    dbc.Col(dbc.Label("CustomerId", html_for="CustomerId", width=2)),
    dbc.Col(dbc.Input(id='CustomerId', type='text',placeholder='Please Enter a Random CustomerId', n_submit=0)),   
            ]),

                    dbc.Row([
    dbc.Col(dbc.Label("Surname", html_for="Surname", width=2)),
    dbc.Col(dbc.Input(id='Surname', type='text',placeholder='Please Enter a Surname', n_submit=0)),   
            ]),

                    dbc.Row([
    dbc.Col(dbc.Label("CreditScore", html_for="CreditScore", width=2)),
    dbc.Col(dbc.Input(id='CreditScore', type='text',placeholder='Please Enter a CreditScore', n_submit=0)),   
            ]),

                    dbc.Row([
    dbc.Col(dbc.Label("Geography", html_for="Geography", width=2)),
    dbc.Col(dbc.Input(id='Geography', type='text',placeholder='Please Enter a Country', n_submit=0)),   
            ]),

                    dbc.Row([
    dbc.Col(dbc.Label("Gender", html_for="Gender", width=2)),
    dbc.Col(dbc.Input(id='Gender', type='text',placeholder='Please Enter a Gender', n_submit=0)),   
            ]),

                    dbc.Row([
    dbc.Col(dbc.Label("Age", html_for="Age", width=2)),
    dbc.Col(dbc.Input(id='Age', type='text',placeholder='Please Enter an Age', n_submit=0)),   
            ]),
                            dbc.Row([
    dbc.Col(dbc.Label("Tenure", html_for="Tenure", width=2)),
    dbc.Col(dbc.Input(id='Tenure', type='text',placeholder='Please Enter a Tenure Score Up to 10', n_submit=0)),   
            ]),
                            dbc.Row([
    dbc.Col(dbc.Label("Balance", html_for="Balance", width=2)),
    dbc.Col(dbc.Input(id='Balance', type='text',placeholder='Please Enter a Bank Balance', n_submit=0)),   
            ]),
                            dbc.Row([
    dbc.Col(dbc.Label("NumOfProducts", html_for="NumOfProducts", width=2)),
    dbc.Col(dbc.Input(id='NumOfProducts', type='text',placeholder='Please Enter NumOfProducts Up to 4', n_submit=0)),   
            ]),
                                    dbc.Row([
    dbc.Col(dbc.Label("HasCrCard", html_for="HasCrCard", width=2)),
    dbc.Col(dbc.Input(id='HasCrCard', type='text',placeholder='Please Enter Has Credit Card Value: 0 or 1 where 0 means No & 1 means Yes', n_submit=0)),   
            ]),
                                    dbc.Row([
    dbc.Col(dbc.Label("IsActiveMember", html_for="IsActiveMember", width=2)),
    dbc.Col(dbc.Input(id='IsActiveMember', type='text',placeholder='Please Enter Is Active Member Value: 0 or 1 where 0 means No & 1 means Yes', n_submit=0)),   
            ]),
                                            dbc.Row([
    dbc.Col(dbc.Label("EstimatedSalary", html_for="EstimatedSalary", width=2)),
    dbc.Col(dbc.Input(id='EstimatedSalary', type='text',placeholder='Please Enter an EstimatedSalary', n_submit=0)),   
            ]),
        
    dbc.Row([
        dbc.Label("Exited", html_for="Exited", width=1),
        dbc.Button("Predict",id='Predict', n_clicks=0, color="primary", className="me-1" )]),
]),

html.Div(id='prediction',style={'text-align':'center','display':'inline-block','width':'100%','font-weight':'900','font-size':'150%'})
])


app.layout =html.Div([
html.Div([html.H4([html.H2('Bank Churned Customers Analysis Dashboard',style={'font-size':'200%'})])],style={'float':'center','width':'98%','height':'100px','text-color':'#00FFFF','background-color':'#FF5733','margin-top':'-20px','padding': '20px','border': '2px solid white','border-radius': '25px',}), 
    
html.Div([
    html.H1(''),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Let\'s Break it For you', value='tab-1'),
        dcc.Tab(label='Predict By Uploading CSV File', value='tab-2'),
        dcc.Tab(label='Enter Data To Predict', value='tab-3')        
        


        
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
    if value == 'tab-2':
        return predict
    if value == 'tab-3':
        return predict_choice
        
        
@app.callback(
Output("sunburstgraph1","figure"),
[Input("dropdown1", "value"),
 Input("radio", "value")])

def sunburstdrop(drop,rad):
    if rad=='Tenure':
        cust_fig = px.sunburst(df, path=drop,values='Tenure')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    elif rad=='Balance':
        cust_fig = px.sunburst(df, path=drop,values='Balance')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    elif rad=='NumOfProducts':
        cust_fig = px.sunburst(df, path=drop,values='NumOfProducts')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    elif rad=='HasCrCard':
        cust_fig = px.sunburst(df, path=drop,values='HasCrCard')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    elif rad=='IsActiveMember':
        cust_fig = px.sunburst(df, path=drop,values='IsActiveMember')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    elif rad=='EstimatedSalary':
        cust_fig = px.sunburst(df, path=drop,values='EstimatedSalary')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    elif rad=='Exited':
        cust_fig = px.sunburst(df, path=drop,values='Exited')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
    else:
        cust_fig = px.sunburst(df, path=drop,values='CreditScore')
        cust_fig.update_traces(textinfo='label+percent entry')
        return cust_fig
@app.callback(
Output("barradio","figure"),
Input("radio", "value"))
def barradio(radio1):
    if radio1=='Gender':
        fig_hist=px.bar(df, x="Gender", y="Exited", color='Exited', title="Exited To Gender")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif radio1=='Geography':
        fig_hist=px.bar(df, x="Geography", y="Exited", color='Gender', title="Exited To Geography")
        fig_hist.update_layout(font_size=19)
        return fig_hist

@app.callback(
Output("piechart1", "figure"),
Input("drop_down2", "value"))
def piedrop(value1):
    if value1=='CreditScore':
        fig_hist=px.histogram(df, color="Exited", x="CreditScore")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Tenure':
        fig_hist=px.histogram(df, color="Exited", x="Tenure")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='Balance':
        fig_hist=px.histogram(df, color="Exited", x="Balance")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='NumOfProducts':
        fig_hist=px.histogram(df, color="Exited", x="NumOfProducts")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='HasCrCard':
        fig_hist=px.histogram(df, color="Exited", x="HasCrCard")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='IsActiveMember':
        fig_hist=px.histogram(df, color="Exited", x="IsActiveMember")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    elif value1=='EstimatedSalary':
        fig_hist=px.histogram(df, color="Exited", x="EstimatedSalary")
        fig_hist.update_layout(font_size=19)
        return fig_hist
    else:
        fig_hist=px.histogram(df, color="Exited", x="Exited")
        fig_hist.update_layout(font_size=19)
        return fig_hist
        
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(list_of_contents, list_of_names, list_of_dates)] 
        return children

@app.callback(Output('prediction', 'children'),
           Input('Predict', 'n_clicks'),
           [State('RowNumber', 'value'),
           State('CustomerId', 'value'),
           State('Surname', 'value'),
           State('CreditScore', 'value'),
           State('Geography', 'value'),
           State('Gender', 'value'),
           State('Age', 'value'),
           State('Tenure', 'value'),
           State('Balance', 'value'),
           State('NumOfProducts', 'value'),
           State('HasCrCard', 'value'),
           State('IsActiveMember', 'value'),
           State('EstimatedSalary', 'value')],)
             #prevent_initial_call=True,)
def update_output(n_clicks,RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    # en=pickle.load(open('en.pkl','rb'))
    scaler=pickle.load(open('scaler.pkl','rb'))
    list_of_contents1=[RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    list_of_contents1=pd.Series(list_of_contents1)
    pd.options.mode.use_inf_as_na = True

    # dk=df.copy()
    # dk.drop('Exited',axis=1, inplace=True)
    # d= pd.DataFrame(list_of_contents1, columns=[dk.columns]).reset_index(drop = True)

    # list_of_contents1.reset_index(drop=True, inplace=True)
    d= pd.DataFrame([list_of_contents1],columns=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    
    # d.reset_index(drop=True, inplace=True)
    # df2= pd.DataFrame(list_of_contents1, columns=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
    if not RowNumber and not CustomerId and not Surname and not CreditScore and not Geography and not Gender and not Age and not Tenure and not Balance and not NumOfProducts and not HasCrCard and not IsActiveMember and not EstimatedSalary:
        return None
    else:
        d.drop(['RowNumber','CustomerId'], axis=1, inplace=True)
        d.drop('Surname',axis=1,inplace=True)
        #geographyTransformed=en.transform(d[['Geography','Gender']]).toarray()
        #geographyTransformed=pd.DataFrame(geographyTransformed,columns=en.get_feature_names_out(input_features=en.feature_names_in_))
        #df_copy2=pd.concat([d,geographyTransformed],axis=1).drop(['Geography','Gender'],axis=1)
        #df_copy3=df_copy2.copy()
        d.drop(['Geography','Gender'], axis=1, inplace=True)
        df_copy3=d.copy()
        df_copy3=pd.DataFrame(scaler.transform(df_copy3),columns=scaler.get_feature_names_out(input_features=scaler.feature_names_in_))
        # df_copy3['Surname']=df_copy3['Surname'].astype(float) # cause of the np.nan error we have to convert to float before entering it to the model
        # df_copy3['Geography']=df_copy3['Geography'].astype(float)
        # df_copy3['Gender']=df_copy3['Gender'].astype(float)
        # df_copy3=df_copy3.astype(float, errors='ignore')
        # df_copy3.drop('Exited',axis=1,inplace=True)
        model=pickle.load(open("model.pkl", "rb"))
        return html.Div(model.predict(df_copy3))
        # return df_copy3.to_json()
        
        
if __name__ == '__main__':
    app.run_server(debug=False)

