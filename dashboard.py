from turtle import width
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import logging
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import r2_score
import nltk
import pickle
import base64
from urllib.parse import quote as urlquote
import os
from flask import Flask, send_from_directory
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

UPLOAD_DIRECTORY = "/project/app_uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash()
server = app.server

@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

logging.getLogger('werkzeug').setLevel(logging.INFO)

### Estilos

tab_style1 = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    'background-color': '#F2F2F2',
    'color': '#323232',
    'margin':'5px',
 
}
 
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'margin':'5px',
    'padding': '6px',
    'border-radius': '15px',
    'box-shadow': '4px 4px 4px 4px lightgrey',
}

tab_style= {
    'color':'#121212',
    #"background-image": "linear-gradient(90deg,#82e0aa,#f9e79f)",
    'width':'50%',
    'margin':'center',
    'margin-bottom':'4px',
    'margin-top':'5px',
    "font-size": "20px",
    "border-width": "1px 2px 5px 2px",
    "border-color": "#172B83",
    "font-weight": "bold",
}
drop_style={
    'color':'#121212',
    #"background-image": "linear-gradient(90deg,#82e0aa,#f9e79f)",
    'width':'80%',
    'margin':'auto',
    'margin-bottom':'4px',
    'margin-top':'5px',
    "font-size": "20px",
    "border-radius":"30px",
    "border-width": "1px 2px 5px 2px",
    "border-color": "#172B83",
    "font-weight": "bold",
    #"padding-top":"10px"
}
block_style= {
    "width": "33%",
    "heigh": "200px",
    "text-align": "center",
    "display": "inline-block",
    "float": "left"
    #"border-style": "ridge",
    #"border-color": "fuchsia"
}
block_style2= {
    "width": "30%",
    "heigh": "200px",
    "text-align": "center",
    "display": "inline-block",
    "float": "center"
    #"border-style": "ridge",
    #"border-color": "fuchsia"
}

#### Aplicacion

app.layout = html.Div([
    html.Div(
        [
            html.H1( # Primera fila
                children = [
                    'Default Predictor'
                ],
                id = "titulo",
                style = { 
                    "text-align": "center", # Alineo el texto al centro
                    "font-size": "50px",
                    "-webkit-text-fill-color": "transparent",
                    "text-fill-color": "transparent",
                    "-webkit-background-clip": "text",
                    "background-clip": "text",
                    "background-image":  "linear-gradient(180deg,#172B83,#119DFF)",
                    "color": "#ec7063"
                }
            )
        ],
        style={
            "width":'900px',
            "margin":"auto",        

        }
    ),
    html.Div(
        dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
            dcc.Tab(label='Manual', value='tab-1',style = tab_style1, selected_style = tab_selected_style),
            dcc.Tab(label='Fichero', value='tab-2',style = tab_style1, selected_style = tab_selected_style),
        ], colors={
            "primary": "#4B39DE",
            #"background": "linear-gradient(180deg,#4B39DE,#ec7063,#DE397F)",
        }),
        style={
            "font-weight": "bold",
            "font-size":"30px",
        }
    ),
    html.Div(id='tabs-content-props', style={"width":"95%","margin":"auto"})
],
    style={
        "font-family":'"Century Gothic", CenturyGothic, Geneva, AppleGothic, sans-serif',
        "color":"white",
    }
)
@app.callback(Output('tabs-content-props', 'children'),
              Input('tabs-styled-with-props', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return [html.Div([
                html.H1('Inserte aquí los datos para realizar la predicción', style={"padding-top":"20px"}),    
                html.Div([
                    html.H3(
                        children = [
                            "Número de empleados"
                        ],
                        id = "titulo_1",
                        style = {
                            "display": "block",
                            "text-align": "center"
                        }
                    ),
                    html.P("Valor entero positivo:"),
                    dcc.Input(
                                        id="input_empleados",
                                        type="number",
                                        min=0,
                                        placeholder="Numero de empleados",
                                        value="100",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style
                    ),


                    html.Div([
                        html.H3(
                            children = [
                                "Año fundación"
                            ],
                            id = "titulo_2",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Valor entero positivo:"),
                        dcc.Input(
                                        id="input_fecha",
                                        type="number",
                                        min=1800,
                                        max=2022,
                                        placeholder="Fecha",
                                        value="2000",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style
                    ),


                    html.Div([
                        html.H3(
                            children = [
                                "Tamaño de la emrpesa"
                            ],
                            id = "titulo_3",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Selecciona tamaño:"),
                        dcc.Dropdown(
                            id='tipo', 
                            options=[{'value': x, 'label': x} 
                                    for x in ["Micro",	"Pequeña",	"Mediana", "Grande"]],
                            value='Mediana',  
                            style=drop_style
                        ),
                    ],
                        style=block_style
                        ), 

                    html.H1('Ratios:', style={"padding-top":"200px"}),

                    html.Div([
                        html.H3(
                            children = [
                                "Working Capital/Total Assets"
                            ],
                            id = "titulo_4",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Valor numérico:"),
                        dcc.Input(
                                        id="input_1",
                                        type="number",
                                        placeholder="Working Capital/Total Assets",
                                        value="0",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style2
                    ),

                    html.Div([
                        html.H3(
                            children = [
                                "Earnings/Total Assets"
                            ],
                            id = "titulo_5",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Valor numérico:"),
                        dcc.Input(
                                        id="input_2",
                                        type="number",
                                        placeholder="Earnings/Total Assets",
                                        value="0",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style2
                    ),

                    html.Div([
                        html.H3(
                            children = [
                                "EBIT/Total Assets"
                            ],
                            id = "titulo_6",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Valor numérico:"),
                        dcc.Input(
                                        id="input_3",
                                        type="number",
                                        placeholder="Earnings Before Interest and Tax/Total Assets",
                                        value="0",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style2
                    ),

                    html.Div([
                        html.H3(
                            children = [
                                "Book Value/Total Liabilities"
                            ],
                            id = "titulo_7",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Valor numérico:"),
                        dcc.Input(
                                        id="input_4",
                                        type="number",
                                        placeholder="Book Value/Total Liabilities",
                                        value="0",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style2
                    ),

                    html.Div([
                        html.H3(
                            children = [
                                "Sales/Total Assets"
                            ],
                            id = "titulo_8",
                            style = {
                                "display": "block",
                                "text-align": "center"
                            }
                        ),
                        html.P("Valor numérico:"),
                        dcc.Input(
                                        id="input_5",
                                        type="number",
                                        placeholder="Sales / Total Assets",
                                        value="0",
                                        style=tab_style
                                    ),
                    ],
                    style=block_style2
                    )   

                ],
                style = {
                    "width": "50%",
                    "text-align": "center",
                    "display": "inline-block",
                    "color": "#172B83"     
            }),html.Div([
                html.H1('Pulse aquí para realizar la predicción', style={"padding-top":"20px"}),
                html.Button("Predicción", id='update-button', n_clicks=0, 
                style={'width': '40%', 'height': '40px', 
                    'cursor': 'pointer', 'border': '0px',  
                    'border-radius': '5px', 'background-color': 
                    '#119DFF', 'color': 'white', 'text-transform': 
                    'uppercase', 'font-size': '20px', "margin-left": "25%" , "margin-right": "25%"}),
                    html.Div([
                            dcc.Graph(id="pred1", style={'display':'none'}),
                            dcc.Graph(id="spiderplot", style={'display':'none'}),
                        ],
                        style={
                            "width":"100%",
                            "display":"inline-block",
                            "padding-top": "35px"
                        }
                    )
            ],style = {
                    "width": "50%",
                    "text-align": "center",
                    "display": "inline-block",
                    "color": "#172B83",
                    "vertical-align": "top"      
            }),
            html.H3(
                '© Juan Blazquez & Nicolás Oriol',
                style={
                    "text-align":"right",
                    "color": "#172B83",
                }
            )]
            

    elif tab == 'tab-2':
        return html.Div([
            html.H1('Inserte aquí fichero Excel', style={"padding":"20px"}),
    dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a file to upload."]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.H2("File Uploaded"),
        html.Ul(id="file-list"),
            html.Button("Descargar predicciones", id='csv-button', n_clicks=0, 
                style={'width': '30%', 'height': '40px', 
                    'cursor': 'pointer', 'border': '0px',  
                    'border-radius': '5px', 'background-color': 
                    '#119DFF', 'color': 'white', 'text-transform': 
                    'uppercase', 'font-size': '20px', "margin-left": "25%" , "margin-right": "25%"}),
                html.Div(id='output-data-upload'),
                dcc.Download(id="download-dataframe-csv"),
                        html.H3(
                            '© Juan Blazquez & Nicolás Oriol',
                            style={
                                "text-align":"right",
                                "color": "#172B83",
                            }
                        )
            ],style = {
                                "width": "100%",
                                "text-align": "center",
                                "display": "inline-block",
                                "color": "#172B83",
                                "vertical-align": "top"      
                        })

### Truco para cambiar el fondo

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Trabajo final</title>
        {%favicon%}
        {%css%}
    </head>
    <body style="background-color: #F7F3D5 ;">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

### Cargamos modelo y Scaler

modelo = tf.keras.models.load_model("resources\modelos\model_nn.h5")


file = open("resources\modelos\scaler.pkl",'rb')
scaler = pickle.load(file)
file.close()

#### Funciones

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

def clear_files():
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        os.remove(path)
    return None

def leeFichero():
    df=None
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        df=pd.read_excel(path)
    return df

#### Callbacks

@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    clear_files()
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No file yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]

@app.callback(
    [Output("output-data-upload", "children"), Output("download-dataframe-csv", "data")],
    [Input("csv-button", "n_clicks")])
def generate_chart(n_clicks):
    descarga=None
    try:
        if n_clicks >0:
            df=leeFichero()
            if df is None:
                children=["No se encontró fichero"]
            else:
                try:
                    muestra_scaled=scaler.transform(df.to_numpy())
                    prediccion_1=modelo.predict(muestra_scaled)
                    pred=np.where(prediccion_1 > 0.5, 1, 0)
                    df["Prob_0"]=1-prediccion_1
                    df["Prob_1"]=prediccion_1
                    df["Pred"]=pred
                    df.index.name='Empresa'
                    children=["¡Descarga realizada!"]
                    return children,dcc.send_data_frame(df.to_excel, "prediction.xlsx", sheet_name="Prediction")         
                except:
                   children=["Fallo al realizar la predcción"] 
    except:
        children=["Error en descarga"]
    return children,descarga
@app.callback(
    [Output("pred1", "figure"),Output("pred1","style"),Output("spiderplot", "figure"),Output("spiderplot","style")],
    [Input("update-button", "n_clicks")],
    [State("input_empleados", "value"),State("input_fecha", "value"),State("tipo", "value"),
    State("input_1", "value"),State("input_2", "value"),State("input_3", "value"),State("input_4", "value"),State("input_5", "value")])
def generate_chart(n_clicks,input_empleados_val,input_fecha_val,tipo_val,input_1_val,input_2_val,input_3_val,input_4_val,input_5_val):
    tipo_empresa=str(tipo_val)
    if (tipo_empresa=="Pequeña"):
        tipo=[1,0,0]
        reference=0.046693997
        empresas="pequeñas"
    elif (tipo_empresa=="Mediana"):
        tipo=[0,1,0]
        reference=0.045075535
        empresas="medianas"
    elif (tipo_empresa=="Grande"):
        tipo=[0,0,1]
        reference=0.383756649
        empresas="grandes"
    else:
        tipo=[0,0,0]
        reference=0.129807994
        empresas="micros"

    if str(input_empleados_val).isdigit():
        empleados=int(input_empleados_val)
    else:
        empleados=100

    if str(input_fecha_val).isdigit():
        fecha=int(input_fecha_val)
    else:
        fecha=2000

    try:
        ratio1 = float(input_1_val)
    except ValueError:
        ratio1=0

    try:
        ratio2 = float(input_2_val)
    except ValueError:
        ratio2=0
    
    try:
        ratio3 = float(input_3_val)
    except ValueError:
        ratio3=0

    try:
        ratio4 = float(input_4_val)
    except ValueError:
        ratio4=0

    try:
        ratio5 = float(input_5_val)
    except ValueError:
        ratio5=0

    categories = ["Working Capital/Total Assets","Retained Earnings/Total Assets","EBIT/Total Assets","Book Value/Liabilities","Sales/Total Assets"]
    suavizador=np.array([200,1,1,1000,10])
    r=np.array([ratio1,ratio2,ratio3,ratio4,ratio5])
    r1=np.array([-50.21068659,-219.7636339,-1.882594179,-1470.006441,32.63329932])/suavizador
    r2=np.array([0.115317113,0.060361681,-14.18088617,-4.968253902,2.953914455])/suavizador
    r3=np.array([0.18818863,0.265890216,0.124307138,-345.1943461,1.668263164])/suavizador
    r4=np.array([0.248826911,0.141365271,2.352097951,-36969.72659,0.35145209])/suavizador

    fig= go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=categories,
        marker_color="#3EE04C",
        marker_line_color="#3EE04C",
        fill='toself',
        name='Valores predicción'
    ))
    if tipo == [0,0,0]:
        fig.add_trace(go.Scatterpolar(
            r=r1,
            theta=categories,
            marker_color="#EAAC39",
            marker_line_color="#EAAC39",
            fill='toself',
            name='Media Empresas Micro'
        ))
    if tipo == [1,0,0]:
        fig.add_trace(go.Scatterpolar(
            r=r2,
            theta=categories,
            marker_color="#EAAC39",
            marker_line_color="#EAAC39",
            fill='toself',
            name='Media Empresas Pequeñas'
        ))
    if tipo == [0,1,0]:
        fig.add_trace(go.Scatterpolar(
            r=r3,
            theta=categories,
            marker_color="#EAAC39",
            marker_line_color="#EAAC39",
            fill='toself',
            name='Media Empresas Medianas'
        ))
    if tipo == [0,0,1]:
        fig.add_trace(go.Scatterpolar(
            r=r4,
            theta=categories,
            marker_color="#EAAC39",
            marker_line_color="#EAAC39",
            fill='toself',
            name='Media Empresas Grandes'
        ))

    fig.update_layout(title=f'RadarPlot de los Ratios para las clases {empresas} (Números escalados)', height=450, 
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#172B83', title_x = 0.5, 
                    title_font_color='#172B83',legend_font_color='#172B83',
                    legend_title_font_color='#172B83', 
                    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#119DFF', zerolinecolor='#172B83')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#119DFF', zerolinecolor='#172B83')
    fig.update_polars(bgcolor='rgba(0,0,0,0)')

    reference=0.5
    muestra=np.array([empleados,fecha,tipo[0],tipo[1],tipo[2],ratio1,ratio2,ratio3,ratio4,ratio5])
    muestra_scaled=scaler.transform(muestra.reshape(1,-1))
    prediccion=modelo.predict(muestra_scaled)[0][0]
    fig1 = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prediccion,
        mode = "gauge+number+delta",
        title = {'text': "Default"},
        delta = {'reference': reference, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, reference], 'color': "limegreen"},
                    {'range': [reference, 1], 'color': "tomato"}]}))

    fig1.update_layout(title=f'Predicción, comparada con las empresas {empresas}', height=450, 
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#172B83', title_x = 0.5, 
                        title_font_color='#172B83',legend_font_color='#172B83',
                        legend_title_font_color='#172B83', 
                        )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#119DFF', zerolinecolor='#172B83')
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#119DFF', zerolinecolor='#172B83')
    fig1.update_polars(bgcolor='rgba(0,0,0,0)')

    
    return fig1, {'display':'block'}, fig, {'display':'block'}

if __name__ == '__main__':
    app.run_server()