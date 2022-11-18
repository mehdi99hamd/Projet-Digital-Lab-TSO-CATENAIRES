import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import pandas as pd
import dash_bootstrap_components as dbc
import grasia_dash_components as gdc
from dash import Dash, html, Input, Output, callback_context, State
import numpy as np
import laspy
import model
import pye57
import os
import tensorflow as tf 
from tensorflow import keras
import sys

external_stylesheets = [dbc.themes.BOOTSTRAP, 
    "https://use.fontawesome.com/releases/v5.0.6/css/all.css",
]

external_scripts = [
    {'src': "https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"}
]

def find_data_file(filename):
    if getattr(sys, 'frozen', False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


app = dash.Dash(__name__, assets_folder=find_data_file('assets/'), external_scripts = external_scripts, 
                external_stylesheets=external_stylesheets)

app.title='Détection Des Caténaires'

hidden = ' hidden'

visible = ' visible'

server = app.server

app.layout = html.Div([
    html.Div([
            html.Img(src='/assets/logo.png', className="logo-img"),
            html.H1('Détection Des Caténaires', className='header'), 
            html.Img(src='/assets/image4.png', className="tso1"),
            html.Img(src='/assets/image3.png', className="tso2"),
            html.Img(src='/assets/logo2.png', className="logo-img2"),
        ], className='nav'),
    html.Div([
        html.Div([], className="design1"),
        html.Div([], className="design2"),
        html.Div([], className="design3"),
        html.Div([], className="design4"),
        #html.Div([
        #   html.Div([html.P('1')],id = 'etape-1', className='etape-1' + active),
        #    html.Div([html.P('2')],id = 'etape-2', className='etape-2'+ inactive),
        #    html.Div([html.P('3')],id = 'etape-3', className='etape-3'+ inactive),
        #], className = 'etapes'),
        html.Div([
            html.Div([
                html.Div([html.P("Lecture D'un Seul Fichier")], className="button-upload1"),
                dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            [html.I(className="fas fa-upload icon"),html.Br(),"Glisser Et Déposer Le Fichier Ici",
                            html.Br(),html.Span("Ou"),
                            html.Br(), html.Button('Selectioner Les Fichiers', className="button-upload2")]
                        ), multiple=False, className="upload-data"
                ),
                html.Div([], id = 'output-tt', className="button-upload-e572 hidden"),
                html.Div([
                    html.Div([html.P("Chemin Du Fichier e57")], id = 'output-data-upload', className="button-upload-e57"),
                    dcc.Input(id="input3", type="text", placeholder="", debounce=True, className="input-path-e57"),
                    dcc.Input(id="content", type="hidden", placeholder="", debounce=True, className="input-path-out"),
                    dcc.Input(id="filename", type="hidden", placeholder="", debounce=True, className="input-path-out"),
                ], id = 'e57', className='e57'),
            ], className="left"),
            html.Div([
                html.Div([html.P("Lecture De Plusieurs Fichiers")], className="button-upload"),
                html.H5("Entrer Le Chemin Des Fichiers", className="title1"),
                dcc.Input(id="input2", type="text", placeholder="", debounce=True, className="input-path"),
                html.Button('Suivant', id='start',className="start", n_clicks=0),
                html.Div([], id = 'error1', className = 'error1'),
            ], className="right")
        ],id = 'contact-box', className="contact-box visible"),
        html.Div([
            html.Div([html.P("Spécifier Le Chemin De Sortie Des Fichiers")], className="button-onload"),
            dcc.Input(id="input4", type="text", placeholder="", debounce=True, className="input-path-out"),
            html.Button('Commencer', id='suivant',className="suivant", n_clicks=0),
            html.Div([], id = 'error', className = 'error'),
        ],id = 'contact-box-1', className="contact-box-1 hidden"),
        html.Div([
            html.Div([html.P('Le Traitement a Terminé')],id = 'termine', className = 'termine hidden'),
            html.Div([html.P("Résultat Du Traitement Des Fichiers")], className="resultat"),
            html.Div([],id = 'output', className= 'resultats'),
            html.Div([
                html.Div([html.Div(), html.Div(), html.Div(), html.Div()], className = "lds-ring"),
                html.P('Traitement En Cours...')
            ],id = 'loading', className = 'loading'),
            html.Div([],id = 'refreash', className='refreash'),
            html.Div([html.Button("", id='download1',n_clicks=0)],id = 'download', className='donwload'),
            dcc.Input(id="input5", type="hidden", placeholder="", debounce=True, className="input-path-out"),
            Download(id="download-dataframe-csv")
        ],id = 'contact-box-2', className="contact-box-2 hidden")
    ], className = "container")
    ], className ="body")

def parse_contents(contents, filename, path, value):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    try:
        if 'las' in filename or 'LAS' in filename:
            File = laspy.read(io.BytesIO(decoded))
            X=np.array(File.x)
            Y=np.array(File.y)
            Z=np.array(File.z)
            classe=np.array(File.classification)
            mx,my,mz=np.mean(X),np.mean(Y),np.mean(Z)
            X,Y,Z=X-mx,Y-my,Z-mz
            d=pd.DataFrame(np.array([X,Y,Z,classe]).T)
            d = d.set_axis(['cartesianX', 'cartesianY', 'cartesianZ', 'classe'], axis=1, inplace=False)
            if 13 not in classe : 
               i = 1 
               d=pd.DataFrame(np.array([X,Y,Z]).T,columns=['cartesianX','cartesianY','cartesianZ'])
               r = model.extraction(d, filename, value, i)
               del File, X, Y, Z, classe, d
               return r
            else :
               i = 2
               d=pd.DataFrame(np.array([X,Y,Z,classe]).T,columns=['cartesianX','cartesianY','cartesianZ','classe'])
               r = model.extraction(d, filename, value, i)
               del File, X, Y, Z, classe, d
               return r
        if 'e57' in filename:
            output = str(path)
            output_path = output.replace('\\','/') + '/'
            output_path = output_path + filename
            e57 = pye57.E57(output_path)
            File = e57.read_scan_raw(0)
            d=pd.DataFrame(File)
            i = 0
            r = model.extraction(d, filename, value, i)
            del File, d
            return r
    except Exception as e:
        print(e)
        return None

def parse_contents_1(path, filename, value):
    try:
        if ('las' in filename) or ('LAS' in filename):
            # Assume that the user uploaded a CSV file
            File = laspy.read(path)
            X=np.array(File.x)
            Y=np.array(File.y)
            Z=np.array(File.z)
            classe=np.array(File.classification)
            mx,my,mz=np.mean(X),np.mean(Y),np.mean(Z)
            X,Y,Z=X-mx,Y-my,Z-mz
            if 13 not in classe : 
               i = 1 
               d=pd.DataFrame(np.array([X,Y,Z]).T,columns=['cartesianX','cartesianY','cartesianZ'])
               r = model.extraction(d, filename, value, i)
               del File, X, Y, Z, classe, d
               return r
            else :
               i = 2
               d=pd.DataFrame(np.array([X,Y,Z,classe]).T,columns=['cartesianX','cartesianY','cartesianZ','classe'])
               r = model.extraction(d, filename, value, i)
               del File, X, Y, Z, classe, d
               return r
        if 'e57' in filename:
            # Assume that the user uploaded a CSV file
            e57 = pye57.E57(path)
            File = e57.read_scan_raw(0)
            d=pd.DataFrame(File)
            i = 0
            r = model.extraction(d, filename, value, i)
            del File, d
            return r
    except Exception as e:
        print(e)
        return None
    
    

    
@app.callback(Output('e57', 'className'),
              Output('output-tt', 'children'),
              Output('output-tt', 'className'),
              Output('content', 'value'),
              Output('filename', 'value'),
              State('upload-data', 'contents'),
              Input('upload-data', 'filename'),
              State('output-tt', 'className'),
              State('e57', 'className'))

def update_output1(contents, filename, classes, class_name):
    children = []
    if contents is not None :
        if ('las' in filename) or ('LAS' in filename) :
            class_e57 = str(class_name) + hidden
            class_file = 'button-upload-e572'
            children.append(html.P("Le Fichier " + filename + " est Téléchargé"))
            value1 = contents
            value2 = filename
            return class_e57, children, class_file, value1, value2 
        elif ".e57" in filename : 
            children.append(html.P("Le Fichier " + filename + " est Téléchargé"))
            class_file = 'button-upload-e572'
            value1 = contents
            value2 = filename
            return class_name, children, class_file, value1, value2 
        else :
            children.append(html.P("Veuillez Choisir Un Fichier e57 Ou las"))
            class_file = 'button-upload-e572 background-red'
            value1 = None
            value2 = None
            return class_name, children, class_file, value1, value2 
    else : 
        value1 = None
        value2 = None
        return class_name, children, classes, value1, value2 

@app.callback([Output('contact-box', 'className'),
              Output('contact-box-1', 'className'),
              Output('contact-box-2', 'className'),
              Output('error1', 'children'),
              Output('error', 'children')],
              State('content', 'value'),
              State('filename', 'value'),
              State('input3', 'value'),
              State('input2', 'value'),
              State('input4', 'value'),
              State('contact-box', 'className'),
              State('contact-box-1', 'className'),
              State('contact-box-2', 'className'),
              Input('start', 'n_clicks'),
              Input('suivant', 'n_clicks'))

def update_output2(contents, filename, path1, path2, path3, class_1, class_2, class_3, n_clicks, n_clicks1):
    children = []
    children2 = []
    if path3 is None and n_clicks1 > 0 :
        classe1 = str(class_1)[:-8] + hidden
        class2 = str(class_2)[:-8] + visible
        class3 = str(class_3)[:-7] + hidden
        children2 = [dbc.Alert("Veuillez Sélectionner Le Chemin De Sortie Des Fichiers !", color="danger")]
        return classe1, class2, class3, children, children2
    elif path3 is not None and n_clicks1 > 0 :
        output = str(path3)
        output = output.replace('\\','/') 
        if os.path.exists(output) :
            classe1 = str(class_1)[:-8] + hidden
            class2 = str(class_2)[:-8] + hidden
            class3 = str(class_3)[:-7] + visible
            return classe1, class2, class3, children, children2
        else : 
            classe1 = str(class_1)[:-8] + hidden
            class2 = str(class_2)[:-8] + visible
            class3 = str(class_3)[:-7] + hidden
            children2 = [dbc.Alert("Veuillez Entrer Un Chemin Valide", color="danger")]
            return classe1, class2, class3, children, children2
    elif path2 is not None and n_clicks > 0 :
        output = str(path2)
        output = output.replace('\\','/') 
        if os.path.exists(output) :
            classe1 = str(class_1) + ' ' + hidden
            class2 = str(class_2)[:-7] + visible
            return classe1, class2, class_3, children, children2
        else :
            children = [dbc.Alert("Veuillez Entrer Un Chemin Valide", color="danger")]
            return class_1, class_2, class_3, children, children2
    elif contents is not None and n_clicks > 0:
        if "e57" in filename :
            if path1 is not None :
                output = str(path1)
                output = output.replace('\\','/') + '/' + filename
                if os.path.isfile(output) :
                    classe1 = str(class_1)[:-8] + hidden
                    class2 = str(class_2)[:-7] + visible
                    return classe1, class2, class_3, children, children2
                else :
                    children = [dbc.Alert("Veuillez Entrer Un Chemin Valide Du Fichier e57 !", color="danger")]
                    return class_1, class_2, class_3, children, children2 
            else :
                children = [dbc.Alert("Veuillez Sélectionner Le Chemin Du Fichier e57 !", color="danger")]
                return class_1, class_2, class_3, children, children2
        elif (('las' in filename) or ('LAS' in filename)) and n_clicks > 0:
            classe1 = str(class_1)[:-8] + hidden
            class2 = str(class_2)[:-7] + visible
            return classe1, class2, class_3, children, children2
        elif path2 is None and n_clicks > 0:
            children = [dbc.Alert("Veuillez Sélectionner Le Chemin Des Fichiers Ou De Télécharger Le Fichier !", color="danger")]
            return class_1, class_2, class_3, children, children2
    else :
        return str(class_1), str(class_2), str(class_3), children, children2

@app.callback([Output('output', 'children'),
              Output('refreash', 'children'),
              Output('download', 'children'),
              Output('loading', 'className'), 
              Output('termine', 'className'),
              Output('input5', 'value')],
              State('content', 'value'),
              State('filename', 'value'),
              State('input3', 'value'),
              State('input2', 'value'),
              State('input4', 'value'),
              State('loading', 'className'), 
              State('termine', 'className'),
              Input('suivant', 'n_clicks'))


def update_output3(contents, filename, path, path2, path3, class_1, class_2, n_clicks):
    children = []
    children2 = []
    children3 = []
    output1 = str(path)
    output1 = output1.replace('\\','/') 
    output2 = str(path2)
    output2 = output2.replace('\\','/') 
    output3 = str(path3)
    output3 = output3.replace('\\','/') 
    value = ''
    if os.path.exists(output2) and os.path.exists(output3) and n_clicks > 0 :
        folder_path=os.listdir(output2)
        files=[os.path.join(output2,s) for s in folder_path]
        for f in files :
                filename = f.split('/')[-1]
                filename = filename.split('\\')[-1]
                r = parse_contents_1(f, filename, path3)
                if r is not None :
                    value = value + filename + '/'
                    now = datetime.datetime.now()
                    value = value + str(r) + '/' + now.strftime("%m-%d-%Y") + '/' + now.strftime("%H:%M:%S") + '\\'
                    children.append(html.P('Le Fichier ' + filename + ' est traité'))
                    children.append(html.I(className="fa fa-check icon2"))
                else :
                    children.append(html.P('Le Fichier ' + filename + " n'est pas traité", className = 'red'))
                    children.append(html.I(className="fas fa-times icon2 red"))
                    #return children, children2, children3, value
        children2 = [html.A([html.Button("Revenir à L'acceuil", id='finish',className="finish", n_clicks=0)],
             href='/')]
        children3 = [html.Button("Télécharger La Liste Des Fichiers Traités", id='download1',className="finish1", n_clicks=0)]
        class1 = str(class_1)[:-8] + hidden
        class2 = str(class_2)[:-7] + visible
        return children, children2, children3, class1, class2, value
    elif contents is not None  and n_clicks > 0:
        if '.e57' in filename and os.path.exists(output1) and os.path.exists(output2) == False:
            r = parse_contents(contents, filename, path, path3)
            children.append(html.P('Le Fichier ' + filename + ' est traité'))
            children.append(html.I(className="fa fa-check icon2"))
            children2 = [html.A([html.Button("Revenir à L'acceuil", id='finish',className="finish", n_clicks=0)],href='/')]
            class1 = str(class_1)[:-8] + hidden
            class2 = str(class_2)[:-7] + visible
            return children, children2, children3, class1, class2, value
        elif (('las' in filename) or ('LAS' in filename)) and os.path.exists(output2) == False:
            r = parse_contents(contents, filename, path, path3)
            children.append(html.P('Le Fichier ' + filename + ' est traité'))
            children.append(html.I(className="fa fa-check icon2"))
            children2 = [html.A([html.Button("Revenir à L'acceuil", id='finish',className="finish", n_clicks=0)],href='/')]
            class1 = str(class_1)[:-8] + hidden
            class2 = str(class_2)[:-7] + visible
            return children, children2, children3, class1, class2, value
    else :
        return children, children2, children3, class_1, class_2, value


@app.callback(
    Output("download-dataframe-csv", "data"),
    State('input5', 'value'),
    Input("download1", "n_clicks"),
    prevent_initial_call=True,
)
def func(value, n_clicks):
    if n_clicks > 0 :
        x = value.split('\\')
        dd = []
        for i in range(len(x)) :
            if x[i] != '' :
                y = x[i].split('/') 
                dd.append(y)
        df = pd.DataFrame(dd)
        df = df.set_axis(['Nom_Fichier', 'Nombre_Coupe_Detecte', 'Date', 'Time'], axis=1, inplace=False)
        return send_data_frame(df.to_csv, "Liste_Fichiers_Traités.csv")



if __name__ == '__main__':
    app.run_server(debug = True)