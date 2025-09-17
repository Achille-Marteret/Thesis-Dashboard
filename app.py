import openpyxl
import dash
from dash import html, dcc, Input, Output, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ======================================= Initialisation de l'application =================================== #
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Immigration & Criminalité"

# ======================================= Chargement et traitement des données ============================== #
df = pd.read_excel("IRIS_FRANCE_METRO.xlsx")

# Normalisation des codes départements (y compris 2A, 2B)
df['CODE_DEPT'] = df['CODDEP'].apply(lambda x: str(x).zfill(2) if str(x).isdigit() else str(x))

# Vérification de la colonne TAUX_IMMIGRATION
if 'TAUX_IMMIGRATION' not in df.columns:
    raise ValueError("La colonne 'TAUX_IMMIGRATION' est absente du fichier Excel.")

# === Chargement du GeoJSON des départements ===
with open("departements.geojson", encoding='utf-8') as f:
    geojson_depts = json.load(f)

# === Détection automatique de la clé dans le GeoJSON ===
print("Exemple de propriétés dans le GeoJSON :", geojson_depts['features'][0]['properties'])

# Modifie cette ligne si besoin après avoir vu la clé exacte
geojson_key = "code"  # ou "code_insee", selon le contenu réel

# === Liste des indicateurs de criminalité avec couleurs ===
indicateurs_crime = {
    "taux_crime_cheques": {"label": "Chèques", "color": "#00008B"},
    "taux_crime_plaintes": {"label": "Plaintes", "color": "#FF0000"},
    "taux_crime_vols_vehicules": {"label": "Vols de véhicules", "color": "#99ccff"},
    "taux_crime_stupefiants": {"label": "Stupéfiants et auteurs", "color": "#6699cc"},
    "taux_crime_vols_victime": {"label": "Vols victimes", "color": "#8B0000"},
    "taux_crime_violences": {"label": "Violences", "color": "#e37c7c"},
    "taux_crime_autres": {"label": "Autres crimes", "color": "#FFA500"},
    "taux_crime_infractions_pub": {"label": "Infractions publiques ou privées", "color": "#FFFF00"}
}

# === Dictionnaire des noms de départements ===
dept_names = {
    '01': 'Ain', '02': 'Aisne', '03': 'Allier', '04': 'Alpes-de-Haute-Provence',
    '05': 'Hautes-Alpes', '06': 'Alpes-Maritimes', '07': 'Ardèche', '08': 'Ardennes',
    '09': 'Ariège', '10': 'Aube', '11': 'Aude', '12': 'Aveyron', '13': 'Bouches-du-Rhône',
    '14': 'Calvados', '15': 'Cantal', '16': 'Charente', '17': 'Charente-Maritime',
    '18': 'Cher', '19': 'Corrèze', '2A': 'Corse-du-Sud', '2B': 'Haute-Corse',
    '21': 'Côte-d\'Or', '22': 'Côtes-d\'Armor', '23': 'Creuse', '24': 'Dordogne',
    '25': 'Doubs', '26': 'Drôme', '27': 'Eure', '28': 'Eure-et-Loir', '29': 'Finistère',
    '30': 'Gard', '31': 'Haute-Garonne', '32': 'Gers', '33': 'Gironde', '34': 'Hérault',
    '35': 'Ille-et-Vilaine', '36': 'Indre', '37': 'Indre-et-Loire', '38': 'Isère',
    '39': 'Jura', '40': 'Landes', '41': 'Loir-et-Cher', '42': 'Loire', '43': 'Haute-Loire',
    '44': 'Loire-Atlantique', '45': 'Loiret', '46': 'Lot', '47': 'Lot-et-Garonne',
    '48': 'Lozère', '49': 'Maine-et-Loire', '50': 'Manche', '51': 'Marne', '52': 'Haute-Marne',
    '53': 'Mayenne', '54': 'Meurthe-et-Moselle', '55': 'Meuse', '56': 'Morbihan',
    '57': 'Moselle', '58': 'Nièvre', '59': 'Nord', '60': 'Oise', '61': 'Orne',
    '62': 'Pas-de-Calais', '63': 'Puy-de-Dôme', '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées', '66': 'Pyrénées-Orientales', '67': 'Bas-Rhin', '68': 'Haut-Rhin',
    '69': 'Rhône', '70': 'Haute-Saône', '71': 'Saône-et-Loire', '72': 'Sarthe',
    '73': 'Savoie', '74': 'Haute-Savoie', '75': 'Paris', '76': 'Seine-Maritime',
    '77': 'Seine-et-Marne', '78': 'Yvelines', '79': 'Deux-Sèvres', '80': 'Somme',
    '81': 'Tarn', '82': 'Tarn-et-Garonne', '83': 'Var', '84': 'Vaucluse',
    '85': 'Vendée', '86': 'Vienne', '87': 'Haute-Vienne', '88': 'Vosges',
    '89': 'Yonne', '90': 'Territoire de Belfort', '91': 'Essonne', '92': 'Hauts-de-Seine',
    '93': 'Seine-Saint-Denis', '94': 'Val-de-Marne', '95': 'Val-d\'Oise'
}

df['NOM_DEPT'] = df['CODE_DEPT'].map(dept_names)

# Préparation des données pour l'évolution 
evolution_data = {}
for indicator in indicateurs_crime.keys():
    df_evol = df[['ANNEES', 'CODE_DEPT', indicator]].dropna()
    df_mean = df_evol.groupby('ANNEES')[indicator].mean().reset_index()
    evolution_data[indicator] = df_mean

# ======================================= Fonctions pour l'analyse de régression ============================== #

def calculate_within_transformation(df, y_col, x_col, dept_col='CODE_DEPT', year_col='ANNEES'):
    """
    Calcule la transformation within pour les effets fixes département-année
    """
    df_copy = df.copy()
    
    # Moyennes par département et année
    dept_year_means = df_copy.groupby([dept_col, year_col])[[y_col, x_col]].mean()
    
    # Moyennes par département
    dept_means = df_copy.groupby(dept_col)[[y_col, x_col]].mean()
    
    # Moyennes par année
    year_means = df_copy.groupby(year_col)[[y_col, x_col]].mean()
    
    # Moyennes globales
    global_means = df_copy[[y_col, x_col]].mean()
    
    # Merge des moyennes
    df_copy = df_copy.merge(dept_means, on=dept_col, suffixes=('', '_dept_mean'))
    df_copy = df_copy.merge(year_means, on=year_col, suffixes=('', '_year_mean'))
    
    # Transformation within
    df_copy[f'{y_col}_within'] = (df_copy[y_col] - df_copy[f'{y_col}_dept_mean'] - 
                                 df_copy[f'{y_col}_year_mean'] + global_means[y_col])
    df_copy[f'{x_col}_within'] = (df_copy[x_col] - df_copy[f'{x_col}_dept_mean'] - 
                                 df_copy[f'{x_col}_year_mean'] + global_means[x_col])
    
    return df_copy

def create_regression_comparison_plot(df, crime_indicator):
    """
    Crée un graphique comparant régression simple et effets fixes
    """
    # Filtrer les données non nulles
    df_clean = df[[crime_indicator, 'TAUX_IMMIGRATION', 'CODE_DEPT', 'ANNEES']].dropna()
    
    if len(df_clean) < 10:  # Vérification minimum de données
        return go.Figure().add_annotation(
            text="Données insuffisantes pour l'analyse de régression",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    # Régression linéaire simple
    X_simple = df_clean['TAUX_IMMIGRATION'].values.reshape(-1, 1)
    y_simple = df_clean[crime_indicator].values
    
    model_simple = LinearRegression()
    model_simple.fit(X_simple, y_simple)
    y_pred_simple = model_simple.predict(X_simple)
    
    # Calcul des données within pour effets fixes
    df_within = calculate_within_transformation(df_clean, crime_indicator, 'TAUX_IMMIGRATION')
    
    # Régression avec effets fixes (sur données transformées)
    X_within = df_within['TAUX_IMMIGRATION_within'].values.reshape(-1, 1)
    y_within = df_within[f'{crime_indicator}_within'].values
    
    model_within = LinearRegression()
    model_within.fit(X_within, y_within)
    y_pred_within = model_within.predict(X_within)
    
    # Création du graphique avec sous-graphiques
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Régression Linéaire Simple', 'Régression à Effets Fixes (Département-Année)'),
        horizontal_spacing=0.1
    )
    
    # Graphique 1: Régression simple
    fig.add_trace(
        go.Scatter(
            x=df_clean['TAUX_IMMIGRATION'],
            y=df_clean[crime_indicator],
            mode='markers',
            name='Observations',
            marker=dict(color='darkgrey', size=4, opacity=0.6),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Ligne de régression simple
    x_range_simple = np.linspace(df_clean['TAUX_IMMIGRATION'].min(), 
                                df_clean['TAUX_IMMIGRATION'].max(), 100)
    y_line_simple = model_simple.predict(x_range_simple.reshape(-1, 1))
    
    fig.add_trace(
        go.Scatter(
            x=x_range_simple,
            y=y_line_simple,
            mode='lines',
            name=f'Régression simple (β={model_simple.coef_[0]:.4f})',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )
    
    # Graphique 2: Effets fixes
    fig.add_trace(
        go.Scatter(
            x=df_within['TAUX_IMMIGRATION_within'],
            y=df_within[f'{crime_indicator}_within'],
            mode='markers',
            name='Observations (centrées)',
            marker=dict(color='#377eb8', size=4, opacity=0.6),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Ligne de régression effets fixes
    x_range_within = np.linspace(df_within['TAUX_IMMIGRATION_within'].min(), 
                                df_within['TAUX_IMMIGRATION_within'].max(), 100)
    y_line_within = model_within.predict(x_range_within.reshape(-1, 1))
    
    fig.add_trace(
        go.Scatter(
            x=x_range_within,
            y=y_line_within,
            mode='lines',
            name=f'Effets fixes (β={model_within.coef_[0]:.4f})',
            line=dict(color='#e41a1c', width=2)
        ),
        row=1, col=2
    )
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="Taux d'Immigration (%)", row=1, col=1)
    fig.update_xaxes(title_text="Taux d'Immigration (centré)", row=1, col=2)
    fig.update_yaxes(title_text=indicateurs_crime[crime_indicator]['label'], row=1, col=1)
    fig.update_yaxes(title_text=f"{indicateurs_crime[crime_indicator]['label']} (centré)", row=1, col=2)
    
    # Mise à jour du layout
    fig.update_layout(
        title=f"Comparaison des méthodes de régression - {indicateurs_crime[crime_indicator]['label']}",
        title_x=0.5,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# ======================================= Layout de l'application =========================================== #
app.layout = dbc.Container([
    html.H1("Criminalité et Immigration en France", 
            style={'textAlign': 'center', 'marginBottom': '30px', 'marginTop': '20px'}),

    html.Hr(),

    html.P(
        "Ce tableau de bord interactif propose une analyse de l'impact de l'immigration sur différents indicateurs "
        "de criminalité dans les départements français, de 2006 à 2021. Il se compose de trois onglets :",
        style={'textAlign': 'justify'}
    ),

    html.Ul([
        html.Li("Une carte des indicateurs de criminalité exprimés en taux pour 100 000 habitants."),
        html.Li("Une carte des taux d'immigration par département."),
        html.Li("Une analyse comparative, incluant deux types de régressions : une régression linéaire simple et une régression à effets fixes (département et année)."),
    ], style={'marginBottom': '20px'}),

    html.H5("Définitions", style={'marginTop': '30px'}),
    
    html.Ul([
        html.Li([
            html.B("Taux d'immigration : "),
            "rapport entre le nombre d'immigrés et la population totale du département. "
            "Il est calculé comme suit : ",
            html.I("taux_immigration = (nombre d'immigrés / population totale)")
        ]),
        html.Li([
            html.B("Taux de criminalité pour 100 000 habitants : "),
            "indicateur permettant de comparer les niveaux de criminalité entre départements, "
            "en neutralisant les différences de population. Calcul : ",
            html.I("taux = (nombre d’infractions / population totale) × 100 000.")
        ])
    ]),

    html.H5("Définition de l'immigré selon l'INSEE", style={'marginTop': '30px'}),

    html.P(
        "Selon l'INSEE : « La définition adoptée par le Haut Conseil à l’Intégration définit un immigré comme une personne "
        "née étrangère à l’étranger et résidant en France. Les personnes nées Françaises à l’étranger et vivant en France ne sont donc pas comptabilisées. "
        "Certains immigrés ont pu devenir Français, les autres restant étrangers. "
        "Les populations étrangère et immigrée ne se recoupent que partiellement : un immigré n’est pas nécessairement étranger et réciproquement, "
        "certains étrangers sont nés en France (essentiellement des mineurs). La qualité d’immigré est permanente : un individu continue à appartenir à la population immigrée "
        "même s’il devient Français par acquisition. C’est le pays de naissance, et non la nationalité à la naissance, qui définit l'origine géographique d’un immigré. »",
        style={'textAlign': 'justify'}
    ),

    html.Hr(),

    html.P([
        "Pour plus d’informations sur les données, consultez le site de ",
        html.A("l’INSEE", href="https://www.insee.fr/fr/accueil", target="_blank"),
        " ou celui du ",
        html.A("Ministère de l'Intérieur", href="https://www.interieur.gouv.fr/", target="_blank"),
        "."
    ], style={'textAlign': 'center', 'marginTop': '20px', 'fontStyle': 'italic'}),

# ================== Carte criminalité ================ #
    dcc.Tabs([
        dcc.Tab(label='Carte de la Criminalité', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Indicateur de criminalité:", style={'marginBottom': '5px'}),
                        dcc.Dropdown(id='crime-indicator',
                                     options=[
                                         {'label': 'Falsifications et usages frauduleux de chèques', 'value': 'taux_crime_cheques'},
                                         {'label': 'Infractions signalées par les plaignants', 'value': 'taux_crime_plaintes'},
                                         {'label': 'Vols ou dégradations sur des véhicules', 'value': 'taux_crime_vols_vehicules'},
                                         {'label': 'Infractions liées aux stupéfiants et aux des auteurs', 'value': 'taux_crime_stupefiants'},
                                         {'label': 'Vols sur des victimes', 'value': 'taux_crime_vols_victime'},
                                         {'label': 'Violences physiques et psychologiques graves', 'value': 'taux_crime_violences'},
                                         {'label': "Infractions diverses enregistrées comme 'procédures'", 'value': 'taux_crime_autres'},
                                         {'label': 'Infractions contre des établissements publics ou privés', 'value': 'taux_crime_infractions_pub'}
                                         ],
                                         value="taux_crime_violences",
                                         clearable=False,
                                         style={'marginBottom': '10px'}
                                         )
                    ], width=6),
                    dbc.Col([
                        html.Label("Année:", style={'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='year-selector-crime',
                            options=[{'label': str(year), 'value': year} for year in sorted(df['ANNEES'].unique())],
                            value=sorted(df['ANNEES'].unique())[-1],
                            clearable=False,
                            style={'marginBottom': '10px'}
                        )
                    ], width=6)
                ], style={"marginBottom": "20px"}),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="crime-map", style={'height': '500px'})
                    ], width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("", 
                               style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px'}),
                        dcc.Graph(id="crime-evolution-combined", style={'height': '400px'})
                    ], width=12)
                ])
            ])
        ]),

# ================== Carte immigration ================ #

    dcc.Tab(label="Carte de l'Immigration", children=[
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label(style={'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='year-selector-immig',
                    options=[{'label': 'Sélectionnez une année', 'value': 'tout'}] + 
                    [{'label': str(year), 'value': year} for year in sorted(df['ANNEES'].unique())],
                    value='tout',
                    clearable=False,
                    style={'marginBottom': '20px'}
                )
            ], width=6)
        ], justify="center"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id="immigration-map", style={'height': '500px'})
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col([
                html.H4("", style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px'}),
                dcc.Graph(id="immigration-evolution", style={'height': '400px'})
            ], width=12)
        ])
    ])
]),

# ================== Analyse comparative ================ #

        dcc.Tab(label="Analyse Comparative", children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Type d'analyse:", style={'textAlign': 'center', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='analysis-type',
                            options=[
                                {'label': 'Évolution temporelle par département', 'value': 'temporal'},
                                {'label': 'Comparaison des méthodes de régression', 'value': 'regression'}
                            ],
                            value='temporal',
                            clearable=False,
                            style={'marginBottom': '10px'}
                        )
                    ], width=6, style={'margin': '0 auto'}),
                    ], justify="center"),

                dbc.Row([
                    dbc.Col([
                        html.Div(id='dept-selector-container', children=[
                            html.Label("Département:", style={'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='dept-selector',
                                options=[{'label': f"{code} - {name}", 'value': code}
                                        for code, name in dept_names.items()],
                                value='75',
                                clearable=False,
                                style={'marginBottom': '10px'}
                            )
                        ]), 
                    ], width=6)
                ]),
                
                # Dropdown pour sélectionner l'indicateur de criminalité (pour la régression)
                dbc.Row([
                    dbc.Col([
                        html.Div(id='regression-controls', children=[
                            html.Label("Indicateur de criminalité:", style={'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='regression-crime-indicator',
                                options=[
                                    {'label': 'Falsifications et usages frauduleux de chèques', 'value': 'taux_crime_cheques'},
                                    {'label': 'Infractions signalées par les plaignants', 'value': 'taux_crime_plaintes'},
                                    {'label': 'Vols ou dégradations sur des véhicules', 'value': 'taux_crime_vols_vehicules'},
                                    {'label': 'Infractions liées aux stupéfiants et aux des auteurs', 'value': 'taux_crime_stupefiants'},
                                    {'label': 'Vols sur des victimes', 'value': 'taux_crime_vols_victime'},
                                    {'label': 'Violences physiques et psychologiques graves', 'value': 'taux_crime_violences'},
                                    {'label': "Infractions diverses enregistrées comme 'procédures'", 'value': 'taux_crime_autres'},
                                    {'label': 'Infractions contre des établissements publics ou privés', 'value': 'taux_crime_infractions_pub'}
                                ],
                                value="taux_crime_plaintes",
                                clearable=False,
                                style={'marginBottom': '20px'}
                            )
                        ], style={'display': 'none'})  # Caché par défaut
                    ], width=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="comparative-analysis", style={'height': '600px'})
                    ], width=12)
                ])
            ])
        ])
    ])
], fluid=True)

# ======================================== Callbacks ======================================================== #

# ================== Contrôles d'affichage ================ #
@app.callback(
    [Output('regression-controls', 'style'),
     Output('dept-selector-container', 'style')],
    Input('analysis-type', 'value')
)
def toggle_controls(analysis_type):
    if analysis_type == 'regression':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# ================== Carte criminalité ================ #

@app.callback(
    Output('crime-map', 'figure'),
    [Input('crime-indicator', 'value'),
     Input('year-selector-crime', 'value')]
)
def update_crime_map(indicator, year):
    # Chargement optimisé : filtrer seulement les données nécessaires
    df_filtered = df[df['ANNEES'] == year][['CODE_DEPT', 'NOM_DEPT', indicator]].dropna()
    
    # Création de la carte avec hover amélioré
    fig = px.choropleth(
        df_filtered,
        geojson=geojson_depts,
        locations='CODE_DEPT',
        featureidkey=f"properties.{geojson_key}",
        color=indicator,
        color_continuous_scale=["snow", "darkred"],
        hover_name='NOM_DEPT',
        hover_data={indicator: ':.2f', 'CODE_DEPT': False},
        labels={indicator: indicateurs_crime[indicator]['label']},
        title=f"{indicateurs_crime[indicator]['label']} - {year}"
    )
    
    fig.update_geos(
        center={"lat": 46.5, "lon": 2.5},
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showsubunits=False
    )
    
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        dragmode=False,
        title_x=0.5
    )
    return fig


# ================== Carte immigration ================ #

@app.callback(
    Output('immigration-map', 'figure'),
    Input('year-selector-immig', 'value')
)
def update_immigration_map(selected_year):
    if selected_year == 'tout':
        df_filtered = df[['CODE_DEPT', 'NOM_DEPT', 'TAUX_IMMIGRATION']].groupby(['CODE_DEPT', 'NOM_DEPT']).mean().reset_index()
        title = "Taux moyen d'immigration (toutes années)"
    else:
        df_filtered = df[df['ANNEES'] == selected_year][['CODE_DEPT', 'NOM_DEPT', 'TAUX_IMMIGRATION']].dropna()
        title = f"Taux d'immigration - {selected_year}"

    fig = px.choropleth(
        df_filtered,
        geojson=geojson_depts,
        locations='CODE_DEPT',
        featureidkey=f"properties.{geojson_key}",
        color='TAUX_IMMIGRATION',
        color_continuous_scale=["snow", "darkblue"],
        hover_name='NOM_DEPT',
        hover_data={'TAUX_IMMIGRATION': ':.4f', 'CODE_DEPT': True},
        labels={'TAUX_IMMIGRATION': 'Taux Immigration'},
        title=title
    )

    fig.update_geos(
        center={"lat": 46.5, "lon": 2.5},
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        dragmode=False,
        title_x=0.5
    )

    return fig


# ================== Evolution de la criminalité ================ #

@app.callback(
    Output('crime-evolution-combined', 'figure'),
    Input('crime-indicator', 'value')
)
def update_crime_evolution_combined(selected_indicator):
    """Affiche toutes les courbes d'évolution sur le même graphique"""
    
    fig = go.Figure()
    
    for indicator, info in indicateurs_crime.items():
        data = evolution_data[indicator]
        
        is_selected = indicator == selected_indicator
        
        fig.add_trace(
            go.Scatter(
                x=data['ANNEES'],
                y=data[indicator],
                mode='lines+markers',
                name=info['label'],
                line=dict(
                    color=info['color'],
                    width=4 if is_selected else 1.5,
                    dash='solid' if is_selected else 'dot'  # ligne en pointillé pour les non-sélectionnées
                ),
                marker=dict(
                    size=8 if is_selected else 4,
                    symbol='circle'
                ),
                opacity=1.0 if is_selected else 0.8,
                hovertemplate=f"<b>{info['label']}</b><br>" +
                              "Année: %{x}<br>" +
                              "Taux: %{y:.2f}<br>" +
                              "<extra></extra>"
            )
        )
    
    fig.update_layout(
        title="Évolution comparative de tous les indicateurs de criminalité",
        title_x=0.5,
        yaxis_title="Taux moyen",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=40, l=60, r=40)
    )
    
    return fig


# ================== Evolution de l'immigration ================ #

@app.callback(
    Output('immigration-evolution', 'figure'),
    Input('year-selector-immig', 'value')  # Trigger
)
def update_immigration_evolution(year_selected):
    # Utiliser les données pré-calculées
    df_evol = df[['ANNEES', 'TAUX_IMMIGRATION']].dropna()
    df_mean = df_evol.groupby('ANNEES')['TAUX_IMMIGRATION'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df_mean['ANNEES'],
            y=df_mean['TAUX_IMMIGRATION'],
            mode='lines+markers',
            name="Taux d'immigration",
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=6),
            hovertemplate="<b>Taux d'immigration</b><br>" +
                         "Année: %{x}<br>" +
                         "Taux: %{y:.2f}%<br>" +
                         "<extra></extra>"
        )
    )
    
    # Marquer l'année sélectionnée
    if year_selected in df_mean['ANNEES'].values:
        year_value = df_mean[df_mean['ANNEES'] == year_selected]['TAUX_IMMIGRATION'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[year_selected],
                y=[year_value],
                mode='markers',
                name=f"Année sélectionnée ({year_selected})",
                marker=dict(size=12, color='red', symbol='diamond')
            )
        )
    
    fig.update_layout(
        title="Évolution du taux d'immigration moyen en France",
        title_x=0.5,
        xaxis_title="Année",
        yaxis_title="Taux d'immigration (%)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=40, l=60, r=40)
    )
    
    return fig

# ================== Analyse comparative ================ #

# ================== Callback principal pour l'analyse comparative ================ #

@app.callback(
    Output('comparative-analysis', 'figure'),
    [Input('analysis-type', 'value'),
     Input('dept-selector', 'value'),
     Input('regression-crime-indicator', 'value')]
)
def update_comparative_analysis(analysis_type, selected_dept, crime_indicator):
    """Callback principal pour l'analyse comparative"""
    
    if analysis_type == 'temporal':
        return create_temporal_analysis(selected_dept)
    elif analysis_type == 'regression':
        return create_regression_comparison_plot(df, crime_indicator)
    else:
        return go.Figure()

# ================== Fonction pour l'analyse temporelle ================ #

def create_temporal_analysis(selected_dept):
    """Analyse temporelle pour un département sélectionné"""
    
    if not selected_dept:
        return go.Figure().add_annotation(
            text="Veuillez sélectionner un département",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    # Filtrer les données pour le département sélectionné
    dept_data = df[df['CODE_DEPT'] == selected_dept].copy()
    
    if dept_data.empty:
        return go.Figure().add_annotation(
            text=f"Aucune donnée disponible pour le département {selected_dept}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font_size=16
        )
    
    # Créer le graphique avec axe Y secondaire
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ajouter l'évolution de l'immigration (axe Y principal)
    if 'TAUX_IMMIGRATION' in dept_data.columns:
        immigration_data = dept_data[['ANNEES', 'TAUX_IMMIGRATION']].dropna()
        fig.add_trace(
            go.Scatter(
                x=immigration_data['ANNEES'],
                y=immigration_data['TAUX_IMMIGRATION'],
                mode='lines+markers',
                name="Taux d'immigration",
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6),
                hovertemplate="<b>Taux d'immigration</b><br>" +
                             "Année: %{x}<br>" +
                             "Taux: %{y:.2f}%<br>" +
                             "<extra></extra>"
            ),
            secondary_y=False
        )
    
    # Ajouter plusieurs indicateurs de criminalité sur l'axe Y secondaire
    colors = ["#00008B", '#FF0000', "#99ccff", "#6699cc",
               "#8B0000", "#e37c7c", "#FFA500", "#FFFF00"]
    selected_crimes = ['taux_crime_cheques', 'taux_crime_plaintes', 'taux_crime_vols_vehicules', 'taux_crime_stupefiants',
                        'taux_crime_vols_victime', 'taux_crime_violences', 'taux_crime_autres', 'taux_crime_infractions_pub']
    
    for i, crime_indicator in enumerate(selected_crimes):
        if crime_indicator in dept_data.columns:
            crime_data = dept_data[['ANNEES', crime_indicator]].dropna()
            if not crime_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=crime_data['ANNEES'],
                        y=crime_data[crime_indicator],
                        mode='lines+markers',
                        name=indicateurs_crime[crime_indicator]['label'],
                        line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'),
                        marker=dict(size=4),
                        hovertemplate=f"<b>{indicateurs_crime[crime_indicator]['label']}</b><br>" +
                                     "Année: %{x}<br>" +
                                     "Taux: %{y:.2f}<br>" +
                                     "<extra></extra>"
                    ),
                    secondary_y=True
                )
    
    dept_name = dept_names.get(selected_dept, selected_dept)
    
    # Mise à jour des labels des axes
    fig.update_xaxes(title_text="Année")
    fig.update_yaxes(title_text="Taux d'immigration (%)", secondary_y=False)
    fig.update_yaxes(title_text="Taux de criminalité", secondary_y=True)
    
    fig.update_layout(
        title=f"Évolution comparative - {dept_name} ({selected_dept})",
        title_x=0.5,
        hovermode='x unified',
        margin=dict(t=80, b=40, l=60, r=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# ======================================== Exécution ======================================================== #
if __name__ == '__main__':
    app.run_server(debug=True, port=8053)