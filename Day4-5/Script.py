# import dash
# from dash import dcc, html
# import plotly.express as px
# import pandas as pd
# from sklearn.datasets import load_iris

# # Load Iris dataset
# iris = load_iris()
# iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
# iris_df['species'] = iris.target_names[iris.target]

# # Create a scatter plot using Plotly
# fig = px.scatter(iris_df, x='sepal length (cm)', y='sepal width (cm)', color='species',
#                  title="Iris Dataset: Sepal Length vs Sepal Width")

# # Initialize the Dash app
# app = dash.Dash(__name__)

# # Define the layout of the dashboard
# app.layout = html.Div([
#     html.H1("Iris Dataset Visualization"),
    
#     dcc.Graph(figure=fig),
    
#     html.P("Use the dropdown to select a species:"),
    
#     dcc.Dropdown(
#         id='species-dropdown',
#         options=[{'label': species, 'value': species} for species in iris.target_names],
#         value='setosa',
#         multi=False
#     ),
    
#     html.Div(id='species-info')
# ])

# # Callback to update the plot based on selected species
# @app.callback(
#     dash.dependencies.Output('species-info', 'children'),
#     [dash.dependencies.Input('species-dropdown', 'value')]
# )
# def update_species_info(species):
#     species_data = iris_df[iris_df['species'] == species]
#     return f"Selected species: {species}. The dataset contains {len(species_data)} samples of this species."

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import plotly.express as px

# الخطوة 1: تحميل مجموعة بيانات الأيريس
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# الخطوة 2: تطبيق PCA لتقليص الأبعاد
pca = PCA(n_components=2)  # تقليص الأبعاد إلى مكونين لتصوير البيانات في بعدين
pca_result = pca.fit_transform(iris_df[iris.feature_names])

# الخطوة 3: إضافة نتائج PCA إلى مجموعة البيانات
iris_df['PCA1'] = pca_result[:, 0]
iris_df['PCA2'] = pca_result[:, 1]

# الخطوة 4: تصوير نتائج PCA
fig = px.scatter(iris_df, x='PCA1', y='PCA2', color='species',
                 title="تحليل المكونات الرئيسية (PCA) لمجموعة بيانات الأيريس",
                 labels={'PCA1': 'المكون الرئيسي 1', 'PCA2': 'المكون الرئيسي 2'})
fig.show()
