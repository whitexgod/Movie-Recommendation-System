import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
from fuzzywuzzy import process
from sklearn.neighbors import NearestNeighbors
import requests
import sklearn

movie_features_df=pickle.load(open('movie_features_df_remake.pkl','rb'))
movie_names=pickle.load(open('movie_names_remake.pkl','rb'))


app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/redirect', methods=['POST'])
def redirect():
    try:
        movie_title=request.form.get('movie-title').upper()

        def get_matches(query, choices, limit=5):
            results = process.extract(query, choices, limit=limit)
            return results

        movie_list = get_matches(movie_title, movie_names)
        new_list = []
        for i in range(5):
            if movie_list[i][1] > 70:
                new_list.append(movie_list[i])
        movie_nam = []
        for i in range(len(new_list)):
            movie_nam.append(str(new_list[i][0]))
        d = {'movie': movie_nam}
        data = pd.DataFrame(d)
        data.drop_duplicates(inplace=True)
        data_len=data.shape[0]

        return render_template('redirect.html', movie_title=movie_title, data=data, data_len=data_len)
    except:
        return render_template('index.html', label=1)


@app.route('/predict',methods=['POST'])
def predict():
    movie_title = request.form.get('movie-title').upper()
    try:
        from scipy.sparse import csr_matrix
        movie_features_df_matrix = csr_matrix(movie_features_df.values)

        from sklearn.neighbors import NearestNeighbors
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(movie_features_df_matrix)

        rec_mov = []
        query_index = np.where(movie_features_df.index == movie_title)[0][0]
        distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1),n_neighbors=10)
        for i in range(10):
            rec_mov.append(movie_features_df.index[indices.flatten()[i]])
        d = {'recommended_movies': rec_mov}
        df = pd.DataFrame(d)

        return render_template('predict.html', movie_title=movie_title, df=df)
    except:
        return render_template('index.html', label=1)


if __name__=="__main__":
    app.run(debug=True)