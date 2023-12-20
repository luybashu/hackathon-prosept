import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_texts(name):
    """
    Text cleaning function.
    Takes a string as input - the product name,
    returns it in edited form
    In the case of Nan returns empty string.
    """     
    if not pd.isna(name):
        # Separating concatenated words and digits with spaces
        name = ' '.join(re.split(r"([0-9][0-9]*)", name))
        # Lowercasing
        name = name.lower()
        # Removing punctuation
        name = re.sub(r"[^а-яa-z\d\s]+", ' ', name)        
        # Removing words "prosept", "просепт", "professional"
        name = re.sub(r"prosept|просепт|professional", ' ', name)
    else:
        name = ''    
    return name

def prosept_predict(product:list, dealerprice:list, N_BEST=10) -> list: 
    """
    Function for predicting the n nearest (N_BEST) customer 
    product names for each dealer's product.
    Parameters:
    - product - a list of dictionaries, 
    database of products produced by customer;
    - dealerprice - a list of dictionaries, 
    the result of the parser's work on dealer platforms;
    - N_BEST=10 - number of nearest customer product names.
    Returns list of dictionaries, n nearest customer 
    product ids for each dealer's product runked by similarity value,
    similarity value and date. 
    """
    
    # Converting dictionaries to DataFrame
    df_product = pd.DataFrame.from_dict(product)
    df_dealerprice = pd.DataFrame.from_dict(dealerprice)

    # Creating DataFrame that will contain recommendations
    df_res = df_dealerprice[['id', 'product_key']]
    
    # Columns with product names
    columns = ['name', 'ozon_name', 'name_1c', 'wb_name']
    
    # Instantiate the TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2), max_df=0.6)

    def t_fit_tfidf(df,func=clean_texts,df_columns=['name']):
        """
        Transform the text data in the specified DataFrame using TF-IDF vectorization.
        Learn vocabulary and idf from customer product names.
        Parameters:
        - df: the input DataFrame containing text data;
        - func: function for text preprocessing;
        - df_columns: the list of column names in the DataFrame containing text data.
        Returns:
        - model: sparse matrix, the TF-IDF transformed features matrix;
        - df_subset: a subset of the original DataFrame containing the 'id' and 'name' columns.
        """
        df_tmp = df[df_columns[0]].apply(func)
        if len(df_columns)>1:
            for i in range(1,len(df_columns)):
                df_tmp = df_tmp + ' ' + df[df_columns[i]].apply(func)

        model = vectorizer.fit_transform(df_tmp)

        return model, df[['id','name']]

    # Apply function to get vectors from customer product names
    product_vec, df_product_tfidf =  t_fit_tfidf(df_product,clean_texts,columns)

    def t_predict_tfidf(dealer_names, product_vec=product_vec):
        """
        Vectorize dealer product names.
        Predict the similarity between dealer and 
        customer product names using their TF-IDF representation.
        Parameters:
        - dealer_names: a Pandas Series containing dealer 
        names to be compared.
        - product_vec: sparse matrix, the TF-IDF vector 
        representation of customer product names. 
        Returns:
        - similarity_scores: 2D array, a matrix of cosine similarity 
        scores between each dealer name and all product vectors.
        Each row corresponds to a dealer product name, and each column 
        corresponds to a customer product name.
        """
        dealer_vec = vectorizer.transform(dealer_names.apply(clean_texts))
        return cosine_similarity(dealer_vec, product_vec) 

    # Apply function to get a matrix of cosine similarity
    df_predict_tfidf = t_predict_tfidf(df_dealerprice['product_name'], product_vec)
    
    # top 10 nearest customer product name indicies and their similarity values
    N_BEST = N_BEST
    indices =  df_predict_tfidf.argsort()[:, -N_BEST:][:, ::-1]
    quality = np.take_along_axis(df_predict_tfidf, indices, axis=1)
    
    # Saving results
    df_res.loc[:,'predict'] = indices.tolist()
    df_res.loc[:,'quality'] = quality.tolist()
    df_res['queue'] = [list(range(1, N_BEST+1)) for _ in range(len(df_res))]
    df_res=df_res.explode(['predict', 'quality', 'queue'])
    df_res = df_res.reset_index(drop=True)
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict',axis=1)
    df_res['create_date'] = datetime.now()

    return df_res.to_dict()