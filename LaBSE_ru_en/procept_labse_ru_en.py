import re

import pandas as pd
import torch

from datetime import datetime
from nltk.corpus import stopwords as nltk_stopwords
nltk.download('stopwords')
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel

# Stop words for Russian and English
stop_words_en = set(nltk_stopwords.words('english'))
stop_words_ru = set(nltk_stopwords.words('russian'))
stop_words = stop_words_en.union(stop_words_ru)

# Loading the tokenizer and LaBSE_ru_en model (516 MB)
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

def clean_texts(name):
    """
    Text cleaning function.
    Takes a string as input - the product name,
    returns it in edited form
    In the case of Nan returns empty string.
    """     
    if not pd.isna(name):
        name = ' '.join(re.split(r"([A-Za-z][A-Za-z]*)", name))
        name = ' '.join(re.split(r"([0-9][0-9]*)", name))
        name = name.lower()
        name = re.sub(r"[^а-яa-z\d\s]+", ' ', name)
        name = re.sub(r"prosept", ' ', name)
        name = ' '.join([word for word in name.split() if word not in stop_words])
    else:
        name = ''
    return name

def prosept_predict(product: list, dealerprice: list, N_BEST=10) -> list:
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
    
    # DataFrame with unique dealer product names
    df_dealerprice_unique = (df_dealerprice[['product_name']]
                             .drop_duplicates(subset='product_name')
                             .reset_index(drop=True))

    # Creating DataFrame that will contain recommendations
    df_res = df_dealerprice[['id', 'product_key', 'product_name']]
    
    # Columns with product names
    columns = ['name', 'ozon_name', 'name_1c', 'wb_name']

    def t_fit_LaBSE(df,func=clean_texts,df_columns=['name']):
        """
        Transform the text data in the specified DataFrame using 
        pretrained LaBSE_ru_en model.
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

        list_tmp = df_tmp.tolist()
        
        encoded_input = tokenizer(list_tmp
                                  , padding=True
                                  , truncation=True
                                  , max_length=61
                                  , return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)            

        return embeddings, df[['id','name']]

    # Apply function to get vectors from customer product names
    product_embedding_LaBSE, df_product_LaBSE =  t_fit_LaBSE(df_product,clean_texts,columns)
    
    def t_predict_LaBSE(dealer_names, product_vec=product_embedding_LaBSE):
        """
        Vectorize dealer product names.
        Predict the similarity between dealer and 
        customer product names.
        Parameters:
        - dealer_names: a Pandas Series containing dealer 
        names to be compared.
        - product_vec: embeddings of customer product names. 
        Returns:
        - similarity_scores: 2D array, a matrix of cosine similarity 
        scores between each dealer name and all product vectors.
        Each row corresponds to a dealer product name, and each column 
        corresponds to a customer product name.
        """
        encoded_input = tokenizer(dealer_names.apply(clean_texts)
                                  , padding=True
                                  , truncation=True
                                  , max_length=61
                                  , return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.pooler_output
        txt_embedding = torch.nn.functional.normalize(embeddings)             

        return util.pytorch_cos_sim(txt_embedding, product_vec)
       
    # Apply function to get a matrix of cosine similarity
    df_predict_LaBSE = t_predict_LaBSE(df_dealerprice_unique['product_name'], product_embedding_LaBSE)
    
    # top 10 nearest customer product name indicies and their similarity values
    N_BEST = N_BEST
    quality,indices = df_predict_LaBSE.topk(N_BEST)
    
    # Saving results
    df_dealerprice_unique.loc[:,'predict'] = indices.tolist()
    df_dealerprice_unique.loc[:, 'quality'] = quality.tolist()
    df_res=df_res.merge(df_dealerprice_unique, how='left', on=['product_name'])
    df_res['queue'] = [list(range(1, N_BEST+1)) for _ in range(len(df_res))]
    df_res = df_res.explode(['predict', 'queue', 'quality'])
    df_res = df_res.reset_index(drop=True)
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict', axis=1)
    df_res['create_date'] = datetime.now()

    return df_res.to_dict()