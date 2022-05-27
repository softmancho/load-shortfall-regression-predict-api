"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json


def split_time(df):
    #make a copy of the original dataframe
    df_copy=df.copy()
    
    #Create a DataFrame with Datetime values
    df_copy['time']=pd.to_datetime(df_copy['time'])
    
    #The year of the datetime.
    df_copy['Year']=df_copy['time'].dt.year
    
    #The month as January=1, December=12.
    df_copy['Month']=df_copy['time'].dt.month
    
    #The day of the datetime.
    df_copy['Day']=df_copy['time'].dt.day
    
    #The hours of the datetime.
    df_copy['Hour']=df_copy['time'].dt.hour
    
    #The minutes of the datetime.
    df_copy['Minute']=df_copy['time'].dt.minute
    
    #The day of the week with Monday=0, Sunday=6.
    df_copy['DayOfWeek']=df_copy['time'].dt.dayofweek
    
    #The ordinal day of the year.
    df_copy['DayOfYear']=df_copy['time'].dt.dayofyear
    
    #The quarter of the date.
    df_copy['Quarter']=df_copy['time'].dt.quarter
    
    #drop the time column
    df_copy.drop(columns='time',axis=1,inplace=True)
    return df_copy

    #We are interested in finding out  the effect of weather on the the energy loadshorfall in the main Spanish cities.
def season_of_date(df):
    #make a copy of the original dataframe
    df_copy=df.copy()
    
    for month in df['Month']:
        
        #assign winter=1 when 
        if (month==12 or month <3):
            df_copy['Winter']=1
        elif month < 6:
            df_copy['Spring']=2
        elif month < 9:
            df_copy['Summer']=3
        else:
            df_copy['Autumn']=4  
  
    return df_copy

    #Since our dataset now have columns for year, and month. 
    #A good approach to take when replacing the missing values in the valencia_pressure 
    # columns is to replace with the average reading for the month in which the reading was made

    #- The mean values of the `Valencia_pressure` is aggregated by Year and month
    #- Missing values are then replaced with the mean of the corresponding month

def replace_valencia_pressure(df):
    #make a copy of the original dataframe
    df_copy=df.copy()
    
    df_copy['Valencia_pressure']=df_copy.groupby(['Year', 'Month'])['Valencia_pressure'].transform(func = lambda x: x.fillna(x.mean() )  )      
        
    return df_copy

    #There are several ways to handle categorical variables;

    #- The `handle_categorical_column` function creates dummy variables for each column. 
    #- The `handle_categorical_column_v2` function simply subsets the string values to portions 
    # that can be parsed as into numeric values

    #We have both so that we can safely avoid the problem of a data set with too many dimensions 
    # and not enough data to train the model

def handle_categorical_column(input_df, colunmn_name):
    #make a copy of the original dataframe
    copy_df = input_df.copy()
    # extract the numerical value from the columns 
    copy_df[colunmn_name] = copy_df[colunmn_name].str.extract('(\d+)')
    copy_df[colunmn_name] = pd.to_numeric(copy_df[colunmn_name])
    # your code here
    return copy_df

    ###  4.7.0  Droping noise(unimportant_ columns ) in our dataset
    # drop any unneccesary column

def drop_columns(input_df):
    #make a copy of the original dataframe
    copy_df = input_df.copy()
    
    # columns to drop because of the are indexes and donot contribute to our model performance
    irrelevant_columns = ['Bilbao_weather_id',
                            'Seville_weather_id',
                            'Barcelona_weather_id',
                            'Barcelona_temp',
                             'Barcelona_temp_min',
                             'Bilbao_temp',
                             'Bilbao_temp_max',
                             'Madrid_temp',
                             'Madrid_temp_min',
                             'Seville_temp_min',
                             'Valencia_temp',
                             'Valencia_temp_min']
    
    drop_total = irrelevant_columns 
    copy_df.drop(drop_total,inplace=True, axis=1)
    return copy_df

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    
    feature_vector_df.drop('Unnamed: 0',axis=1, inplace=True)
    feature_vector_df=split_time(feature_vector_df)
    feature_vector_df=replace_valencia_pressure(feature_vector_df )
    feature_vector_df=handle_categorical_column(feature_vector_df ,'Valencia_wind_deg')
    feature_vector_df=handle_categorical_column(feature_vector_df ,'Seville_pressure' )
    feature_vector_df=drop_columns(feature_vector_df )



    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
