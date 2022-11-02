# -*- coding: utf-8 -*-

# import necessary libraries
import pandas as pd
import os
import glob
  
from os import path
from requests import get
import numpy as np
import re
import unicodedata
import prepare
import spacy



def acquire ():
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    # use glob to get all the csv files 
    # in the folder
    path = os.getcwd()
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    df = pd.DataFrame()  
    # loop over the list of csv files
    for f in csv_files:
        
        # read the csv file
        temp = pd.read_csv(f)
        
        # print the location and filename
        temp['region']=f.split("\\")[-1][-13:-11]

        df= pd.concat([df, temp])

    df['rank'] = df.index + 1
    df['top_25'] = np.where(df['rank'] < 26, 1, 0)
    df.description = df.description.fillna('no description')
    df[df.duplicated(['video_id'], keep=False) == True].sort_values(by='rank').drop_duplicates(['video_id'], inplace=True)
    df.region = np.where(df.region == 'IN', 'IND', df.region)
    return df

def clean_duration(duration):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    if ('S' not in duration) & ('M' not in duration):
        duration += '00M00S'
    elif 'M' not in duration:
        duration = list(duration)
        print('stop')
        print(list(duration))
        duration.insert(-3, '00M') if len(duration) > 6 else duration.insert(-2, '00M')
        duration = ''.join(duration)
    elif 'S' not in duration:
        duration += '00S'

    if 'H' in duration:
        duration = int(duration.split('H')[0].split('T')[1]) * 3600 + int(duration.split('H')[1].split('M')[0]) * 60 + int(duration.split('M')[1][:-1])
    elif duration.__contains__('M'):
        duration = int(duration.split('M')[0].split('T')[1]) * 60 + int(duration.split('M')[1][:-1])
    else:
        duration = int(duration[-3:-1])

    return duration

def clean_text(text):
    """ 
    Purpose:
        to clean text input into function by removing duplicate words, punctuations, and other things
    ---
    Parameters:
        text: a string
    ---
    Returns:
        tokens: a set of words found in the input text
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = set()
    
    for token in doc:
        if token.pos_ not in ['SYM', 'PUNCT', 'DET']:
            tokens.add(token.text)

    for ent in doc.ents:
        tokens.add(ent.text)

    return tokens
    
def title_in_desc(df):
    df['title_in_description'] = 0
    for row in df.index:
        title = df.iloc[row]['title']
        description = df.iloc[row]['description']
                    
        if title in description:
                df.loc[row, 'title_in_description'] = 1
    return df
    
def title_in_tags (df):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    df['title_in_tags'] = 0
    for row in df.index:
        title = df.iloc[row]['title']
        tags = df.iloc[row]['tags']
            
    if title in tags:
            df.loc[row, 'title_in_tags'] = 1
    return df
def pct_tags(df):
    
    df['pct_tags_in_description'] = 0
    for row in df.index:
        counter = 0
        tags = df.iloc[row]['cleaned_tags']
        description = df.iloc[row]['cleaned_desc']

        for tag in tags:
            if tag in description:
                counter += 1
        pct = counter/len(tags)
        df.loc[row, 'pct_tags_in_description'] = pct

    return df
    
def create_bank(text):
    """ 
    Purpose:
        to clean text input into function by removing duplicate words, punctuations, and other things
    ---
    Parameters:
        text: a string
    ---
    Returns:
        tokens: a set of words found in the input text
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = list()
    
    for token in doc:
        if token.pos_ not in ['SYM', 'PUNCT', 'DET']:
            tokens.append(token.text)

    for ent in doc.ents:
        tokens.append(ent.text)

    return tokens
    
def create_vaults(df):        #these vaults are to create the word clouds for each portion
    word_vault = list()
    for row in df.index:
        word_vault.extend(df.iloc[row].word_bank)

    top_25_words = list()
    for row in df[df.top_25==1].index:
        top_25_words.extend(df[df.top_25==1].iloc[row].word_bank)

    outside_25_words = list()
    for row in df[df.top_25!=1].index:
        outside_25_words.extend(df[df.top_25!=1].iloc[row].word_bank)

    return word_vault, top_25_words, outside_25_words


def clean_duration(duration):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    if ('S' not in duration) & ('M' not in duration):
        duration += '00M00S'
    elif 'M' not in duration:
        duration = list(duration)
        print('stop')
        print(list(duration))
        duration.insert(-3, '00M') if len(duration) > 6 else duration.insert(-2, '00M')
        duration = ''.join(duration)
    elif 'S' not in duration:
        duration += '00S'

    if 'H' in duration:
        duration = int(duration.split('H')[0].split('T')[1]) * 3600 + int(duration.split('H')[1].split('M')[0]) * 60 + int(duration.split('M')[1][:-1])
    elif duration.__contains__('M'):
        duration = int(duration.split('M')[0].split('T')[1]) * 60 + int(duration.split('M')[1][:-1])
    else:
        duration = int(duration[-3:-1])

    return duration

def clean_text(text):
    """ 
    Purpose:
        to clean text input into function by removing duplicate words, punctuations, and other things
    ---
    Parameters:
        text: a string
    ---
    Returns:
        tokens: a set of words found in the input text
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = set()
    
    for token in doc:
        if token.pos_ not in ['SYM', 'PUNCT', 'DET']:
            tokens.add(token.text)

    for ent in doc.ents:
        tokens.add(ent.text)

    return tokens
    
def title_in_desc(df):
    df['title_in_description'] = 0
    for row in df.index:
        title = df.iloc[row]['title']
        description = df.iloc[row]['description']
                    
        if title in description:
                df.loc[row, 'title_in_description'] = 1
    return df
    
def title_in_tags (df):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    df['title_in_tags'] = 0
    for row in df.index:
        title = df.iloc[row]['title']
        tags = df.iloc[row]['tags']
            
    if title in tags:
            df.loc[row, 'title_in_tags'] = 1
    return df

def pct_tags(df):
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    df['pct_tags_in_description'] = 0
    for row in df.index:
        counter = 0
        tags = df.iloc[row]['cleaned_tags']
        description = df.iloc[row]['cleaned_desc']

        for tag in tags:
            if tag in description:
                counter += 1
        pct = counter/len(tags)
        df.loc[row, 'pct_tags_in_description'] = pct

    return df
    
def create_bank(text):
    """ 
    Purpose:
        to clean text input into function by removing duplicate words, punctuations, and other things
    ---
    Parameters:
        text: a string
    ---
    Returns:
        tokens: a set of words found in the input text
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = list()
    
    for token in doc:
        if token.pos_ not in ['SYM', 'PUNCT', 'DET']:
            tokens.append(token.text)

    for ent in doc.ents:
        tokens.append(ent.text)

    return tokens
    
def create_vaults(df):        #these vaults are to create the word clouds for each portion
    word_vault = list()
    for row in df.index:
        word_vault.extend(df.iloc[row].word_bank)

    top_25_words = list()
    for row in df[df.top_25==1].index:
        top_25_words.extend(df[df.top_25==1].iloc[row].word_bank)

    outside_25_words = list()
    for row in df[df.top_25!=1].index:
        outside_25_words.extend(df[df.top_25!=1].iloc[row].word_bank)

    return word_vault, top_25_words, outside_25_words

def prepare_youtube():
    """ 
    Purpose:
        
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    df = acquire()
    
    # creates ranking for each video based off of country index
    df.categoryId = df.categoryId.astype('object')
    df.publishedAt = pd.to_datetime(df.publishedAt, utc=True)
    df.trending_date = '22.2.11'
    df.trending_date = pd.to_datetime(df.trending_date, format='%y.%d.%m', utc=True)
    df['age']=(df.trending_date - df.publishedAt)
    df['engagement'] = (df.likes + df.comment_count * 4 )/df.view_count
    df['sponsored'] = np.where(df.description.str.contains('sponsor'), 1, 0)
    df['duration'] = df['duration'].apply(lambda x:clean_duration(x))
    #countes number of tags given to video BEFORE stripping out extraneous things
    df['num_of_tags'] = df.tags.str.split('|').str.len()
    #gets rid of separator
    df.tags = df.tags.str.replace('|'," ")
    #import the create bank function first
    df['word_bank']= df['description'].apply(lambda x: create_bank(x))
    #import clean_text function first
    df['cleaned_tags'] =  df['tags'].apply(lambda x: clean_text(x))
    df['cleaned_desc'] = df['description'].apply(lambda x: clean_text(x))
    df = title_in_desc(df)
    df = title_in_tags(df)
    df = pct_tags(df)
    #making categorid into actual category titles
    #all categoryId optain from youtube website
    df.categoryId = df.categoryId.map({1: 'Film_Animation', 2: 'Autos_Vehicles',10:'Music',15: 'Pets_Animals'
                                       ,17:'Sports',18:'Short_Movies',19:'Travel_Events',20:'Gaming',21:'Videoblogging',
                                       22:'People_Blogs',23:'Comedy',24:'Entertainment',25:'News_Politics',
                                       26:'Howto_Style',27: 'Education',28: 'Science_Technology', 
                                       29:'Nonprofits_Activism',30:'Movies',31:'Anime/Animation',32:'Action/Adventure',
                                       33:'Classics',34:'Comedy',35:'Documentary',36:'Drama',37:'Family',38:'Foreign',
                                       39:'Horror',40: 'Sci-Fi/Fantasy', 41: 'Thriller', 42:'Shorts',43:'Shows',44:'Trailers'})
    return df

def final_prep():
    """ 
    Purpose:
        To perform final cleaning of DF on either pickled file or crea
    ---
    Parameters:
        
    ---
    Returns:
    
    """
    df = prepare_youtube()
    #converts the age category timedelta into hours .. so the videos are x hours old now 
    df.age = (df.age.dt.days * 24) + (df.age.dt.seconds/3600)
    df['title_lengths'] = df["title"].apply(lambda x: len(x))
    df['desc_lengths'] = df["description"].apply(lambda x: 0 if pd.isnull(x) else len(x))
    df['tags_length'] = df['tags'].apply(lambda x: len(x.replace('|', '')) if x != '[none]' else 0)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)

    return df