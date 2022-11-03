import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os 

#prepareddd----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
import pandas as pd
import nltk
def remove_stopwords(article_processed,words_to_add=[],words_to_remove=[]):
    ''' 
    takes in string, and two lists
    creates list of words to remove from nltk, modifies as dictated in arguements
    prints result of processing
    returns resulting string
    '''
    from nltk.corpus import stopwords
    #create the stopword list
    stopwords_list = stopwords.words("english")
    #modify stopword list
    [stopwords_list.append(word) for word in words_to_add]
    [stopwords_list.remove(word) for word in words_to_remove]
    #remove using stopword list
    words = article_processed.split()
    filtered_words = [w for w in words if w not in stopwords_list]
    #filtered_words =[word for word in article_processed if word not in stopwords_list]
    #print("removed ",len(article_processed)-len(filtered_words), "words")
    #join back
    article_without_stopwords = " ".join(filtered_words)
    return article_without_stopwords

def lemmatize(article):
    ''' 
    input article
    makes object, applies to string, and returns results
    '''
    import nltk
    #create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    #use lemmatizer
    lemmatized = [wnl.lemmatize(word) for word in article.split()]
    #join words back together
    article_lemmatized = " ".join(lemmatized)
    return article_lemmatized

def stem(article):
    ''' 
    input string
    create object, apply it to the each in string, rejoin and return
    '''
    import nltk
    #create porter stemmer
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in article.split()]
    #join words back together
    article_stemmed = " ".join(stems)
    return article_stemmed

def tokenize(article0):
    ''' 
    input string
    creates object, returns string after object affect
    '''
    import nltk
    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    #use the tokenizer
    article = tokenize.tokenize(article0,return_str=True)
    return article

def basic_clean(article0):
    ''' 
    input string
    lowers cases, makes "normal" characters, and removes anything not expected
    returns article
    '''
    import unicodedata
    import re
    #lower cases
    if isinstance(article0, float):
        article = str(article0).lower()
    else:
        article = article0.lower()
    ## decodes to change to "normal" characters after encoding to ascii from a unicode normalize
    article = unicodedata.normalize("NFKD",article).encode("ascii","ignore").decode("utf-8")
    # removes anything not lowercase, number, single quote, or a space
    article = re.sub(r'[^a-z0-9\'\s]','',article)
    return article

def prepare_df(df):
    df['rank'] = df.index + 1
    df=df.reset_index()
    # cleaning the data for world cloud
    df.region = df.region.map({'IND': 'India', 'JP': ' Japan', 'DE':'Germany','FR':'France','KR':'Korea','RU':'Russia','MX':'Mexico','BR':'Brazil','US':'United_States','CA':'Canada','GB':'United_Kingdon'})
    df = df[df.description.isna()==False]
    df["clean"] = [remove_stopwords(tokenize(basic_clean(each))) for each in df.description]
    df["stemmed"] = df.clean.apply(stem)
    df["lemmatized"] = df.clean.apply(lemmatize)
    df=df.drop(columns=('description'))
    # making categorid into actual category titles
    return df


#split----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#split
def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                      )
    return train, validate, test

#split
#stats----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#anova test
def comments_stats(train):  
    from scipy import stats
    alternative_hypothesis = 'comments_disabled is related to view_count'
    alpha = .05
    f, p = stats.f_oneway(train.comments_disabled, train.view_count)
    if p < alpha:
        print("We reject the null hypothesis")
        print("We can say that we have confidence that", alternative_hypothesis)
    else:
        print("We fail to reject the null")
        
def comments_stats2(train):  
    from scipy import stats
    alternative_hypothesis = 'comment_count is related to view_count'
    alpha = .05
    x = train.comment_count
    y = train.view_count
    corr, p = stats.pearsonr(x, y)
    if p < alpha:
        print("We reject the null hypothesis")
        print("We can say that we have confidence that", alternative_hypothesis)
#         print(f'corr={corr},p={p}')
    else:
        print("We fail to reject the null")
        
        
#graph----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def disable_comments(train):
    labels = pd.concat([train.comments_disabled.value_counts(),train.comments_disabled.value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    plt.figure(figsize=(16,16))
    mylabels = train['comments_disabled']
    textprops = {"fontsize":15}
    textprops = {"fontsize":15}
    plt.pie(labels.percent, labels = labels.index, textprops=textprops, autopct='%.1f%%')
    plt.legend()
    plt.title('Overall Distribution',fontsize=18)
    plt.show() 
    
def disable_comments2(train):
    plt.figure(figsize=(18,8))

    sns.barplot(data=train,x='comments_disabled',y='view_count')
    plt.show()
    
    
def comments_views(train):
    #the amount of comments effect the amoutn. of views 
    train = train[~train.index.duplicated()]
    plt.figure(figsize=(18,10))
    sns.lineplot(data=train,x='comment_count',y='view_count')
    plt.title('view per comment_count')
    plt.show()
    
def comments_views2(train):
    #the amount of comments effect the amoutn. of views 
    plt.figure(figsize=(18,10))
    sns.lineplot(data=train,x='comment_count',y='view_count',hue='categoryId')
    plt.title('view per comment_count')
    plt.show()
    
def category_views(df):
    labels = pd.concat([df.categoryId.value_counts(),df.categoryId.value_counts(normalize=True)], axis=1)
    labels.columns = ['n', 'percent']
    plt.figure(figsize=(16,16))
    mylabels = df['categoryId']
    textprops = {"fontsize":12}
    textprops = {"fontsize":12}
    plt.pie(labels.percent, labels = labels.index, textprops=textprops, autopct='%.1f%%')
    plt.legend(loc='upper left')
    plt.title('Overall Distribution',fontsize=18)
    plt.show() 
    
def category_views2(train):
    plt.figure(figsize=(18,8))
    sns.barplot(data=train,x='categoryId',y='view_count')
    plt.title('view per category')
    plt.xticks(rotation=90)
    plt.show()
    
def category_views3(train,population_name="categoryId",numerical_feature="view_count"):
    ''' 
    input df dataset and two strings (discrete and continous)
    does a ttest prints results, plots relation
    returns nothing
    '''
    
    #plot the results
    plt.figure(figsize=(25,25))
    plt.suptitle(f"Sample Values Compared for Non-Repeating Words", fontsize=12, y=0.99)
    i=0
    for feature in train[population_name].unique():
        temp1=train.copy()
        #plots out a grouping of the features
        i+=1
        ax = plt.subplot(5,3,i)
        temp1[population_name] = np.where(temp1[population_name]==feature,feature,"Other categoryId")
        temp1[[numerical_feature,population_name]].groupby(population_name).agg("mean").plot.bar(rot=0,color="blue",edgecolor="white",linewidth=5,ax=ax)
        ax.axhline(y=temp1[numerical_feature].mean(),label=f"Mean {(round(temp1[numerical_feature].mean(),3))}",color="red",linewidth=3)
        ax.set_ylabel("% of Non-Repeating Words")
        plt.legend()
        ax.set_title(f"{feature} means for likes",fontsize=8)
    plt.show()
    import scipy.stats as stats

    has_similar_list=[]
    not_similar_list=[]

    for sample_name in train[population_name].unique():
        # sets variables
        alpha = .05
        print(numerical_feature,"<-target |",population_name,"<-population name |",sample_name,"<-sample name")

        #sets null hypothesis
        H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding likes"
        Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a likes"

        #runs test and prints results
        t, p = stats.ttest_1samp( train[train[population_name] == sample_name][numerical_feature], train[numerical_feature].mean())
        if p > alpha:
            print("We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
            has_similar_list.append(sample_name)
        else:
            print("We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))
            not_similar_list.append(sample_name)
        print("----------")
    
    
def sport_biograms(train):
    Sports = ' '.join(train[train.categoryId == 'Sports'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Sports, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='yellow', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring sport bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
def entertainment_biograms(train):
    Entertainment = ' '.join(train[train.categoryId == 'Entertainment'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Entertainment, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='blue', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring Entertainment bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
def gaming_biograms(train):
    Gaming = ' '.join(train[train.categoryId == 'Gaming'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Gaming, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='red', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring Entertainment bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def Music_biograms(train):
    Music = ' '.join(train[train.categoryId == 'Music'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Music, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='green', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring Entertainment bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def word_count(train):
    lang_dict={"Language":[],"Words":[]}
    for lang in train["categoryId"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(train[train["categoryId"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    sns.catplot(data=lang, x="count_set_words", y="Language", kind="bar",height=11,aspect=1.5)
    plt.title('Total count of words')
    plt.show()
    print(lang.count_set_words)
    
def word_count2(train):
    lang_dict={"Language":[],"Words":[]}
    for lang in train["categoryId"].unique():
        lang_dict["Language"].append(lang)
        lang_dict["Words"].append((" ".join(train[train["categoryId"]==lang]["lemmatized"])).replace("'","").split())
    lang = pd.DataFrame(lang_dict)
    most_common_list=[]
    for i,each in enumerate(lang["Language"].unique()):
        looped_series = pd.Series(lang["Words"].loc[i]).value_counts()
        most_common = looped_series[looped_series > looped_series.quantile(.95)]
        most_common_list.append(most_common[:5].index.tolist())
    lang["most_common"] = pd.Series(most_common_list)
    lang["count_set_words"] = lang["Words"].apply(set).apply(len)
    sns.pointplot(data=lang, x="count_set_words", y="Language",height=11,aspect=1.5)
    plt.title('Total count of words')
    plt.show()
    
    
def region_category(train):
    #fix the percentages
    plt.figure(figsize=(30,20))
    plt.title('view per category')
    plt.xticks(rotation=90)
    sns.barplot(data=train,x=train.categoryId, y=train.view_count,hue=train.region)
    plt.show()
    
def region_category2(train):
    #fix the percentages
    plt.figure(figsize=(15,10))
    plt.title('likes per category/region')
    plt.xticks(rotation=90)
    sns.swarmplot(data=train,x=train.categoryId, y=train.likes,hue=train.region)
    plt.show()