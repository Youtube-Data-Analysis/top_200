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
    '''preparing the data frame by renaming region names
    rreplacing the nulls with non desceription'''
    # cleaning the data for world cloud
    df.region = df.region.map({'IND': 'India', 'JP': ' Japan', 'DE':'Germany','FR':'France','KR':'Korea','RU':'Russia','MX':'Mexico','BR':'Brazil','US':'United_States','CA':'Canada','GB':'United_Kingdon'})
    df = df[df.description.isna()==False]
    df["clean"] = [remove_stopwords(tokenize(basic_clean(each))) for each in df.description]
    df["stemmed"] = df.clean.apply(stem)
    df["lemmatized"] = df.clean.apply(lemmatize)
    #df=df.drop(columns=('description'))
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
    ''' a nova test that compares comments disabled to view count'''
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
    '''corr test that compared commment count to engagement '''
    from scipy import stats
    alternative_hypothesis = 'comment_count is related to engagement'
    alpha = .05
    x = train.comment_count
    y = train.engagement
    corr, p = stats.pearsonr(x, y)
    if p < alpha:
        print("We reject the null hypothesis")
        print("We can say that we have confidence that", alternative_hypothesis)
#         print(f'corr={corr},p={p}')
    else:
        print("We fail to reject the null")
        
        
#graph----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_conf(ax, xlbl='', ylbl='', t='', back_color='#ffffff',
              text_color='#616161', grid_color='#e9e9e9', 
              tick_color='', ticklbl_size=9, lbl_size=11, lang='en'):
    """
    This function perform operations to produce better-looking 
    visualizations
    """
    # changing the background color of the plot
    ax.set_facecolor(back_color)
    # modifying the ticks on plot axes
    ax.tick_params(axis='both', labelcolor=text_color, color=back_color)
    if tick_color != '':
        ax.tick_params(axis='both', color=tick_color)
    ax.tick_params(axis='both', which='major', labelsize=ticklbl_size)
    # adding a grid and specifying its color
    ax.grid(True, color=grid_color)
    # making the grid appear behind the graph elements
    ax.set_axisbelow(True)
    # hiding axes
    ax.spines['bottom'].set_color(back_color)
    ax.spines['top'].set_color(back_color) 
    ax.spines['right'].set_color(back_color)
    ax.spines['left'].set_color(back_color)
    # setting the title, x label, and y label of the plot
    if lang == 'ar':
        ax.set_title(get_display(reshaper.reshape(t)), fontweight='bold', family='Amiri',
                     fontsize=14, color=text_color, loc='right', pad=24);
        ax.set_xlabel(get_display(reshaper.reshape(xlbl)), fontweight='bold', family='Amiri',
                      labelpad=16, fontsize=lbl_size, color=text_color, fontstyle='italic');
        ax.set_ylabel(get_display(reshaper.reshape(ylbl)), fontweight='bold', family='Amiri',
                      color=text_color, labelpad=16, fontsize=lbl_size, fontstyle='italic');
    else:
        ax.set_title(t, fontsize=14, color='#616161', loc='left', pad=24, fontweight='bold');
        ax.set_xlabel(xlbl, labelpad=16, fontsize=lbl_size, color='#616161', fontstyle='italic');
        ax.set_ylabel(ylbl, color='#616161', labelpad=16, fontsize=lbl_size, fontstyle='italic');
        
def graph_overall(df):
    '''This function is to view a overall graph of every categories '''
    tdf = df.groupby("categoryId").size().reset_index(name="video_count").sort_values("video_count", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(14,10))
    sns.barplot(x="video_count", y="categoryId", data=tdf,
                palette=sns.color_palette('gist_heat', n_colors=25)[3:], ax=ax);
    plot_conf(ax, xlbl='Number of videos', ylbl='categoryId', ticklbl_size=11, lbl_size=12)
    plt.tight_layout()

def disable_comments(train):
    '''graph for disable comments overall'''
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
    '''do comments effect the amount of views a video would acquired'''
    plt.figure(figsize=(18,8))

    sns.barplot(data=train,x='comments_disabled',y='view_count')
    plt.show()
    
    
def comments_views(train):
    ''' comment count graph cmompared to engagement'''
    
    #the amount of comments effect the amoutn. of views 
    train = train[~train.index.duplicated()]
    plt.figure(figsize=(18,10))
    sns.lineplot(data=train,x='comment_count',y='engagement')
    plt.title('view per comment_count')
    plt.show()
    
def comments_views2(train):
    '''comment count graph compared to engagement'''
    #the amount of comments effect the amoutn. of views 
    plt.figure(figsize=(18,10))
    sns.regplot(data=train[train.comment_count < 30000],x='subscribers',y='engagement')
    plt.title('view per comment_count')
    plt.show()
    #the amount of comments effect the amoutn. of views 
    plt.figure(figsize=(18,10))
    sns.regplot(data=train[train.comment_count < 30000],x='subscribers',y='engagement')
    plt.xlim(0,4000000)
    plt.title('view per comment_count')
    plt.show()
    
def category_views(df):
    '''category graph that show the percent of all category'''
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
    '''category compared to view count graph '''
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
    
    
def sport_bigrams(train):
    '''bigrams sport'''
    Sports = ' '.join(train[train.categoryId == 'Sports'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Sports, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='yellow', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring bigrams in the Sports category')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
def entertainment_bigrams(train):
    '''bigrams entertainment'''
    Entertainment = ' '.join(train[train.categoryId == 'Entertainment'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Entertainment, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='blue', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring bigrams in the Entertainment category')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
def gaming_bigrams(train):
    '''bigramsgaming '''
    Gaming = ' '.join(train[train.categoryId == 'Gaming'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Gaming, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='red', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring bigrams in the Gaming category')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def Music_bigrams(train):
    '''bigrams music'''
    Music = ' '.join(train[train.categoryId == 'Music'].lemmatized).split()
    top_20_ham_bigrams = (pd.Series(nltk.ngrams(Music, 2))
                          .value_counts()
                          .head(25))
    top_20_ham_bigrams.sort_values(ascending=False).plot.barh(color='green', width=.9, figsize=(10, 6))

    plt.title('25 Most frequently occuring bigrams in the Music category')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

        # make the labels pretty
    ticks, _ = plt.yticks()
    labels = top_20_ham_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    
    
def word_count(train):
    '''total word count for all of categories'''
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
    lang["count_set_words"] = lang["Words"].apply(set).apply(len).mode()
    result = lang.groupby(["Language"])['count_set_words'].aggregate(np.median).reset_index().sort_values('Language')
    sns.catplot(data=lang, x="count_set_words", y="Language", kind="bar",height=11,aspect=1.5,order=result['Language'])
    plt.title('Total count of words')
    plt.show()
    
def word_count2(train):
    '''total word count for all of categories'''
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
    result = df.groupby(["categoryId"])['categoryId'].aggregate(np.median).reset_index().sort_values('categoryId')
    sns.pointplot(data=lang, x="count_set_words", y="Language",order=result['categoryId'])
    plt.title('Total count of words')
    plt.show()
    
    

def Region_categories(train):
    '''comparing categories with region '''
    top25_only=train[train.top_25==1]
    Not_top=train[train.top_25==0]
    Russia=top25_only[top25_only.region=='Russia']
    Korea=top25_only[top25_only.region=='Korea']
    Germany=top25_only[top25_only.region=='Germany']
    Japan=top25_only[top25_only.region=='Japan']
    Mexico=top25_only[top25_only.region=='Mexico']
    Brazil=top25_only[top25_only.region=='Brazil']
    France=top25_only[top25_only.region=='France']
    United_Kingdon=top25_only[top25_only.region=='United_Kingdon']
    Canada=top25_only[top25_only.region=='Canada']
    United_States=top25_only[top25_only.region=='United_States']
    India=top25_only[top25_only.region=='India']
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=.5, right=.8, top=2, wspace=.3, hspace=1)
    #
    plt.subplot(521) 
    ax1 = sns.barplot(data=Mexico, x='categoryId', y='view_count')
    ax1.tick_params(labelrotation=90)
    ax1.title.set_text('Mexico by categoryId') 
    plot_conf(ax1)
    #
    plt.subplot(522) 
    ax2 = sns.barplot(data=Russia, x='categoryId', y='view_count')
    ax2.tick_params(labelrotation=90)
    ax2.title.set_text('Russia by categoryId') 
    plot_conf(ax2)
    #
    plt.subplot(523) 
    ax3 = sns.barplot(data=Korea, x='categoryId', y='view_count')
    ax3.tick_params(labelrotation=90)
    ax3.title.set_text('Korea by categoryId ')
    plot_conf(ax3)
    #
    plt.subplot(524) 
    ax4 = sns.barplot(data=Brazil, x='categoryId', y='view_count')
    ax4.tick_params(labelrotation=90)
    ax4.title.set_text('Brazil by categoryId')
    plot_conf(ax4)
    #
    plt.subplot(525)
    ax5 = sns.barplot(data=Germany, x='categoryId', y='view_count')
    ax5.tick_params(labelrotation=90)
    ax5.title.set_text('Germany by categoryId')
    plot_conf(ax5)
    #
    plt.subplot(526)
    ax6 = sns.barplot(data=France, x='categoryId', y='view_count')
    ax6.tick_params(labelrotation=90)
    ax6.title.set_text('France by categoryId')
    plot_conf(ax6)
    #
    plt.subplot(527) 
    ax4 = sns.barplot(data=United_Kingdon, x='categoryId', y='view_count')
    ax4.tick_params(labelrotation=90)
    ax4.title.set_text('United_Kingdon by categoryId')
    plot_conf(ax4)
    #
    plt.subplot(528)
    ax5 = sns.barplot(data=Canada, x='categoryId', y='view_count')
    ax5.tick_params(labelrotation=90)
    ax5.title.set_text('Canada by categoryId')
    plot_conf(ax5)
    #
    plt.subplot(529)
    ax6 = sns.barplot(data=United_States, x='categoryId', y='view_count')
    ax6.tick_params(labelrotation=90)
    ax6.title.set_text('United_States by categoryId')
    plot_conf(ax6)
    
def region_category2(train):
    '''Does region effect what categories would be trending or not '''
    #fix the percentages
    plt.figure(figsize=(15,10))
    plt.title('likes per category/region')
    plt.xticks(rotation=90)
    sns.swarmplot(data=train,x=train.categoryId, y=train.likes,hue=train.region)
    plt.show()