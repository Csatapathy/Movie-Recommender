import pandas as pd
import numpy as np
from surprise import SVD,Dataset,Reader
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from ast import literal_eval



# DEMOGRAPHIC FILTERING
# ---------------------
def demographicFiltering(n):
    
    # Data Collection
    df1=pd.read_csv('tmdb_5000_credits.csv')
    df2=pd.read_csv('tmdb_5000_movies.csv')

    # Joining the two datasets on 'id' column
    df1.columns=['id','title','cast','crew']
    df2=df2.merge(df1.drop('title',axis=1),on='id')
    """
    @param:
    n: the no of popular movies required
    returns the n most popular movies
    """
    C = df2['vote_average'].mean() # this is the mean rating for all movies on a scale of 10
    m = df2['vote_count'].quantile(0.9)# minimum votes required to be in the chart, ie, for a movie to be in the charts the votes it must have must be more than 90% of the movies in the list
    q_movies=df2.copy().loc[df2['vote_count']>=m] # filtering out the movies that qualify

    def weighted_rating(x,m=m, c=C):
        """
        returns the weighted rating to apply on the qualified movies
        """
        v=x['vote_count']
        R=x['vote_average']
        return(v/(v+m)*R)+(m/(m+v)*c)

    q_movies['score']=q_movies.apply(weighted_rating,axis=1) # making a new col called score that would keep the WR

    # Popular recommendations
    pop=pd.DataFrame(q_movies.sort_values('score',ascending=False)[['title','vote_count','vote_average','score']].head(n))
    return pop


# CONTENT BASED FILTERING
# ---------------------
def contentFiltering(title,n):
        # Data Collection
    df1=pd.read_csv('tmdb_5000_credits.csv')
    df2=pd.read_csv('tmdb_5000_movies.csv')

    # Joining the two datasets on 'id' column
    df1.columns=['id','title','cast','crew']
    df2=df2.merge(df1.drop('title',axis=1),on='id')
    # importing TfIdVectorizer from sklearn
    
    tfidf=TfidfVectorizer(stop_words='english') # removes all the english stop words like 'the' and 'a'

    df2['overview']=df2['overview'].fillna(" ") # replace NaN with " "
    tfidf_matrix=tfidf.fit_transform(df2['overview']) # construct TF-IDF matrix by fitting and transforming

    
    cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
    # constructing a reverse map of indices and titles
    indices=pd.Series(df2.index,index=df2['title'])

    # def get_recommendations(title,cos_sim=cosine_sim,n):
    """
    Function that takes in movie titles and outputs n similar movies
    """
    if title in indices:
        idx=indices[title] # getting index of the title requested
        sim_scores=list(enumerate(cosine_sim[idx])) # get pairwise similarity scores of all movies with that movie
        sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True) # sort the movie based on similarity scores
        sim_scores=sim_scores[1:n] # get the scores of the top 10 movies
        movie_index=[i[0] for i in sim_scores]
        rec=pd.DataFrame(df2['title'].iloc[movie_index])
        rec.columns=['Title']
        rec.index=range(1,n)
        return rec
    else:
        print("OOPS! There is some error in the spelling of the movie.")



# CREDITS,GENRES,KEYWORDS BASED FILTERING
# ---------------------
def factorFiltering(hell,hell1):
    # Parse the stringified features into their corresponding python objects
        # Data Collection
    df1=pd.read_csv('tmdb_5000_credits.csv')
    df2=pd.read_csv('tmdb_5000_movies.csv')

    # Joining the two datasets on 'id' column
    df1.columns=['id','title','cast','crew']
    df2=df2.merge(df1.drop('title',axis=1),on='id')

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(literal_eval)

    # Get the director's name from the crew feature. If director is not listed, return NaN
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # Returns the list top 3 elements or entire list; whichever is more.
    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        #Return empty list in case of missing/malformed data
        return []

    # Define new director, cast, genres and keywords features that are in a suitable form.
    df2['director'] = df2['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        df2[feature] = df2[feature].apply(get_list)

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
                
    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        df2[feature] = df2[feature].apply(clean_data)

    # We are now in a position to create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).
    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
    df2['soup'] = df2.apply(create_soup, axis=1)

    # The next steps are the same as what we did with our plot description based recommender.
    # Import CountVectorizer and create the count matrix
    

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df2['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Reset index of our main DataFrame and construct reverse mapping as before
    df2 = df2.reset_index()
    indices = pd.Series(df2.index, index=df2['title'])
    # def get_recommendations(title,cos_sim=cosine_sim2,n):
    """
    Function that takes in movie titles and outputs n similar movies
    """
    if hell in indices:
        idx=indices[hell] # getting index of the title requested
        sim_scores=list(enumerate(cosine_sim2[idx])) # get pairwise similarity scores of all movies with that movie
        sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True) # sort the movie based on similarity scores
        sim_scores=sim_scores[1:hell1] # get the scores of the top 10 movies
        movie_index=[i[0] for i in sim_scores]
        rec=pd.DataFrame(df2['title'].iloc[movie_index])
        rec.columns=['Title']
        rec.index=range(1,hell1)
        return rec
    else:
        print("OOPS! There is some error in the spelling of the movie.")

# COLLABORATIVE FILTERING
# ---------------------

#  Since the dataset we used before did not have userId(which is necessary for collaborative filtering) let's load another dataset. We'll be using the Surprise library to implement SVD.
def collaborativeFiltering():
        # Data Collection
    df1=pd.read_csv('tmdb_5000_credits.csv')
    df2=pd.read_csv('tmdb_5000_movies.csv')

    # Joining the two datasets on 'id' column
    df1.columns=['id','title','cast','crew']
    df2=df2.merge(df1.drop('title',axis=1),on='id')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')

    reader=Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']],reader)
    # data.split(n_folds=5)
    svd = SVD()

    # Run 5-fold cross-validation and then print results
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)
    # here the root mean square error is 0.869- which is great!
    # training our dataset now
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    
