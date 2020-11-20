# Movie Recommender Project:

In this project, I have tried to make a recommender system that gives an option of various recommendation algorithms to used in this project.

## Data :
The dataset for this system is generated from the [tmdb api website](https://developers.themoviedb.org/3/getting-started/introduction).
1. The tmdb_5000_credits dataset contains data:
    * movie_id -A unique identifier for each movie.
    * cast - The name of lead and supporting actors.
    * crew - The name of Director, Editor, Composer, Writer etc.
2. The tmdb_5000_movies dataset contains:
    * budget - The budget in which the movie was made.
    * genre - The genre of the movie, Action, Comedy ,Thriller etc.
    * homepage - A link to the homepage of the movie.
    * id - This is infact the movie_id as in the first dataset.
    * keywords - The keywords or tags related to the movie.
    * voriginal_language - The language in which the movie was made.
    * original_title - The title of the movie before translation or adaptation.
    * overview - A brief description of the movie.
    * popularity - A numeric quantity specifying the movie popularity.
    * production_companies - The production house of the movie.
    * production_countries - The country in which it was produced.
    * release_date - The date on which it was released.
    * revenue - The worldwide revenue generated by the movie.
    * runtime - The running time of the movie in minutes.
    * status - "Released" or "Rumored".
    * tagline - Movie's tagline.
    * title - Title of the movie.
    * vote_average - average ratings the movie recieved.
    * vote_count - the count of votes recieved.
 ## Types of recommender systems used: 
 ### 1. Demographic Filtering:
This algorithm offers generalized recommnendations to every user based on movie popularity and (sometimes) genre. The basic idea behind this recommender is that movies that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience. This model does not give personalized recommendations based on the user.

The implementation of this model is extremely trivial. All we have to do is sort our movies based on ratings and popularity and display the top movies of our list.

I will use IMDB's weighted rating formula to construct my chart. Mathematically, it is represented as follows:

![image](https://user-images.githubusercontent.com/68659873/99789721-09aef500-2b49-11eb-859d-04e8fc4ed549.png)

where,
    * v is the number of votes for the movie
    * m is the minimum votes required to be listed in the chart
    * R is the average rating of the movie
    * C is the mean vote across the whole report
we would be getting the result like this:

![image](https://user-images.githubusercontent.com/68659873/99789989-75915d80-2b49-11eb-88c0-a45ad86bc187.png)

### 2. Content Based Filtering:
In this recommender system the content of the movie (overview, cast, crew, keyword, tagline etc) is used to find its similarity with other movies. Then the movies that are most likely to be similar are recommended. We will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score. 

We will be calculating the similarity using the TF-IDF vectorization and cosine similarity. We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate. The formula is:

![image](https://user-images.githubusercontent.com/68659873/99790194-c6a15180-2b49-11eb-82e0-f97fb66c9e5d.png)

We are now in a good position to define our recommendation function. These are the following steps we'll follow :-
* Get the index of the movie given its title.
* Get the list of cosine similarity scores for that particular movie with all movies. Convert it into * a list of tuples where the first element is its position and the second is the similarity score.
* Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
* Get the top n elements of this list. Ignore the first element as it refers to self (the movie most similar to a particular movie is the movie itself).
* Return the titles corresponding to the indices of the top elements.

This is how the result looks like for <b>"The Dark Knight Rises"</b>:

![image](https://user-images.githubusercontent.com/68659873/99790735-87bfcb80-2b4a-11eb-896b-2e022d62cd50.png)


### 3. Credits, Genres and Keywords Based Recommender:
It goes without saying that the quality of our recommender would be increased with the usage of better metadata. That is exactly what we are going to do in this section. We are going to build a recommender based on the following metadata: the 3 top actors, the director, related genres and the movie plot keywords.

We would be doing the same process as done in content based filtering but one import difference is that we would be using CountVectorizer() instead of TF-IDF. This is because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies. It doesn't make much intuitive sense.

This is how the result looks likefor <b>"The Dark Knight Rises"</b>:

![image](https://user-images.githubusercontent.com/68659873/99790593-5e06a480-2b4a-11eb-8da9-c15a0d48378e.png)

### 4. Collaborative Filtering:
Our other algorithms don't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who she/he is.

Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to Movie Watchers. It is basically of two types:-

1.<ins><b>User based filtering</ins></b>- These systems recommend products to a user that similar users have liked. For measuring the similarity between two users we can either use pearson correlation or cosine similarity. 
2.<ins><b>Item Based Collaborative Filtering</ins></b>- Instead of measuring the similarity between users, the item-based CF recommends items based on their similarity with the items that the target user rated. Likewise, the similarity can be computed with Pearson Correlation or Cosine Similarity. The major difference is that, with item-based collaborative filtering, we fill in the blank vertically, as oppose to the horizontal manner that user-based CF does. 

However, several problems remain for this method. First, the main issue is scalability. The computation grows with both the customer and the product. The worst case complexity is O(mn) with m users and n items. In addition, sparsity is another concern.

#### Single Value Decomposition
One way to handle the scalability and sparsity issue created by CF is to leverage a latent factor model to capture the similarity between users and items. Essentially, we want to turn the recommendation problem into an optimization problem. We can view it as how good we are in predicting the rating for items given a user. One common metric is Root Mean Square Error (RMSE). The lower the RMSE, the better the performance.

## Code:
1. analysis.ipynb : contains the analysis and the step by step introduction to every one of these algorithms and instructions on how to approach them.
2. recommender.py : this file contains the methods for the recommender system
3. main.py : this file is the user interface side file that interacts with the user while calling the recommender.py



