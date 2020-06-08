# Evaluat-inator
## Movie Recommendation System | Machine Learning
### Types of recommendation system 
There are two types of recommendation system:-

<div style="text-align:center"><img src="https://github.com/avyaktawrat/Evaluat-inator/blob/master/images/Types-of-Recommendation-Systems.jpg" width="400">

### - Content based recommendation system.
First, the system executes a model-building stage by finding the similarity between all pairs of items. This similarity function can take many forms, such as correlation between ratings or cosine of those rating vectors.<br/>
Second, the system executes a recommendation stage. It uses the most similar items to a user's already-rated items to generate a list of recommendations.

### - Collaborative recommendation system
A user expresses his or her preferences by rating items (e.g. books, movies or CDs) of the system. These ratings can be viewed as an approximate representation of the user's interest in the corresponding domain.<br/>
The system matches this user's ratings against other users' and finds the people with most "similar" tastes.<br/>
With similar users, the system recommends items that the similar users have rated highly but not yet being rated by this user.

#### Data Set
Data set of 1M ratings is used taken from [Movie lens](https://grouplens.org/datasets/movielens/).<br/>
It contains around 1M ratings given by around 6k users on around 4k movies. 
#### Libraries used 
### EDA

###
## KNN based Approach 
- The data is read in data frame as  *ratings, users*  and  *movies.* These df's are processed as discribed in EDA section.<br/>
- The processed data is used to create a matrix(namely  *movie_user_mat* ) between moviesId and userId as rows and columns respectively. The values of the cell of matrix( *movie_user_mat[i, j]* ) is the rating given by  j<sup>th</sup> user on i<sup>th</sup> movie. This matrix is transformed into scipy sparse matrix for easy computation.
- A mapper(namely  *movie_to_idx* ) is a dictionary which is created, that maps movie to it's index according to *movies* dataframe.
- The matrix is fed into NearestNeighbors model of sklearn. 'Cosine' similarity metric is used with brute algorithm.
```
# define model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit
model_knn.fit(movie_user_mat_sparse)
```
-  *Fuzzy_matching function*  takes in favorite movie as input and gives out index of most similar movie listed in mapper. The similarity is calculated via fuzz ratio.
- In the function  *make_recommendation*  the data(i.e. the movie_user_mat) is fit in knn model, it then finds n nearest neighbours of data[idx], where idx given out by  *Fuzzy_matching function* for favorite movie. <br/>
The distance is sorted in top n neighbours with maximum distance( i.e. minimum angle as cosine similarity is used) is printed.
```
    # fit
    model_knn.fit(data)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
```
<div style="text-align:center"><img src="https://github.com/avyaktawrat/Evaluat-inator/blob/master/images/knn%20result.PNG" width="600">
