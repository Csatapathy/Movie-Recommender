import recommender 
def showMenu():
    print("Welcome to the Movie recommender system:")
    print("We have used the tmdb and the movielens dataset to create this recommender system and can recommend based on different theories!")
    print("Select a solution based on how you want the recommendations:")
    print("0. Show Menu")
    print("1. Demographic recommendations : get the most popular movies based on user ratings!")
    print("2. Content Based recommendations : get the top similar movies to a movie based on cosine similarity and Tf-IDF vectorization!")
    print("3. Factor Based recommendations : get better similar movies based on similar actors,directors and genres!")
    print("4. Collaborative recommendations : get recommendations based on what others most watch!")
    print("5. QUIT")


showMenu()

while True:
    
    n=int(input())
    if n==0:
        showMenu()
    if n==1:
        print("Enter the number of popular movies required:")
        x=int(input())
        print(recommender.demographicFiltering(x))
    elif n==2:
        print("Enter the name of the movie!")
        name=input()
        print("Enter the number of similar movies required!")
        x=int(input())
        print(recommender.contentFiltering(name,x))
    elif n==3:
        print("Enter the name of the movie!")
        name=input()
        print("Enter the number of similar movies required!")
        x=int(input())
        print(recommender.factorFiltering(name,x))
    elif n==4:
        print("Enter the name of the movie!")
        name=input()
        print("Enter the number of similar movies required!")
        x=int(input())
        print(recommender.contentFiltering(name,x))
    elif n==5:
        print("Thank You for showing interest in this application!")
        break


