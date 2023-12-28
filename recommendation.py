import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
 
 

def create_similarity_matrix(data_path):
    
    generated_user_data = pd.read_csv(data_path)
    gdf=pd.DataFrame(generated_user_data)
    gdf=gdf.head(200)

    gdf['Interests'] = gdf['Interests'].apply(literal_eval)
    # print(gdf['Id'].head())

    def create_soup(x):
        return str.lower(' '.join(x['Interests']) +' '+' '.join(x['Interests']) + ' ' +x['City']+' '+x['Situation']+' '+x['Noise']+' '+x['SleepSchedule']+' '+x['Personnality'])

    gdf['soup'] = gdf.apply(create_soup, axis=1)
    # print(gdf.head(2))


    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(gdf['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    #Construct a reverse map of indices and userIds
    indices = pd.Series(gdf.index,index=gdf['Id'].values).drop_duplicates()

    cosine_sim_json={
        "matrix":cosine_sim.tolist(),
        "indices":indices.to_dict()
    }


    with open("./cosine_sim_matrix.json", "w") as outfile:
        json.dump(cosine_sim_json, outfile)




# Function that takes in user id as input and outputs most similar movies
def get_recommendations(userId,nUsersToRecommend=20,data_path='./generated-users2.csv'):

    # data_json=pd.read_json('./cosine_sim_matrix.json')
    with open('./cosine_sim_matrix.json', 'r') as openfile:
        data_json = json.load(openfile)

    # print(data_json)
    cosine_sim=data_json['matrix']
    indices=data_json['indices']

    generated_user_data = pd.read_csv(data_path)
    gdf=pd.DataFrame(generated_user_data)

    # Get the index of the user that matches the userId
    idx = indices[userId]

    # Get the pairwsie similarity scores of all the users with that user
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the users based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 'nUsersToRecommend' most similar users
    sim_scores = sim_scores[1:nUsersToRecommend+1]

    # Get the user indices
    user_indices = [i[0] for i in sim_scores]

    # Return the top 'nUsersToRecommend' most similar users
    print(gdf.iloc[user_indices])
    return (gdf.iloc[user_indices]).values.tolist()


if __name__=="__main__":
    create_similarity_matrix('./generated-users2.csv')
    # get_recommendations(2)
    get_recommendations('0LSd3oWzU7sVw4J104')