import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
 
# gdf['Interests'] = gdf['Interests'].apply(literal_eval)
# # print(gdf['Id'].head())

# def create_soup(x):
#     return str.lower(' '.join(x['Interests']) +' '+' '.join(x['Interests']) + ' ' +x['City']+' '+x['Situation']+' '+x['Noise']+' '+x['SleepSchedule']+' '+x['Personnality'])

def create_soup(x):
    x['Property Type Preference'] = str(x['Property Type Preference']) if str(x['Property Type Preference'])!='nan' else ""
    x['Smoker']= "SmokeAllow" if x['Smoker']==1 else "SmokeForbid"
    x['Pets']= "PetsAllow" if x['Pets']==1 else "PetsForbid"

    return str.lower(x['Interests'].replace(";"," ")+' '+x['Interests'].replace(";"," ")+' '+x['Smoker']+' '+x['Pets']+' ' +x['Home City']+' '+x['Situation']+' '+x['Noise']+' '+x['School']+' '+x['Work Place']+' '+x['Sleep Schedule']+' '+x['Personality Type']+' '+x['Property Type Preference'])

def create_similarity_matrix_male(data_path='./user-data-comma.csv'):
    generated_user_data = pd.read_csv(data_path)
    gdf=pd.DataFrame(generated_user_data)
    gdfMale=gdf[gdf['Sexe']=='Male']

    gdfMale['soup'] = gdfMale.apply(create_soup, axis=1)
    # print(gdfMale.head(2))

    count = CountVectorizer(stop_words='english')
    count_matrix_male = count.fit_transform(gdfMale['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim_male = cosine_similarity(count_matrix_male, count_matrix_male)

    gdfMale.reset_index(drop=True, inplace=True)

    #Construct a reverse map of indices and userIds
    indices = pd.Series(gdfMale.index,index=gdfMale['Email'].values).drop_duplicates()
    # print(indices)
    cosine_sim_json_male={
        "matrix":cosine_sim_male.tolist(),
        "indices":indices.to_dict()
    }

    print("calculated matrix male")
    with open("./cosine_sim_matrix_male.json", "w") as outfile:
        json.dump(cosine_sim_json_male, outfile)
 

def create_similarity_matrix_female(data_path='./user-data-comma.csv'):
    generated_user_data = pd.read_csv(data_path)
    gdf=pd.DataFrame(generated_user_data)
    gdfFemale=gdf[gdf['Sexe']=='Female']
    
    gdfFemale['soup'] = gdfFemale.apply(create_soup, axis=1)
    # print(gdfFemale.head(2))

    count = CountVectorizer(stop_words='english')
    count_matrix_female = count.fit_transform(gdfFemale['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim_female = cosine_similarity(count_matrix_female, count_matrix_female)

    gdfFemale.reset_index(drop=True, inplace=True)

    #Construct a reverse map of indices and userIds
    indices = pd.Series(gdfFemale.index,index=gdfFemale['Email'].values).drop_duplicates()
    # print(indices)
    cosine_sim_json_female={
        "matrix":cosine_sim_female.tolist(),
        "indices":indices.to_dict()
    }

    print("calculated matrix female")
    with open("./cosine_sim_matrix_female.json", "w") as outfile:
        json.dump(cosine_sim_json_female, outfile)




# Function that takes in user id as input and outputs most similar movies
def get_recommendations(userEmail,nUsersToRecommend=20,data_path='./user-data-comma.csv'):

    generated_user_data = pd.read_csv(data_path)
    gdf=pd.DataFrame(generated_user_data)
    # gdf.head()
    user=gdf[gdf['Email']==userEmail]
    print(user)
    print(user['Sexe'].values[0])

    if user['Sexe'].values[0]=='Male':
        gdf=gdf[gdf['Sexe']=='Male']
        # data_json=pd.read_json('./cosine_sim_matrix.json')
        with open('./cosine_sim_matrix_male.json', 'r') as openfile:
            data_json = json.load(openfile)
    else :
        gdf=gdf[gdf['Sexe']=='Female']
        with open('./cosine_sim_matrix_female.json', 'r') as openfile:
            data_json = json.load(openfile)


    # print(data_json)
    cosine_sim=data_json['matrix']
    indices=data_json['indices']
    # print(indices)


    # Get the index of the user that matches the userId
    idx = indices[userEmail]
    print(idx)

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
    create_similarity_matrix_male('./user-data-comma.csv')
    create_similarity_matrix_female('./user-data-comma.csv')
    get_recommendations('jesse.lawhorn@gmail.com')