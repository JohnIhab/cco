import requests
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np

app = FastAPI()

# Fetch data from the API
api_url = 'https://fast-plat1.vercel.app/meals/getUserRatting'
response = requests.get(api_url)

if response.status_code == 200:
    data = response.json()
    print("Data fetched successfully")
else:
    print(f"Failed to retrieve data: {response.status_code}")
    data = None

users_list = data['users'] if data else []
ratings = pd.DataFrame(users_list)
ratings_exploded = ratings.explode('ratings')
ratings_expanded = pd.json_normalize(ratings_exploded['ratings'])
ratings_expanded['_id'] = ratings_exploded['_id'].values
ratings_expanded['userName'] = ratings_exploded['userName'].values
ratings_expanded.columns = ['mealId', 'rating', 'user_id', 'userName']

file_path = 'food.csv'  # Update the file path if necessary
food_data = pd.read_csv(file_path)
ratings_expanded = ratings_expanded.merge(food_data[['index', 'name']], left_on='mealId', right_on='index')
ratings_expanded.drop('index', axis=1, inplace=True)

user_item_matrix = ratings_expanded.pivot(index='user_id', columns='mealId', values='rating').fillna(0)
user_item_sparse = sparse.csr_matrix(user_item_matrix.values)
item_similarity = cosine_similarity(user_item_sparse.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_recommendations(user_id: str, user_item_matrix: pd.DataFrame, item_similarity_df: pd.DataFrame, food_data: pd.DataFrame, top_n: int = 5, random_recommendations: bool = True):
    if user_id in user_item_matrix.index:
        user_ratings = user_item_matrix.loc[user_id]
        user_predicted_ratings = item_similarity_df.dot(user_ratings) / item_similarity_df.sum(axis=1)
        user_unrated_items = user_ratings[user_ratings == 0].index
        user_predicted_ratings = user_predicted_ratings[user_unrated_items]
        top_recommendations = user_predicted_ratings.nlargest(top_n).index
    else:
        top_recommendations = None

    if top_recommendations is None or random_recommendations:
        random_indices = np.random.choice(food_data['name'].size, size=top_n, replace=False)
        random_recommendations = food_data.iloc[random_indices]['index']
        top_recommendations = random_recommendations.tolist()

    recommendations_info = food_data.loc[food_data['index'].isin(top_recommendations), 
                                         ['name', 'time', 'no_of_ppl', 'prep_time', 'caloaries', 'ingredients', 'prep', 'img_link']]
    
    return recommendations_info

@app.get('/recommend')
async def recommend(user_id: str = Query(..., description="User ID for recommendations"),
                    top_n: int = Query(10, description="Number of recommendations to fetch")):
    try:
        top_n = int(top_n)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid value for top_n: {top_n}")
    
    recommendations = get_recommendations(user_id, user_item_matrix, item_similarity_df, food_data, top_n)
    
    return recommendations.to_dict(orient='records')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)
