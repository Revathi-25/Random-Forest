import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model

def load_vec(path):
    with open(path, 'rb') as file:
        vec = pickle.load(file)
        return vec
    
if __name__ == '__main__':

    path_to_model = 'backend/models/random_forest_model.pkl'
    path_to_vec = 'backend/models/tfidf.pkl'
    model = load_model(path_to_model)
    vec = load_vec(path_to_vec)
    print(model)
    print(vec.get_feature_names_out())

