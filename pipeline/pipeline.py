#Importing all necessary libraries and modules
#Data processing
import pandas as pd

#Feature engineering 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Modelling
from catboost import CatBoostClassifier, Pool

#Metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

#Saving machine learning model into pickle format
import dill
from datetime import datetime

#Dropping unnecessary columns from the dataset
def drop_unimportant(df : pd.DataFrame) -> pd.DataFrame : 
    left_cols = ['Customer Type', 'Age', 'Type of Travel', 'Flight Distance', 'Inflight wifi service', 'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Class']
    df = df[left_cols]
    return df

#Encoding binary features
def encode_binary(df : pd.DataFrame) -> pd.DataFrame :
    df['Customer Type'] = df['Customer Type'].apply(lambda x : 1 if x == 'Loyal Customer' else 0)
    df['Type of Travel'] = df['Type of Travel'].apply(lambda x : 1 if x == 'Business travel' else 0)
    return df


def pipeline() -> None :
    print('Customer satisfaction predictor pipeline !')

    #Data loading
    df = pd.read_csv('./data/imputed_train_dataset.csv', sep = ',')

    # Preprocess the entire DataFrame first
    X = df.drop(columns = 'satisfaction')
    y = df[['satisfaction']]

    #Feature engineering
    ohe_cols = ['Class']
    std_scaler = ['Age', 'Flight Distance', 'Inflight wifi service', 'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']
    remaining_cols = ['Customer Type', 'Type of Travel']

    dropper_feature_changer = Pipeline(steps = [
        ('drop_cols', FunctionTransformer(drop_unimportant)),
        ('binary_encoding', FunctionTransformer(encode_binary))
    ])
    numerical_transformer = Pipeline(steps = [
        ('scaler', StandardScaler())
    ])
    ohe_transformation = Pipeline(steps = [
        ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    remaining_transformation = Pipeline(steps = [
        ('remaining_features', FunctionTransformer(lambda x : x))
    ])
    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, std_scaler),
        ('ohe_transformation', ohe_transformation, ohe_cols),
        ('remaining_features', remaining_transformation, remaining_cols)
    ])
    preprocessor = Pipeline(steps = [
        ('feature_change', dropper_feature_changer),
        ('column_transformer', column_transformer)
    ])

    cat_model = CatBoostClassifier(
        iterations = 1500,
        learning_rate = 0.01,
        depth = 10,
        eval_metric = 'AUC',
        random_seed = 1,
    )
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', cat_model)
    ])

    #Fitting perfect pipeline for whole dataset
    pipe.fit(X = X, y = y)

    pred = pipe.predict(X = X)

    train_auc_cat = roc_auc_score(y, pred)
    print(f"The ROC AUC score of CatBoostClassifier : {train_auc_cat}")

    model_filename = f'./models/customer_satisfaction.pkl'
    dill.dump({'model' : pipe,
    'metadata' :{
        'name' : 'Customer satisfaction predictor',
        'author' : 'Umidjon Sattorov',
        'version' : 1,
        'date' : datetime.now(),
        'type' : type(pipe.named_steps['classifier']).__name__,
        'roc_auc score' : roc_auc_score(y_true = y, y_score = pred)
    }
    }, open('./models/customer_satisfaction.pkl', 'wb'))

    print(f'Model is saved as {model_filename} in models directory')

if __name__ == '__main__':
    pipeline()
