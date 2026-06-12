import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    LabelEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("healthcare_data.csv")
df=df.dropna()

df["Test Results"] = df["Test Results"].replace({
    "Norm": "Normal"
})

X=df[["Age","Billing Amount","Admission Type","Medication","Test Results","Length of Stay","Gender","Blood Type"]]
y = df["Medical Condition"]

#Making pipeline
# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)

# preprocessing(scaling and encoding)
preprocessor = ColumnTransformer([
 #OneHotEncoding
    ('ohe',
     OneHotEncoder(handle_unknown='ignore'),
     ['Medication',"Gender","Blood Type"]),
 #OrdinalEncoding
    ('ord',
     OrdinalEncoder(categories=[
         ['Elective','Routine','Urgent','Emergency'],
         ['Normal','Inconclusive','Abnormal']
     ]),
     ['Admission Type','Test Results']),
  #Scaling
    ('scale',
     MinMaxScaler(),
     ['Age','Billing Amount','Length of Stay'])

], remainder='passthrough')
#Splits data into train and test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.2,random_state=5)



# pipeline using model kNN
from sklearn.ensemble import RandomForestClassifier

pipe1 = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=chi2, k=10)),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

from sklearn.metrics import accuracy_score


pipe1.fit(X_train, y_train)

pred = pipe1.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model and encoder
joblib.dump(pipe1, "disease_model.joblib")
joblib.dump(le, "label_encoder.joblib")

print("Model Saved Successfully!")