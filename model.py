import pickle
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('maintenance_prediction.csv')

X = df.drop(['target', 'failure_type'], axis= 1).copy()
y = df['target'].copy()

X_encoded = pd.get_dummies(X, columns=['type'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, 
                                                    random_state= 42, stratify=y)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

os = RandomOverSampler(0.6)
X_train_os, y_train_os = os.fit_resample(X_train, y_train)

svm = SVC()
svm = SVC(C=1)
svm.fit(X_train_os, y_train_os)

pickle.dump(svm, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

