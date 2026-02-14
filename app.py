import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'prev_cfg' not in st.session_state:
    st.session_state.prev_cfg = (15, 0)

def check_my_dataset(df:pd.DataFrame, target_col= None):
    if (target_col == None):
        target_col = df.columns[-1]
        y = df[target_col]

    if target_col not in df.columns :
        return False
    if df.shape[1] < 19:
        return False
    if df.shape[0] < 500:
        return False
    if df.select_dtypes(include=[np.number]).shape[1] != 19:
        return False
     
    n_classes = y.nunique(dropna=True)
     
    if n_classes != 4:
        return False

    # check for NULL values in columns
    round(100*(df.isnull().sum()/len(df)), 2).sort_values(ascending=False)

    #check the NULL values in rows 
    round((df.isnull().sum(axis=1)/len(df))*100,2).sort_values(ascending=False)
    # Drop rows with NULL values
    df = df.dropna()

    # Data seems to be correct lets show the data 
    st.markdown('<h2 class="sub-header"> Dataset Uploaded</h2>', unsafe_allow_html=True)
    num_samples, num_features, num_numeric, num_cat = st.columns(4)

    with num_samples:
        st.metric("Samples", df.shape[0])
    with num_features:
        st.metric ("Features", df.shape[1]-1)
    with num_numeric:
        st.metric("Numeric Features", df.select_dtypes(include=[np.number]).shape[1])
    with num_cat:
        st.metric("Categorical Features", df.select_dtypes(include=['object']).shape[1])

    with st.expander("View Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        st.write(df.describe(include='all').T)
    st.markdown("---")
    st.markdown("#### Training Parameters")
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    df[target_col] = y
    st.session_state.df=df
    st.session_state.data_loaded = True 
    
    return True


IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD", "").lower() == "true"
MODELS_DIR = "model"
os.makedirs(MODELS_DIR, exist_ok=True)

def add_performance_row (table, algo_name, y_true, y_pred, y_proba, average="binary"):
    table[algo_name] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba, average=average, multi_class='ovr') if y_proba is not None else None,
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

performance_table = {}
preds_by_model = {}
st.markdown("""
<style>
/* Blue config panel */
.config-panel {
  background: #0B5ED7;
  color: white;
  padding: 1rem 1rem 0.5rem 1rem;
  border-radius: 12px;
}

/* Make labels white */
.config-panel label,
.config-panel p,
.config-panel span,
.config-panel div {
  color: white !important;
}

/* Inputs background + borders (number input, selectbox, etc.) */
.config-panel input,
.config-panel textarea {
  color: white !important;
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
}

/* Selectbox "button" area */
.config-panel [data-baseweb="select"] > div {
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
  color: white !important;
}

/* Slider: try to force readable text */
.config-panel [data-testid="stSlider"] * {
  color: white !important;
}

div.stButton > button {
    background-color: #1f77ff;
    color: white;
    font-size: 20px;
    padding: 0.75em 2em;
    border-radius: 8px;
    border: none;
}
div.stButton > button:hover {
    background-color: #1558b0;
    color: white;
}
</style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(layout="wide")

st.title("Machine Learning Classification Model Dashboard")

st.write("Environment variables snapshot:")
st.write(dict(os.environ))


st.markdown("Download the test dataset.")
RAW_URL="https://raw.githubusercontent.com/tusharmulkar/ml-model-app/main/vehicle.csv"

if st.button("Get Sample Dataset"):
    s_df = pd.read_csv(RAW_URL)
    st.download_button(
        "Download CSV",
        s_df.to_csv(index=False).encode("utf-8"),
        "vehicle_dataset.csv",
        "text/csv"
    )

st.markdown("Upload your dataset, configure model parameters, and compare classification algorithms.")
# File upload
dataset_file = None
dataset_file = st.file_uploader(
    "Upload your dataset (CSV or Excel)",
    type=['csv', 'xlsx', 'xls'],
    help=""
)
st.markdown("---")

if dataset_file is not None:
    df = pd.read_csv(dataset_file)

if dataset_file != None and check_my_dataset(df) == True:
    
    st.markdown('### Configuration Panel <div class="config-panel">', unsafe_allow_html=True)
    c1, c2 = st. columns(2, vertical_alignment="center")

    with c1:
        ml_models_list = [
            'Logistic Regression',
            'Decision Tree',
            'KNN',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ]

        models_to_run = st.multiselect(
            "Select models to train:",
            ml_models_list,
            default=ml_models_list,
            help="Select one or more models for comparison"
        )
        st.session_state.models_to_run = models_to_run

    with c2:
        test_split = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=15,
            step=5,
            help="train test split"
        )

        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=1000,
            value=0,
            help="Set random seed for reproducibility"
        )
    
        st.session_state.test_size = test_split
        st.session_state.random_seed = random_seed

    st.markdown("---")
  
    if st.session_state.data_loaded :
        df = st.session_state.df

        y = df.iloc[:, -1].copy()
        X = df.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=st.session_state.test_size/100,
            random_state=st.session_state.random_seed,
            stratify=y
        )   

        labels = np.unique(y_test)  # consistent label order for all models
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        model_trainer = None
        if st.button("Train & Evaluate"):
            prev = st.session_state.get("prev_cfg", (15,0))
            curr = (st.session_state.test_size, st.session_state.random_seed)

            if curr != prev:
                cfg_changed = True
                st.session_state["prev_cfg"] = curr
            else:
                cfg_changed = False
            
            with st.spinner("Training models..."):
                for i, model_name in enumerate(st.session_state.models_to_run):
                    model_path = os.path.join(MODELS_DIR, f"vehicle_{model_name}.joblib")

                    if not IS_STREAMLIT_CLOUD and (not os.path.exists(model_path) or cfg_changed):
                        st.info(f"Model {model_name} not found or config chaged. Training new model.")
                        st.write(f"Training model: {model_name}")
                        match model_name:
                            case 'Logistic Regression':
                                model_trainer = LogisticRegression(max_iter=1000,
                                                                C=0.01, 
                                                                penalty='l2', 
                                                                solver='lbfgs', 
                                                                random_state=st.session_state.random_seed)
                            case 'Decision Tree':
                                model_trainer = DecisionTreeClassifier(random_state=st.session_state.random_seed)

                            case 'KNN':
                                model_trainer = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
                            case 'Naive Bayes':
                                model_trainer = GaussianNB()
                            case 'Random Forest':
                                model_trainer = RandomForestClassifier(n_estimators=300, 
                                                                    max_depth=10, 
                                                                    min_samples_leaf=3,
                                                                    random_state=st.session_state.random_seed)
                            case 'XGBoost':
                                if IS_STREAMLIT_CLOUD: 
                                    st.write("Not training XGboost on cloud using pretrained model if exits")
                                    if os.path.exists(model_path) :
                                        model_trainer = joblib.load(model_path)
                                    else:
                                        model_trainer = None
                                else:
                                    # if os.path.exists(model_path) :
                                    #     model_trainer = joblib.load(model_path)
                                    # else:
                                    model_trainer = XGBClassifier(n_estimators=300, 
                                                                 max_depth=4, 
                                                                 learning_rate=0.05,
                                                                 subsample=0.8,
                                                                 colsample_bytree=0.8,
                                                                 objective='multi:softprob',
                                                                 num_class=4,
                                                                 eval_metric='mlogloss',
                                                                 random_state=st.session_state.random_seed)
                            case _:
                                st.error(f"Unknown model: {model_name}")
                                continue
                        if model_trainer is not None:    
                            model_trainer.fit(scaled_X_train, y_train)
                            joblib.dump(model_trainer, model_path)
                            st.success(f"Model {model_name} trained successfully!")

                    else:
                        st.write(f"Model {model_name} already exists. Loading existing model.")
                        if os.path.exists(model_path):
                            model_trainer = joblib.load(model_path)

                    if model_trainer is not None:
                        y_pred = model_trainer.predict(scaled_X_test)
                        y_proba = model_trainer.predict_proba(scaled_X_test)
                        add_performance_row(performance_table, model_name, y_test, y_pred, y_proba, average="macro")
                        preds_by_model[model_name] = y_pred
                        
            st.subheader("Model Performance")
            perf_df = pd.DataFrame.from_dict(performance_table, orient="index")
            perf_df = perf_df.reset_index().rename(columns={"index": "model"})
            st.dataframe(perf_df, use_container_width=True)

            st.subheader("Confusion Matrices")

            tabs = st.tabs(list(preds_by_model.keys()))
            for tab, name in zip(tabs, preds_by_model.keys()):
                with tab:
                    cm = confusion_matrix(y_test, preds_by_model[name], labels=labels)

                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                    disp.plot(ax=ax, cmap=None, values_format="d")  # don't set cmap to respect default
                    ax.set_title(name)
                    st.pyplot(fig)

            cfg_changed = False
