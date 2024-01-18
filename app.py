import numpy as np
import pandas as pd
import streamlit as st
from keras.models import model_from_json

from id_seq import get_seq, get_word_token, tokenization, pad


# returns the following datasets:
# true_dataset: a dataset that has the full list of interactions with true labels
# graph_dataset a dataset that has the predictions of the graph model
# for simplicity. the datasets should have the following columns:
# 'id1', 'id2', 'label'
def prepare_datasets():
    true_dataset = pd.read_csv('final_dataset.csv')
    graph_dataset = pd.read_csv('graph_dataset.csv')
    return true_dataset, graph_dataset


# prepares sequence models, that will be later used for live predictions
# will return 4 models
def prepare_models(num_models=1):
    model_arch = ['model_architecture.json']
    model_weights = ['model_weights.h5']
    models = []
    # Load the pre-trained machine learning model
    for i in range(0, num_models):
        with open(model_arch[i], 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(model_weights[i])
        models.append(model)

    return models


def preprocess_input(id, word_token):
    seq = get_seq(id)
    seq = tokenization(seq, word_token)
    seq = np.array(pad(seq))
    seq = np.reshape(seq, (1, 1000))  # Assuming 1 is the batch size
    seq = seq.astype('float32')
    return seq


# Find the interaction score from the sequence model prediction
def model_predict(input_data, model):
    y_pred = model.predict([input_data[0], input_data[1]])
    prediction_label = 'Positive' if y_pred > threshold else 'Negative'
    msg = prediction_label
    return msg


def dataset_predict(input_data, dataset, column):
    # find the interaction score from the true dataset
    interaction_entry = dataset[(dataset['id1'] == protein1) & (dataset['id2'] == protein2)]

    if not interaction_entry.empty:
        y_true = interaction_entry[column].values[0]
        true_label = 'Positive' if y_true > threshold else 'Negative'
        msg = true_label
    else:
        interaction_entry = dataset[(dataset['id2'] == protein1) & (dataset['id1'] == protein2)]
        if not interaction_entry.empty:
            y_true = interaction_entry[column].values[0]
            true_label = 'Positive' if y_true > threshold else 'Negative'
            msg = true_label
        else:
            msg = "Interaction pair not found"
    return msg


def string_input(input_data):
    model_predictions = []
    graph_predictions = []
    true_predictions = []
    for model in models:
        model_predictions.append(model_predict(input_data, model))
    graph_predictions.append(dataset_predict(input_data, graph_dataset, 'y_pred'))
    true_predictions.append(dataset_predict(input_data, true_dataset, 'y_true'))
    return true_predictions, model_predictions, graph_predictions


# edit later
def file_input(uploaded_file):
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        st.stop()

    with st.spinner("Predicting values..."):
        true_predictions = []
        model_predictions = []
        graph_predictions = []
        for index, row in uploaded_df.iterrows():
            input_data = [preprocess_input(row["id1"], word_token), preprocess_input(row["id2"], word_token)]
            y_true, y_pred_model, y_pred_graph = string_input(input_data)
            true_predictions.append(y_true)
            model_predictions.append(y_pred_model)
            graph_predictions.append(y_pred_graph)

    data = list(zip(true_predictions, model_predictions[0], graph_predictions))
    df = pd.DataFrame(data, columns=['y_true', 'y_pred_sequence', 'y_pred_graph'])
    st.write(df)
    csv_file_path = 'file_to_download.csv'

    df.to_csv(csv_file_path, index=False)
    show_download_button = True
    if show_download_button:
        st.download_button(
            label='Download CSV File',
            data=csv_file_path,
            key='download_csv_button',
            file_name='predictions.csv',
            mime='text/plain'
        )


#####################################################
fixed_length = 1000
word_token = get_word_token()
true_dataset, graph_dataset = prepare_datasets()
models = prepare_models()

# streamlit app
interaction_score = None
message = None
threshold = 0.5
# Streamlit app
logo_path = 'psut_red_logo.png'
st.set_page_config(page_title="PPI predictor", layout="wide", page_icon=logo_path)

# Display the sidebar with tabs
# st.sidebar.title("Navigation")
# selected_tab = st.sidebar.radio("Go to", ["Documentation", "Presentation"])

st.title('Protein Interaction Predictor')

# User input for protein sequences
protein1 = st.text_input('Enter Protein 1 Sequence ID:')
protein2 = st.text_input('Enter Protein 2 Sequence ID:')
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if st.button('Predict Interaction Score'):
    if uploaded_file is not None:
        file_input(uploaded_file)
    else:
        input_data = [preprocess_input(protein1, word_token), preprocess_input(protein2, word_token)]
        true_predictions, model_predictions, graph_predictions = string_input(input_data)
        data = list(zip(true_predictions, model_predictions[0], graph_predictions))
        df = pd.DataFrame(data, columns=['y_true', 'y_pred_model_0', 'y_pred_graph'])
        st.write(df)
