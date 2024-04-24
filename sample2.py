from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

app = Flask(__name__)

# Load the trained model and tokenizer from the directory
model_path = "detector1/checkpoint-6246"  # Directory path containing the model files
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the mapping of class indices to labels
class_labels = {0: 'Positive', 1: 'Neutral', 2: 'Negative', 3: 'Irrelevant'}

@app.route('/quiz1')
def quiz1():
    return render_template('quiz1.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    selected_answers = []

    # Loop through form items to extract selected answers
    inputs_for_prediction = {}
    for key, value in request.form.items():
        if key != 'submit':
            inputs_for_prediction[key] = value
        if request.form.getlist(key) and key != 'submit':  # Check if it's a checkbox and it's checked
            selected_answers.extend(request.form.getlist(key))
        elif key != 'submit':  # Check if it's a radio button
            selected_answers.append(value)

    # Concatenate the selected answers into a single string separated by commas
    input_text = ', '.join(selected_answers)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = class_labels[logits.argmax(axis=-1).item()]

    # Prepare data for Excel
    data = {}
    for key, value in inputs_for_prediction.items():
        data[key] = [value]
    data['Predicted Label'] = [predicted_label]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define Excel file path
    excel_file_path = f'quiz1_results.xlsx'

     # Load existing data from Excel file, if it exists
    try:
        existing_df = pd.read_excel(excel_file_path)
        df = pd.concat([existing_df, pd.DataFrame(data)])
    except FileNotFoundError:
        df = pd.DataFrame(data)

    # Move the predicted label column to the last position
    df = df[[col for col in df.columns if col != 'Predicted Label'] + ['Predicted Label']]

    # Write DataFrame to Excel
    df.to_excel(excel_file_path, index=False)

    # Render the result template with the predicted label and inputs for prediction
    return render_template('result.html', predicted_label=predicted_label, inputs=inputs_for_prediction)

@app.route('/quiz2')
def quiz2():
    return render_template('quiz2.html')

@app.route('/predict2', methods=['POST'])
def predict2():
    selected_answers = []

    # Loop through form items to extract selected answers
    inputs_for_prediction = {}
    for key, value in request.form.items():
        if key != 'submit':
            inputs_for_prediction[key] = value
        if request.form.getlist(key) and key != 'submit':  # Check if it's a checkbox and it's checked
            selected_answers.extend(request.form.getlist(key))
        elif key != 'submit':  # Check if it's a radio button
            selected_answers.append(value)

    # Concatenate the selected answers into a single string separated by commas
    input_text = ', '.join(selected_answers)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = class_labels[logits.argmax(axis=-1).item()]

    # Prepare data for Excel
    data = {}
    for key, value in inputs_for_prediction.items():
        data[key] = [value]
    data['Predicted Label'] = [predicted_label]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define Excel file path
    excel_file_path = f'quiz2_results.xlsx'

     # Load existing data from Excel file, if it exists
    try:
        existing_df = pd.read_excel(excel_file_path)
        df = pd.concat([existing_df, pd.DataFrame(data)])
    except FileNotFoundError:
        df = pd.DataFrame(data)

    # Move the predicted label column to the last position
    df = df[[col for col in df.columns if col != 'Predicted Label'] + ['Predicted Label']]

    # Write DataFrame to Excel
    df.to_excel(excel_file_path, index=False)

    # Render the result template with the predicted label and inputs for prediction
    return render_template('result.html', predicted_label=predicted_label, inputs=inputs_for_prediction)

@app.route('/quiz3')
def quiz3():
    return render_template('quiz3.html')

@app.route('/predict3', methods=['POST'])
def predict3():
    selected_answers = []

    # Loop through form items to extract selected answers
    inputs_for_prediction = {}
    for key, value in request.form.items():
        if key != 'submit':
            inputs_for_prediction[key] = value
        if request.form.getlist(key) and key != 'submit':  # Check if it's a checkbox and it's checked
            selected_answers.extend(request.form.getlist(key))
        elif key != 'submit':  # Check if it's a radio button
            selected_answers.append(value)

    # Concatenate the selected answers into a single string separated by commas
    input_text = ', '.join(selected_answers)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = class_labels[logits.argmax(axis=-1).item()]

    # Prepare data for Excel
    data = {}
    for key, value in inputs_for_prediction.items():
        data[key] = [value]
    data['Predicted Label'] = [predicted_label]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define Excel file path
    excel_file_path = f'quiz3_results.xlsx'

     # Load existing data from Excel file, if it exists
    try:
        existing_df = pd.read_excel(excel_file_path)
        df = pd.concat([existing_df, pd.DataFrame(data)])
    except FileNotFoundError:
        df = pd.DataFrame(data)

    # Move the predicted label column to the last position
    df = df[[col for col in df.columns if col != 'Predicted Label'] + ['Predicted Label']]

    # Write DataFrame to Excel
    df.to_excel(excel_file_path, index=False)

    # Render the result template with the predicted label and inputs for prediction
    return render_template('result.html', predicted_label=predicted_label, inputs=inputs_for_prediction)

@app.route('/quiz4')
def quiz4():
    return render_template('quiz4.html')

@app.route('/predict4', methods=['POST'])
def predict4():
    selected_answers = []

    # Loop through form items to extract selected answers
    inputs_for_prediction = {}
    for key, value in request.form.items():
        if key != 'submit':
            inputs_for_prediction[key] = value
        if request.form.getlist(key) and key != 'submit':  # Check if it's a checkbox and it's checked
            selected_answers.extend(request.form.getlist(key))
        elif key != 'submit':  # Check if it's a radio button
            selected_answers.append(value)

    # Concatenate the selected answers into a single string separated by commas
    input_text = ', '.join(selected_answers)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get the predicted label
    predicted_label = class_labels[logits.argmax(axis=-1).item()]

    # Prepare data for Excel
    data = {}
    for key, value in inputs_for_prediction.items():
        data[key] = [value]
    data['Predicted Label'] = [predicted_label]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define Excel file path
    excel_file_path = f'quiz4_results.xlsx'

     # Load existing data from Excel file, if it exists
    try:
        existing_df = pd.read_excel(excel_file_path)
        df = pd.concat([existing_df, pd.DataFrame(data)])
    except FileNotFoundError:
        df = pd.DataFrame(data)

    # Move the predicted label column to the last position
    df = df[[col for col in df.columns if col != 'Predicted Label'] + ['Predicted Label']]

    # Write DataFrame to Excel
    df.to_excel(excel_file_path, index=False)

    # Render the result template with the predicted label and inputs for prediction
    return render_template('result.html', predicted_label=predicted_label, inputs=inputs_for_prediction)



if __name__ == '__main__':
    app.run(debug=True)
