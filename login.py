from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import hashlib

app = Flask(__name__)

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Route to display the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        # Check if the user exists in the Excel file
        df = pd.read_excel('user_credentials.xlsx')
        if (df['Username'] == username) & (df['Password'] == password).any():
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', message='Invalid username or password')
    else:
        return render_template('login.html', message='')

# Route to display the register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        # Check if the username already exists
        df = pd.read_excel('user_credentials.xlsx')
        if (df['Username'] == username).any():
            return render_template('register.html', message='Username already exists')
        else:
            # Add the new user to the Excel file
            new_user = pd.DataFrame({'Username': [username], 'Password': [password]})
            df = pd.concat([df, new_user])
            df.to_excel('user_credentials.xlsx', index=False)
            return redirect(url_for('login'))
    else:
        return render_template('register.html', message='')

# Route for the dashboard page (after successful login)
@app.route('/dashboard')
def dashboard():
    return 'Welcome to the dashboard'

if __name__ == '__main__':
    app.run(debug=True)
