from flask import Flask, render_template, request, redirect, url_for
import clean
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/upload', methods=['POST'])
def upload():
    cleaned_data = clean.clean_dataset(request.files['file'])

    #return "File uploaded and cleaned successfully!"
    return redirect(url_for('output_html'))

@app.route('/output')
def output_html():
    conn = sqlite3.connect('output.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM output_table")
    data = cursor.fetchall()
    conn.close()

    return render_template('output.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)