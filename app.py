from flask import Flask, render_template, request, redirect, url_for, session,flash
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask_bcrypt import Bcrypt, check_password_hash
import subprocess
from email.message import EmailMessage
import smtplib
from datetime import datetime
import random
from functools import wraps
from flask_bcrypt import generate_password_hash
from flask_socketio import SocketIO
import secrets
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sns
from io import BytesIO
import base64
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import plot
from flask import send_file
matplotlib.use('Agg')
from flask_wtf import FlaskForm, RecaptchaField
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from config import Config  # Importer la classe Config
import requests


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
# Middleware pour effacer les sessions à chaque nouvelle requête

app.config.from_object('config.Config')

# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
# 

# Initialize SQLAlchemy for database management
db = SQLAlchemy(app)

# Initialize Bcrypt for password hashing
bcrypt = Bcrypt(app)

socketio = SocketIO(app)

# Define a data model for users
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    nom = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    confirm = db.Column(db.String(255), nullable=False)
    otp = db.Column(db.String(6), nullable=False)
    otp_timestamp = db.Column(db.DateTime, nullable=False)
    verified = db.Column(db.Boolean, default=False)

    def __init__(self, nom, email, password, confirm):
        self.nom = nom
        self.email = email
        self.password = password
        self.confirm = confirm
        
class RegistrationForm(FlaskForm):
    nom = StringField('Nom', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    recaptcha = RecaptchaField()
    submit = SubmitField('Sign Up')
       
# Function to get users from the database
def get_utilisateurs():
    utilisateurs = User.query.all()
    return utilisateurs

@socketio.on('nouvel_utilisateur')
def handle_nouvel_utilisateur():
    utilisateurs = get_utilisateurs()
    socketio.emit('maj_utilisateurs', utilisateurs)

# Define the route for the home page
@app.route("/")
def index():
    return render_template('index.html')

# Define the route for the registration form
# Define the route for the registration form
@app.route("/inscription", methods=['GET', 'POST'])
def inscription():
    if request.method == 'POST':
        nom = request.form.get('nom')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        recaptcha_response = request.form.get('g-recaptcha-response')

        if not all([nom, email, password, confirm]):
            return "Veuillez remplir tous les champs du formulaire"

        # Vérifiez le reCAPTCHA
        recaptcha_secret = app.config['RECAPTCHA_PRIVATE_KEY']
        payload = {
            'secret': recaptcha_secret,
            'response': recaptcha_response
        }
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=payload)
        result = response.json()

        if not result.get('success'):
            return "Erreur de reCAPTCHA. Veuillez réessayer."

        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            if not bcrypt.check_password_hash(hashed_password, confirm):
                return "Les mots de passe ne correspondent pas"

            new_user = User(nom=nom, email=email, password=hashed_password, confirm=hashed_password)

            db.session.add(new_user)
            db.session.commit()

            flash('Compte créé avec succès!', 'success')
            return render_template('indexx.html')
        except IntegrityError as e:
            db.session.rollback()
            return f"Erreur lors de l'inscription : {str(e)}"
    return render_template('indexx.html')

# Function to generate OTP
def generate_otp():
    return str(random.randint(100000, 999999))
def send_test_email(email_address, otp):
    msg = EmailMessage()
    msg.set_content(f"""\
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    padding: 20px;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                p {{
                    color: #555;
                    margin-bottom: 20px;
                }}
                .footer {{
                    margin-top: 20px;
                    padding-top: 10px;
                    border-top: 1px solid #ddd;
                    text-align: center;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Verification de votre E-mail</h1>
                <p>L'OTP pour la verification de votre E-mail est : {otp}</p>
                <div class="footer">
                    <p>Copyright © 2024 FraudDetect. Tout droits reservé. </p>
                    
                </div>
            </div>
        </body>
    </html>
    """, subtype='html')
    msg["Subject"] = "Verification E-mail"
    msg["From"] = "habibzeinab81@gmail.com"
    msg["To"] = email_address

    # Update the SMTP server and port with your email provider's settings
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('habibzeinab81@gmail.com', 'yudb wipm xhiy yadw')
        smtp.send_message(msg)


@app.route('/connexion', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if not user or not bcrypt.check_password_hash(user.password, password):
            return 'E-mail ou Mot de passe incorrecte'

        otp = generate_otp()
        send_test_email(email, otp)

        user.otp = otp
        user.otp_timestamp = datetime.now()
        db.session.commit()

        # Définir la variable de session pour l'utilisateur connecté
        session['user_id'] = user.id

        return redirect(url_for('verify_otp', email=email))
    return render_template('indexx.html')


@app.route('/verification', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        
        # Recherche de l'utilisateur en fonction de l'OTP entré
        user = User.query.filter_by(otp=entered_otp).first()
        
        # Vérification si l'utilisateur existe et si l'OTP correspond
        if user and user.otp == entered_otp:
            # Vérification OTP réussie, réinitialisation de l'OTP et redirection
            user.otp = None
            user.otp_timestamp = None
            user.verified = True
            db.session.commit()
            session['user_id'] = user.id
            return redirect(url_for('user'))
        
        # Si l'OTP est incorrect ou si l'utilisateur n'existe pas
        return 'Utilisateur non existant ou code incorrect'
    
    # GET request, afficher la page de vérification
    return render_template('otp_verification.html')



@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        admin = Admin.query.filter_by(username=username).first()
        if not admin or admin.password != password:
   

            return 'Nom d\'utilisateur ou mot de passe incorrect'

        # Admin is logged in, redirect to the 'i' page
        return redirect(url_for('ii'))

    return render_template('admin_login.html')

@app.route('/index')
def i():
    return render_template('html/index.html')
@app.route('/UtilisateurIndex')
def user():
    if 'user_id' in session:
        user_id = session['user_id']
        return render_template('UtilisateurIndex.html')
    else:
        return redirect(url_for('login'))
  

@app.route('/ui-forms')
def iii():
    return render_template('html/ui-forms.html')
@app.route('/supprimer')
def iiii():
    return render_template('html/supprimer.html')
@app.route('/modifiermdp')
def iiiii():
    # Ici, vous devez définir la variable utilisateur avec une valeur appropriée
    # Si vous souhaitez simplement afficher le formulaire sans utilisateur, vous pouvez définir utilisateur à None
    utilisateur = None
    return render_template('html/modifiermdp.html', utilisateur=utilisateur)
@app.route('/consulter')
def ii():
    utilisateurs = get_utilisateurs()
    return render_template('html/consulter.html', utilisateurs=utilisateurs)


# Define a data model for admins
class Admin(db.Model):
    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = password

@app.route('/ajouter_utilisateur', methods=['GET', 'POST'])
def ajouter_utilisateur():
    if request.method == 'POST':
        nom = request.form.get('nom')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        if not all([nom, email, password, confirm]):
            return "Veuillez remplir tous les champs du formulaire"

        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            if not bcrypt.check_password_hash(hashed_password, confirm):
                return "Les mots de passe ne correspondent pas"

            new_user = User(nom=nom, email=email, password=hashed_password, confirm=hashed_password)
            db.session.add(new_user)
            db.session.commit()


            return redirect(url_for('ajouter_utilisateur'))  # Rediriger vers la page de consultation des utilisateurs
        except IntegrityError as e:
            db.session.rollback()
            return f"Erreur lors de l'ajout de l'utilisateur : {str(e)}"
    else:
        return render_template('html/ui-forms.html')  # Afficher le formulaire d'ajout d'utilisateur

# Route pour supprimer un utilisateur
@app.route('/supprimer_utilisateur', methods=['GET', 'POST'])
def supprimer_utilisateur_page():
    if request.method == 'POST':
        email = request.form.get('email')
        utilisateur = User.query.filter_by(email=email).first()
        if utilisateur:
            db.session.delete(utilisateur)
            db.session.commit()
            return redirect(url_for('supprimer_utilisateur_page')) 
        else:
            return "Utilisateur non trouvé."
    return render_template('html/supprimer.html')

@app.route('/modifier_utilisateur', methods=['POST'])
def modifier_utilisateur():
    if request.method == 'POST':
        email = request.form.get('email')
        
        # Vérifier si l'utilisateur avec cet email existe dans la base de données
        utilisateur = User.query.filter_by(email=email).first()
        if utilisateur is None:
            return "Utilisateur non trouvé"
        
        try:
            nouvel_email = request.form['email']
            nouveau_mot_de_passe = request.form['password']
            confirm_mot_de_passe = request.form['confirm']

            if nouveau_mot_de_passe != confirm_mot_de_passe:
                erreur_message = "Les mots de passe ne correspondent pas."
                return render_template('html/modifiermdp.html', utilisateur=utilisateur, erreur_message=erreur_message)

            utilisateur.email = nouvel_email
            utilisateur.password = generate_password_hash(nouveau_mot_de_passe)
            
            db.session.commit()
            
            return redirect(url_for('ii'))
        except Exception as e:
            db.session.rollback()
            erreur_message = f"Erreur lors de la modification de l'utilisateur : {str(e)}"
            return render_template('html/modifiermdp.html', utilisateur=utilisateur, erreur_message=erreur_message)

    return redirect(url_for('modifiermdp'))

@app.route('/result')
def result():
    data = pd.read_excel('result.xlsx')
    return render_template('File.html', table=data.to_html(classes='data', header="true"))

UPLOAD_FOLDER = 'Upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
uploaded_filename = None
@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_filename
    # Get the uploaded file
    file = request.files['file']
    
    # Save the file to the upload folder
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        uploaded_filename = filename
    # Check if the file is a CSV file
    if file.filename.endswith('.csv'):
        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(file)
            # Get the column names and data rows from the DataFrame
            columns = df.columns
            rows = df.values.tolist()
            # Render the template with the data
            return render_template('uploaded.html', columns=columns, rows=rows)
        except pd.errors.ParserError as e:
            return f"Error parsing CSV file: {e}"
    
    # Check if the file is an Excel file
    elif file.filename.endswith('.xlsx'):
        # Read the Excel file into a pandas DataFrame
        try:
            df = pd.read_excel(file)
            # Get the column names and data rows from the DataFrame
            columns = df.columns
            rows = df.values.tolist()
            # Render the template with the data
            return render_template('uploaded.html', columns=columns, rows=rows)
        except pd.errors.ParserError as e:
            return f"Error parsing Excel file: {e}"
    
    else:
        return "Please upload a CSV or Excel file"

    return render_template('UtilisateurIndex.html')


from sklearn.preprocessing import LabelEncoder

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocess():
    global uploaded_filename
    if uploaded_filename is None:
        return "No file uploaded"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    
    # Vérifier si c'est un fichier CSV ou Excel et lire en conséquence
    if uploaded_filename.endswith('.csv'):
        data = pd.read_csv(file_path, encoding='utf-8', delimiter=';', error_bad_lines=False)
    elif uploaded_filename.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        return "Unsupported file format"
    
    # Appliquer l'encodage des données
    label_encoder = LabelEncoder()
    data['Destination_encoded'] = label_encoder.fit_transform(data['Destination'])

    # Affichage des embeddings des destinations
    num_unique_destinations = len(label_encoder.classes_) 
    embedding_dim = 3
    embedding_matrix = np.random.randn(num_unique_destinations, embedding_dim)

    for i in range(num_unique_destinations):
        destination = label_encoder.inverse_transform([i])[0]
        embedding_vector = embedding_matrix[i]
        print(f"{destination}: {embedding_vector}")

    # Créer un dictionnaire pour mapper les destinations aux embeddings
    embedding_dict = {}
    for i in range(num_unique_destinations):
        destination = label_encoder.inverse_transform([i])[0]
        embedding_vector = embedding_matrix[i]
        embedding_dict[destination] = embedding_vector.tolist()

    # Remplacer les valeurs de la colonne "Destination" par les embeddings correspondants
    data['Destination'] = data['Destination'].map(embedding_dict)

    # Afficher le DataFrame complet
    encoded_data = data.head()
    # Utiliser to_html() avec des options de formatage
    encoded_data_html = encoded_data.to_html(index=False, classes='data', justify='left')

    # Passer encoded_data_html au template Jinja2
    return render_template('preprocess.html', encoded_data=encoded_data_html)
    

@app.route('/normalisation', methods=['GET','POST'])
def normalisation():
    global uploaded_filename
    if uploaded_filename is None:
        return "No file uploaded"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    
    # Lire le fichier CSV dans un DataFrame
    data = pd.read_excel(file_path)
    
    # Normaliser les données
    X_2 = data[['Appelant2', 'cout ', 'Appelé', 'Durée facturée']].values
    X_scaler2 = preprocessing.StandardScaler()
    X_scaler2.fit(X_2)
    X_standard2 = X_scaler2.transform(X_2)
    
    # Tracer le nuage de points avec les données normalisées
    plt.title("Nuage de points des données normalisées")
    plt.scatter(data['Appelant2'],data['cout '])
    plt.xlabel("appelant")
    plt.ylabel("cout")
    plt.ylabel("cout (normalisé)")
    
    # Sauvegarder la courbe dans un buffer mémoire
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Convertir les données normalisées en DataFrame pour l'affichage HTML
    normalized_data = pd.DataFrame(X_standard2, columns=['Appelant2', 'cout ', 'Appelé', 'Durée facturée'])
    normalized_data_html = normalized_data.head().to_html(index=False, classes='data', justify='left')
    
    # Retourner les données normalisées et la courbe à l'utilisateur
    return render_template('normalisation.html', normalized_data=normalized_data_html, plot_data=plot_data)

@app.route('/elbow_method', methods=['GET', 'POST'])
def elbow_method():
    global uploaded_filename
    if uploaded_filename is None:
        return "No file uploaded"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

    # Lire le fichier CSV dans un DataFrame
    data = pd.read_excel(file_path)

    # Normaliser les données
    X_2 = data[['Appelant2', 'cout ', 'Appelé', 'Durée facturée']].values
    X_scaler2 = preprocessing.StandardScaler()
    X_standard2 = X_scaler2.fit_transform(X_2)

    # Recherche de la valeur de K la plus favorable
    res=np.arange(9,dtype="double")
    sse={}
    for k in np.arange(9):
        km=KMeans(n_clusters=k+2)
        km.fit(X_standard2)
        res[k]=metrics.silhouette_score(X_standard2,km.labels_)
        sse[k]=km.inertia_
    print(sse)

    # Affichage du graphique Elbow avec la méthode d'Elbow
    plt.figure()
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.title("Méthode d'Elbow")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("SSE")
    plt.savefig('elbow_plot.png')  # Sauvegarder le graphique en tant qu'image

    # Incorporer l'image dans la page HTML
    with open("elbow_plot.png", "rb") as f:
        img = f.read()
        img_base64 = base64.b64encode(img).decode()

    plot_elbow = f'<img src="data:image/png;base64,{img_base64}">'

    return render_template('elbow.html', plot=plot_elbow)


@app.route('/silhouette_method', methods=['GET', 'POST'])
def silhouette_method():
    global uploaded_filename
    if uploaded_filename is None:
        return "No file uploaded"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

    # Lire le fichier CSV dans un DataFrame
    data = pd.read_excel(file_path)

    # Normaliser les données
    X_2 = data[['Appelant2', 'cout ', 'Appelé', 'Durée facturée']].values
    X_scaler2 = preprocessing.StandardScaler()
    X_standard2 = X_scaler2.fit_transform(X_2)

    # Recherche de la valeur de K la plus favorable
    res=np.arange(9,dtype="double")
    sse={}
    for k in np.arange(9):
        km=KMeans(n_clusters=k+2)
        km.fit(X_standard2)
        res[k]=metrics.silhouette_score(X_standard2,km.labels_)
        sse[k]=km.inertia_
    print(sse)

    # Créer le graphique de la silhouette
    plt.figure()
    plt.title("Silhouette Method")
    plt.xlabel("Number of Clusters")
    plt.plot(np.arange(2,11),res,'b:+')
    for i in np.arange(9):
        plt.text(i+2,res[i],i+2)
    plt.grid(True)
    plt.savefig("silhouette_plot.png")  # Enregistrer le graphique dans un fichier image
    

    # Incorporer l'image dans la page HTML
    with open("silhouette_plot.png", "rb") as f:
        img = f.read()
        img_base64 = base64.b64encode(img ).decode()

    plot_silhouette = f'<img src="data:image/png;base64,{img_base64}">'

    return render_template('silhouette.html', plot=plot_silhouette)








@app.route('/k_means', methods=['GET', 'POST'])
def k_means():
    if request.method == 'POST':
        try:
            # Récupérer la valeur de K saisie par l'utilisateur
            k_value = int(request.form.get('k_value'))

            # Vérifier si un fichier a été uploadé
            if 'uploaded_filename' not in globals() or uploaded_filename is None:
                return "No file uploaded"

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

            # Lire le fichier CSV dans un DataFrame
            data = pd.read_excel(file_path)

            # Normaliser les données
            X_2 = data[['Appelant2', 'cout ', 'Appelé', 'Durée facturée']].values
            X_scaler2 = preprocessing.StandardScaler()
            X_standard2 = X_scaler2.fit_transform(X_2)

            # Réalisation des kmeans avec le nombre de groupes choisi par l'utilisateur
            model_final2 = KMeans(n_clusters=k_value)
            model_final2.fit(X_standard2)

            # Prédire les clusters
            predicted_labels = model_final2.fit_predict(X_standard2)

            # Création du graphique de clustering
            plt.figure(figsize=(8, 6))
            for cluster_label in np.unique(predicted_labels):
                cluster_data = X_standard2[predicted_labels == cluster_label]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 3], label=f'Cluster {cluster_label}')

            plt.title('Classification avec K-means')
            plt.xlabel('Appelant')
            plt.ylabel('Coût')
            plt.legend()

            # Sauvegarder le graphique en tant qu'image
            plt.savefig('static/kmeans_plot.png')
            plt.close()

            # Convertir l'image en base64
            with open("static/kmeans_plot.png", "rb") as f:
                img = f.read()
                img_base64 = base64.b64encode(img).decode()

            # Renvoyer le code HTML avec l'image en base64
            img_tag = f'<img src="data:image/png;base64,{img_base64}">'

            # Enregistrer les anomalies dans un fichier Excel dans le dossier static
            fraudekm = data[["Appelant", "Appelé", "Durée facturée", "cout ", "Destination"]]
            fraudekm['class'] = model_final2.labels_
            fraudekm = fraudekm[((fraudekm['class'] == 1) | (fraudekm['class'] == 2)) | (fraudekm['class'] == 4)]
            fraudekm.to_excel(os.path.join('static', 'fichier-anomalie.xlsx'), index=False)

            # Afficher la page K_means.html
            return render_template('K_means.html', img=img_tag)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return "Invalid request"


@app.route('/dbscan', methods=['GET', 'POST'])
def dbscan():
    if request.method == 'POST':
        try:
            # Récupérer le fichier uploadé
            global uploaded_filename
            if uploaded_filename is None:
                return "No file uploaded"

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

            # Lire le fichier CSV dans un DataFrame
            data = pd.read_excel(file_path)

            # Normaliser les données
            X_2 = data[['Appelant2', 'cout ', 'Appelé', 'Durée facturée']].values
            X_scaler2 = preprocessing.StandardScaler()
            X_standard2 = X_scaler2.fit_transform(X_2)

            # DBSCAN
            dbscan = DBSCAN(eps=0.4, min_samples=5)
            dbscan.fit(X_standard2)

            # Créer le graphique de classification
            plt.figure(figsize=(8, 6))
            for i in range(data.shape[0]):
                if dbscan.labels_[i] == -1:
                    plt.scatter(data.iloc[i, 0], data.iloc[i, 3], color='orange')
                elif dbscan.labels_[i] == 0:
                    plt.scatter(data.iloc[i, 0], data.iloc[i, 3], color='blue')
                elif dbscan.labels_[i] == 1:
                    plt.scatter(data.iloc[i, 0], data.iloc[i, 3], color='yellow')
                # Ajoutez d'autres couleurs pour les clusters supplémentaires si nécessaire

            plt.title('Classification avec DBSCAN')
            plt.xlabel('Appelant')
            plt.ylabel('Coût')
            plt.legend()

            # Sauvegarder le graphique en tant qu'image
            plt.savefig('static/dbscan_plot.png')
            plt.close()

            # Enregistrer les anomalies dans un fichier CSV
            fraudedbscan = data[["Appelant", "Appelé", "Durée facturée", "cout ", "Destination"]]
            fraudedbscan['class'] = dbscan.labels_
            fraudedbscan = fraudedbscan[(fraudedbscan['class'] == -1) | (fraudedbscan['class'] == 1) | (fraudedbscan['class'] == 4)]
            fraudedbscan.to_csv('static/fraudedbscananomalie.csv', index=False)

            # Convertir l'image en base64
            with open("static/dbscan_plot.png", "rb") as f:
                img = f.read()
                img_base64 = base64.b64encode(img).decode()

            # Renvoyer le code HTML avec l'image en base64
            img_tag = f'<img src="data:image/png;base64,{img_base64}">'
            return render_template('dbscan.html', img=img_tag)
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return "Invalid request"



from flask import send_file

@app.route('/download_anomalies', methods=['GET', 'POST'])
def download_anomalies():
    if request.method == 'POST':
        try:
            # Charger les fichiers d'anomalies
            df_km = pd.read_excel('static/fichier-anomalie.xlsx')
            df_dbscan = pd.read_csv('static/fraudedbscananomalie.csv')

            # Fusionner les deux DataFrames en un seul en supprimant les doublons
            df_merged = pd.concat([df_km, df_dbscan]).drop_duplicates().reset_index(drop=True)

            # Enregistrer le DataFrame fusionné dans un fichier Excel
            file_path = 'static/fraudeanomaliegeneral.xlsx'
            df_merged.to_excel(file_path, index=False)

            # Retourner le fichier Excel en téléchargement
            return send_file(file_path, as_attachment=True)

        except Exception as e:
            return f"An error occurred: {str(e)}"
    else:
        return "Method Not Allowed"




@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))


@app.route('/effacer_sessions', methods=['GET'])
def effacer_sessions():
    session.pop('suivant', None)
    return '', 204  # Renvoie une réponse vide avec le code HTTP 204 (No Content)



# Close the database connection when the app is destroyed
@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()

if __name__ == '__main__':
    socketio.run(app, debug=True)
