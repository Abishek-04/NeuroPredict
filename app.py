from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import os

# ================= APP CONFIG =================
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///database.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "change-this-in-production")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ================= EMAIL CONFIG =================
app.config['MAIL_SERVER'] = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
app.config['MAIL_PORT'] = int(os.environ.get("MAIL_PORT", 587))
app.config['MAIL_USE_TLS'] = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get("MAIL_DEFAULT_SENDER", app.config['MAIL_USERNAME'])
app.config['MAIL_TIMEOUT'] = int(os.environ.get("MAIL_TIMEOUT", 8))
MAIL_ENABLED = bool(app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD'])

mail = Mail(app)
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# ================= LOAD ML MODELS =================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("xgb_classifier.pkl", "rb") as f:
    xgb_cls = pickle.load(f)

with open("xgb_regressor.pkl", "rb") as f:
    xgb_reg = pickle.load(f)

# ================= FEATURES =================
features = [
    'Chest Pain', 'Shortness of Breath', 'Irregular Heartbeat',
    'Fatigue & Weakness', 'Dizziness', 'Swelling (Edema)',
    'Pain in Neck/Jaw/Shoulder/Back', 'Excessive Sweating',
    'Persistent Cough', 'Nausea/Vomiting', 'High Blood Pressure',
    'Chest Discomfort (Activity)', 'Cold Hands/Feet',
    'Snoring/Sleep Apnea', 'Anxiety/Feeling of Doom', 'Age'
]
PREDICTIONS_FILE = os.environ.get("PREDICTIONS_FILE", os.path.join(BASE_DIR, "predictions.csv"))

# ================= USER MODEL =================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure tables exist when running under Gunicorn/Render (not only __main__).
with app.app_context():
    db.create_all()

# ================= FORMS =================
class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))

        hashed = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for("predict_form"))
        else:
            flash("Invalid login!", "danger")
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ================= RECOMMENDATIONS =================
def get_recommendations(risk_status):
    if risk_status == "High Risk":
        doctor = "Consult Cardiologist / Neurologist immediately."
        food = "Low salt, fruits, vegetables, oats, fish. Avoid fried food."
    else:
        doctor = "Regular health check with General Physician."
        food = "Balanced diet, fruits, hydration, avoid junk food."
    return doctor, food

# ================= CHART GENERATION =================
def generate_charts(input_data, risk_percentage):
    os.makedirs("static/charts", exist_ok=True)

    # =======================
    # 1️⃣ STROKE RISK BAR
    # =======================
    plt.figure(figsize=(6, 2))

    if risk_percentage < 40:
        color = "green"
        label = "Low Risk"
    elif risk_percentage < 70:
        color = "orange"
        label = "Moderate Risk"
    else:
        color = "red"
        label = "High Risk"

    plt.barh(["Stroke Risk"], [risk_percentage], color=color)
    plt.xlim(0, 100)
    plt.title(f"Stroke Risk: {risk_percentage}% ({label})", fontsize=14)
    plt.xlabel("Risk Percentage")

    risk_chart = "static/charts/risk_chart.png"
    plt.tight_layout()
    plt.savefig(risk_chart)
    plt.close()

    # =======================
    # 2️⃣ SYMPTOM SUMMARY
    # =======================
    symptom_values = list(input_data.values())
    present = sum(1 for v in symptom_values if v == 1)
    absent = len(symptom_values) - present

    plt.figure(figsize=(5, 4))
    plt.bar(
        ["Symptoms Present", "Symptoms Absent"],
        [present, absent]
    )

    plt.title("Patient Symptom Overview", fontsize=14)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    symptom_chart = "static/charts/symptom_chart.png"
    plt.tight_layout()
    plt.savefig(symptom_chart)
    plt.close()

    return risk_chart, symptom_chart


# ================= PREDICTION =================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_form():
    if request.method == "POST":
        try:
            patient_name = request.form.get("patient_name").replace(",", " ")
            nurse_name = request.form.get("nurse_name").replace(",", " ")
            nurse_email = request.form.get("nurse_email").replace(",", " ")

            input_data = {}
            for feature in features[:-1]:
                input_data[feature] = int(request.form.get(feature, 0))

            age = float(request.form.get("Age", 0))
            input_data["Age"] = scaler.transform([[age]])[0][0]

            input_df = pd.DataFrame([input_data])

            cls_result = xgb_cls.predict(input_df)[0]
            reg_result = xgb_reg.predict(input_df)[0]

            risk_status = "High Risk" if cls_result == 1 else "Low Risk"
            doctor_rec, food_rec = get_recommendations(risk_status)
            mail_status = {
                "kind": "secondary",
                "text": "Email not attempted."
            }

            save_results_to_csv(
                patient_name, nurse_name, nurse_email,
                input_data, risk_status, round(reg_result, 2)
            )

            risk_chart, symptom_chart = generate_charts(input_data, round(reg_result, 2))

            # EMAIL ALERT
            if cls_result == 1 and MAIL_ENABLED and nurse_email:
                msg = Message(
                    subject="🚨 Stroke Risk Alert",
                    recipients=[nurse_email],
                    body=f"""
Patient: {patient_name}
Nurse: {nurse_name}

Status: HIGH STROKE RISK
Risk: {round(reg_result,2)}%

Doctor: {doctor_rec}
Food: {food_rec}

Immediate action required.
"""
                )
                try:
                    # Never block core prediction flow if SMTP is slow/failing.
                    mail.send(msg)
                    mail_status = {
                        "kind": "success",
                        "text": f"Alert email sent to {nurse_email}."
                    }
                except Exception as mail_error:
                    app.logger.exception("Mail send failed: %s", mail_error)
                    mail_status = {
                        "kind": "warning",
                        "text": f"Prediction completed, but email failed to send to {nurse_email}."
                    }
            elif cls_result == 1 and not MAIL_ENABLED:
                mail_status = {
                    "kind": "warning",
                    "text": "Prediction completed, but email is not configured on server."
                }
            elif cls_result == 1 and not nurse_email:
                mail_status = {
                    "kind": "warning",
                    "text": "Prediction completed, but nurse email is missing."
                }
            else:
                mail_status = {
                    "kind": "info",
                    "text": "No email sent because this prediction is Low Risk."
                }

            return render_template(
                "result.html",
                patient_name=patient_name,
                risk_status=risk_status,
                risk_percentage=round(reg_result, 2),
                doctor_rec=doctor_rec,
                food_rec=food_rec,
                risk_chart=risk_chart,
                symptom_chart=symptom_chart,
                mail_status=mail_status
            )

        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for("predict_form"))

    return render_template("predict.html", features=features)

# ================= CSV SAVE =================
def save_results_to_csv(patient_name, nurse_name, nurse_email, input_data, risk_status, risk_percentage):
    filename = PREDICTIONS_FILE

    result = {
        "Patient Name": patient_name,
        "Nurse Name": nurse_name,
        "Nurse Email": nurse_email,
        **input_data,
        "Risk Status": risk_status,
        "Risk Percentage": risk_percentage
    }

    df = pd.DataFrame([result])
    parent_dir = os.path.dirname(filename)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# ================= DOWNLOAD =================
@app.route("/download")
@login_required
def download():
    if os.path.exists(PREDICTIONS_FILE):
        return send_file(PREDICTIONS_FILE, as_attachment=True)
    else:
        flash("No data found!", "warning")
        return redirect(url_for("predict_form"))

# ================= HISTORY =================
@app.route("/history")
@login_required
def history():
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE, on_bad_lines='skip')
        records = df.to_dict(orient="records")
    else:
        records = []
    return render_template("history.html", records=records)

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
