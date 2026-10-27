import streamlit as st
import joblib
import sqlite3
import hashlib
import hmac
import secrets
import re
from pathlib import Path
from datetime import datetime

# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
DB_PATH = BASE_DIR / "users.db"

# ==================================================
# 2. CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }

    h1, h2, h3 {
        color: #172554;
    }

    .login-title {
        text-align: center;
        font-size: 29px;
        font-weight: 750;
        color: #172554;
        margin-bottom: 5px;
    }

    .login-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 19px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
    }

    .metric-value {
        color: #172554;
        font-size: 27px;
        font-weight: 700;
        margin-top: 6px;
    }

    .result-box {
        padding: 18px;
        border-radius: 10px;
        margin-top: 12px;
        font-size: 18px;
        font-weight: 650;
    }

    .real-result {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .fake-result {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2563eb;
        border: none;
        border-radius: 8px;
    }

    .small-note {
        color: #64748b;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. DATABASE SETUP
# ==================================================

def get_connection():
    return sqlite3.connect(DB_PATH, timeout=10)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                article TEXT NOT NULL,
                prediction TEXT NOT NULL,
                checked_at TEXT NOT NULL
            )
        """)


init_database()

# ==================================================
# 4. PASSWORD SECURITY
# ==================================================

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        300000
    ).hex()


def register_user(username, password):
    username = username.strip()

    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return False, (
            "Username must be 3–30 characters "
            "and contain only letters, numbers or underscores."
        )

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO users
                (username, salt, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                username,
                salt,
                password_hash,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

        return True, "Registration successful!"

    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."


def verify_user(username, password):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT salt, password_hash
            FROM users
            WHERE username = ?
        """, (username.strip(),)).fetchone()

    if row is None:
        return False

    salt, stored_hash = row
    entered_hash = hash_password(password, salt)

    return hmac.compare_digest(entered_hash, stored_hash)


# ==================================================
# 5. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Place fake_news_model.pkl beside app.py."
        )

    if not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Vectorizer not found at: {VECTORIZER_PATH}\n"
            "Place tfidf_vectorizer.pkl beside app.py."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# ==================================================
# 6. TEXT PREPROCESSING AND PREDICTION
# ==================================================

def prepare_text(text):
    """
    Keep this preprocessing consistent with the
    preprocessing used when training your model.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_prediction_label(label):
    """
    Expected encoding:
    0 = Fake News
    1 = Real News

    String class labels are also supported.
    Verify numeric encoding against train_model.py.
    """
    value = str(label).strip().lower()

    if "fake" in value:
        return "Fake News"

    if "real" in value or "true" in value:
        return "Real News"

    if value in ("0", "0.0"):
        return "Fake News"

    if value in ("1", "1.0"):
        return "Real News"

    raise ValueError(
        f"Unknown model class: {label}. "
        "Check the labels used during training."
    )


def predict_news(article):
    model, vectorizer = load_model()

    cleaned_article = prepare_text(article)

    if not cleaned_article:
        raise ValueError(
            "The article contains no usable text. "
            "Please enter a longer news article."
        )

    # TF-IDF converts text into numerical features.
    features = vectorizer.transform([cleaned_article])

    # Predict the class.
    raw_prediction = model.predict(features)[0]
    prediction = get_prediction_label(raw_prediction)

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities)) * 100

    return prediction, confidence


# ==================================================
# 7. SAVE AND RETRIEVE PREDICTION HISTORY
# ==================================================

def save_history(username, article, prediction):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO history
            (username, article, prediction, checked_at)
            VALUES (?, ?, ?, ?)
        """, (
            username,
            article,
            prediction,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


def get_history(username):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, article, prediction, checked_at
            FROM history
            WHERE username = ?
            ORDER BY id DESC
        """, (username,)).fetchall()

    return rows


# ==================================================
# 8. SESSION STATE
# ==================================================

if "username" not in st.session_state:
    st.session_state.username = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ==================================================
# 9. LOGIN AND REGISTRATION PAGE
# ==================================================

def show_auth_page():
    left, center, right = st.columns([1, 1.15, 1])

    with center:
        st.markdown(
            '<div class="login-title">📰 NewsGuard</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="login-subtitle">'
            'NLP-Based Fake News Detection System'
            '</div>',
            unsafe_allow_html=True
        )

        mode = st.radio(
            "Account",
            ["Login", "Register"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if mode == "Login":
            with st.form("login_form"):
                st.subheader("Welcome back")

                username = st.text_input(
                    "Username",
                    placeholder="Enter your username"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password"
                )

                submitted = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True
                )

                if submitted:
                    if not username.strip() or not password:
                        st.error("Please enter your username and password.")

                    elif verify_user(username, password):
                        st.session_state.username = username.strip()
                        st.session_state.page = "Dashboard"
                        st.rerun()

                    else:
                        st.error("Invalid username or password.")

            st.caption("New here? Select Register to create an account.")

        else:
            with st.form("register_form"):
                st.subheader("Create your account")

                new_username = st.text_input(
                    "Username",
                    placeholder="Choose a username"
                )

                new_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="At least 8 characters"
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Re-enter your password"
                )

                submitted = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True
                )

                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")

                    else:
                        success, message = register_user(
                            new_username,
                            new_password
                        )

                        if success:
                            st.success(
                                message + " Please log in."
                            )
                        else:
                            st.error(message)

        st.markdown(
            '<p class="small-note" style="text-align:center">'
            'Your news predictions are saved in your account history.'
            '</p>',
            unsafe_allow_html=True
        )


# ==================================================
# 10. DASHBOARD
# ==================================================

def show_dashboard():
    username = st.session_state.username
    records = get_history(username)

    total = len(records)
    fake_count = sum(
        1 for row in records if row[2] == "Fake News"
    )
    real_count = sum(
        1 for row in records if row[2] == "Real News"
    )

    st.title("Dashboard")
    st.write(f"Welcome back, **{username}**!")
    st.caption("Your personal fake news detection overview.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Total Predictions</div>'
            f'<div class="metric-value">{total}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Predicted Fake</div>'
            f'<div class="metric-value">{fake_count}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Predicted Real</div>'
            f'<div class="metric-value">{real_count}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Recent Predictions")

    if not records:
        st.info(
            "No predictions yet. Open News Prediction "
            "to analyze your first article."
        )
    else:
        for row in records[:5]:
            _, article, prediction, checked_at = row

            with st.container(border=True):
                col_a, col_b = st.columns([3, 1])

                with col_a:
                    st.write(article[:180] +
                             ("..." if len(article) > 180 else ""))

                with col_b:
                    st.write(f"**{prediction}**")
                    st.caption(checked_at)

    st.caption(
        "These figures describe your saved predictions, "
        "not verified real-world news statistics."
    )


# ==================================================
# 11. NEWS PREDICTION PAGE
# ==================================================

def show_news_detection():
    st.title("📰 News Prediction")
    st.write(
        "Enter a news article and let the trained machine "
        "learning model classify its text."
    )

    with st.form("prediction_form"):
        article = st.text_area(
            "News article",
            height=190,
            placeholder=(
                "Paste the complete news article here..."
            )
        )

        submitted = st.form_submit_button(
            "Check News",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if len(article.strip()) < 30:
            st.warning(
                "Please enter a longer news article "
                "for a more meaningful prediction."
            )

        else:
            try:
                with st.spinner("Analyzing article..."):
                    prediction, confidence = predict_news(article)

                save_history(
                    st.session_state.username,
                    article.strip(),
                    prediction
                )

                st.session_state.last_prediction = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "article": article.strip()
                }

                st.rerun()

            except Exception as error:
                st.error(
                    "Could not make a prediction. "
                    "Check the model files and preprocessing."
                )
                st.code(str(error))

    result = st.session_state.last_prediction

    if result:
        st.divider()
        st.subheader("Prediction Result")

        prediction = result["prediction"]

        if prediction == "Real News":
            st.markdown(
                '<div class="result-box real-result">'
                '✓ Prediction: Real News'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box fake-result">'
                '⚠ Prediction: Fake News'
                '</div>',
                unsafe_allow_html=True
            )

        if result["confidence"] is not None:
            st.write(
                f"Model confidence: "
                f"**{result['confidence']:.2f}%**"
            )

        st.caption(
            "This is a model prediction, not a verified fact-check. "
            "Confidence is not a guarantee of correctness."
        )


# ==================================================
# 12. PREDICTION HISTORY PAGE
# ==================================================

def show_history():
    st.title("Prediction History")
    st.write("Review your previously analyzed articles.")

    records = get_history(st.session_state.username)

    if not records:
        st.info("Your prediction history is empty.")
        return

    st.write(f"**{len(records)}** saved predictions")

    for row in records:
        record_id, article, prediction, checked_at = row

        with st.expander(
            f"{prediction} — {checked_at}"
        ):
            st.write("**Article**")
            st.write(article)

            st.write("**Prediction**")
            st.write(prediction)

            st.caption(f"Record ID: {record_id}")


# ==================================================
# 13. MAIN APPLICATION
# ==================================================

def main():
    if not st.session_state.username:
        show_auth_page()
        return

    with st.sidebar:
        st.title("📰 NewsGuard")
        st.caption("Fake News Detection System")

        st.divider()

        st.write("Logged in as")
        st.write(f"**{st.session_state.username}**")

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "News Prediction",
                "Prediction History"
            ],
            key="navigation"
        )

        st.divider()

        if st.button("Logout", use_container_width=True):
            st.session_state.username = None
            st.session_state.page = "Dashboard"
            st.session_state.last_prediction = None
            st.rerun()

    if page == "Dashboard":
        show_dashboard()

    elif page == "News Prediction":
        show_news_detection()

    elif page == "Prediction History":
        show_history()


if __name__ == "__main__":
    main()