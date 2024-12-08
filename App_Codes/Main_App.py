import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import base64
from sklearn.linear_model import LogisticRegression
import os
# Login Credits
USER_CREDENTIALS = {
    "admin": "password123",
    "BHARGAV": "BHARGAV"
}

#  Cleaned Dataset Load
@st.cache_data
def load_data():

    file_path = 'census_modified_v2.xlsx'
    data = pd.read_excel(os.path.join(BASE_DIR,"census_modified_v2.xlsx"))
    return data.drop(columns=['Unnamed: 0'])

# Training the model
@st.cache_data
def train_model(data):
    X = data.drop(columns=['income'])
    y = data['income']

    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ##('classifier', RandomForestClassifier(random_state=42))
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    return model, preprocessor, X_train

#  Clustering
@st.cache_data
def perform_clustering(data, _preprocessor):
    X = data.drop(columns=['income'])
    X_preprocessed = _preprocessor.transform(X)

    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(X_preprocessed)

    return kmeans, clusters

# Recommendation Function
def map_financial_products_local(cluster, income):
    if income == 1:  # Income >50K
        if cluster in [0, 1]:
            return "Premium Credit Cards", "./Images/credit_card.jpg.png", "https://card.americanexpress.com/d/platinum-card/?utm_mcid=&utm_source=google&utm_medium=cpc&utm_term=%2Bamex%20%2Bplatinum%20%2Boffer&utm_cmpid=18510627008&utm_adgid=145040663274&utm_tgtid=aud-279874126270:kwd-1810812030648&utm_mt=p&utm_adid=719450130239&utm_dvc=c&utm_ntwk=g&utm_adpos=&utm_plcmnt=&utm_locphysid=1021041&utm_locintid=&utm_feeditemid=&utm_devicemdl=&utm_plcmnttgt=&utm_programname=brandcps&utm_cmp=PlatinumBenefit&utm_sl=&gclid=EAIaIQobChMIjueTk4KWigMVdUlHAR28uQHrEAAYASAAEgKSPvD_BwE"
        elif cluster in [2, 3]:
            return "Investment in S&P Index & ETFs", "./Images/etfs.jpg", "https://robinhood.com/us/en/stocks/SPY/"

        elif cluster in [4]:
            return "Investment in Bitcoin/ Other Crypto", "./Images/bitcoin.jpg.tiff", "https://www.coinbase.com/price/bitcoin"

        else:
            return "Home Loans", "loan.jpg", "https://www.example.com/home-loans"
    else:  # Income <=50K
        if cluster in [0, 1]:
            return "Basic Credit Cards", "./Images/basic_credit_card.jpg", "https://www.discover.com/products/student-it.html?sc=RJQS&iq_id=r43700078743852575&cmpgnid=ps-dca-google-sitelink&iq_id=r43700078743852575&cmpgnid=ps-dca-google-brand-credit-card&source=PSGOOGLE&gad_source=1&gclid=EAIaIQobChMIrI2Ho5CVigMVkWFHAR14_BSTEAAYASABEgL_ZvD_BwE&gclsrc=aw.ds"
        elif cluster in [2, 3]:
            return "Personal Loans", "./Images/personal_loan.png", "https://www.sofi.com/personal-loan-dr-1/?campaign=MRKT_SEM_LND_PL_NONBRA_ACQ_EXT_MBW_tCPA_QUST_20201221_HIGH-EFFICIENCY_PSE_GOG_NONE_US_EN_SFka3danansp2392h6hanc_e_g_c_599945728722_personal%20loans&utm_source=MRKT_ADWORDS&utm_medium=SEM&utm_campaign=MRKT_SEM_LND_PL_NONBRA_ACQ_EXT_MBW_tCPA_QUST_20201221_HIGH-EFFICIENCY_PSE_GOG_NONE_US_EN_SFka3danansp2392h6hanc_e_g_c_599945728722_personal%20loans&cl_vend=google&cl_ch=sem&cl_camp=11896099122&cl_adg=112338067381&cl_crtv=599945728722&cl_kw=personal%20loans&cl_pub=google.com&cl_place=&cl_dvt=c&cl_pos=&cl_mt=e&cl_gtid=aud-693343449970%3Akwd-10131831&opti_ca=11896099122&opti_ag=112338067381&opti_ad=599945728722&opti_key=aud-693343449970%3Akwd-10131831&gclid=EAIaIQobChMIzZ_WxoKWigMVFGNHAR0ijiEZEAAYAiAAEgJLMvD_BwE&gclsrc=aw.ds&gad_source=1&ds_agid=58700006592097569&ds_cid=71700000078164822&ds_eid=700000001842560&ds_kid=43700072219871989"
        else:
            return "Savings Plans", "./Images/saving.jpg", "https://www.banking.barclaysus.com/tiered-savings.html?cjdata=MXxOfDB8WXww&AID=15161582&PID=100333868&SID=8S9FUjMs0t&cjevent=9297d89db46a11ef823f04c50a82b838&refid=CJNNIRTTIER"


def map_financial_products(cluster, income):
    if income == 1:  # Income >50K
        if cluster in [0, 1]:
            return (
                "Premium Credit Cards",
                os.path.join(BASE_DIR, "Images", "credit_card.jpg.png"),
                "https://card.americanexpress.com/d/platinum-card/?utm_mcid=&utm_source=google&utm_medium=cpc&utm_term=%2Bamex%20%2Bplatinum%20%2Boffer&utm_cmpid=18510627008&utm_adgid=145040663274&utm_tgtid=aud-279874126270:kwd-1810812030648&utm_mt=p&utm_adid=719450130239&utm_dvc=c&utm_ntwk=g&utm_adpos=&utm_plcmnt=&utm_locphysid=1021041&utm_locintid=&utm_feeditemid=&utm_devicemdl=&utm_plcmnttgt=&utm_programname=brandcps&utm_cmp=PlatinumBenefit&utm_sl=&gclid=EAIaIQobChMIjueTk4KWigMVdUlHAR28uQHrEAAYASAAEgKSPvD_BwE"
            )
        elif cluster in [2, 3]:
            return (
                "Investment in S&P Index & ETFs",
                os.path.join(BASE_DIR, "Images", "etfs.jpg"),
                "https://robinhood.com/us/en/stocks/SPY/"
            )
        elif cluster in [4]:
            return (
                "Investment in Bitcoin/ Other Crypto",
                os.path.join(BASE_DIR, "Images", "bitcoin.jpg.tiff"),
                "https://www.coinbase.com/price/bitcoin"
            )
        else:
            return (
                "Home Loans",
                os.path.join(BASE_DIR, "Images", "loan.jpg"),
                "https://www.example.com/home-loans"
            )
    else:  # Income <=50K
        if cluster in [0, 1]:
            return (
                "Basic Credit Cards",
                os.path.join(BASE_DIR, "Images", "basic_credit_card.jpg"),
                "https://www.discover.com/products/student-it.html?sc=RJQS&iq_id=r43700078743852575&cmpgnid=ps-dca-google-sitelink&iq_id=r43700078743852575&cmpgnid=ps-dca-google-brand-credit-card&source=PSGOOGLE&gad_source=1&gclid=EAIaIQobChMIrI2Ho5CVigMVkWFHAR14_BSTEAAYASABEgL_ZvD_BwE&gclsrc=aw.ds"
            )
        elif cluster in [2, 3]:
            return (
                "Personal Loans",
                os.path.join(BASE_DIR, "Images", "personal_loan.png"),
                "https://www.sofi.com/personal-loan-dr-1/?campaign=MRKT_SEM_LND_PL_NONBRA_ACQ_EXT_MBW_tCPA_QUST_20201221_HIGH-EFFICIENCY_PSE_GOG_NONE_US_EN_SFka3danansp2392h6hanc_e_g_c_599945728722_personal%20loans&utm_source=MRKT_ADWORDS&utm_medium=SEM&utm_campaign=MRKT_SEM_LND_PL_NONBRA_ACQ_EXT_MBW_tCPA_QUST_20201221_HIGH-EFFICIENCY_PSE_GOG_NONE_US_EN_SFka3danansp2392h6hanc_e_g_c_599945728722_personal%20loans&cl_vend=google&cl_ch=sem&cl_camp=11896099122&cl_adg=112338067381&cl_crtv=599945728722&cl_kw=personal%20loans&cl_pub=google.com&cl_place=&cl_dvt=c&cl_pos=&cl_mt=e&cl_gtid=aud-693343449970%3Akwd-10131831&opti_ca=11896099122&opti_ag=112338067381&opti_ad=599945728722&opti_key=aud-693343449970%3Akwd-10131831&gclid=EAIaIQobChMIzZ_WxoKWigMVFGNHAR0ijiEZEAAYAiAAEgJLMvD_BwE&gclsrc=aw.ds&gad_source=1&ds_agid=58700006592097569&ds_cid=71700000078164822&ds_eid=700000001842560&ds_kid=43700072219871989"
            )
        else:
            return (
                "Savings Plans",
                os.path.join(BASE_DIR, "Images", "saving.jpg"),
                "https://www.banking.barclaysus.com/tiered-savings.html?cjdata=MXxOfDB8WXww&AID=15161582&PID=100333868&SID=8S9FUjMs0t&cjevent=9297d89db46a11ef823f04c50a82b838&refid=CJNNIRTTIER"
            )


# Cluster descriptions
def get_cluster_description(cluster_number):
    descriptions = {
        0: "Cluster 0: Young individuals, entry-level roles or part-time jobs. Focused on basic savings and credit needs.",
        1: "Cluster 1: Mid-level professionals, often married, with stable income. Prefer basic credit cards or small loans.",
        2: "Cluster 2: Single or divorced individuals in mid-career. Likely seeking personal loans or credit repair options.",
        3: "Cluster 3: High-income professionals, married, prioritizing investments in ETFs and S&P index.",
        4: "Cluster 4: Wealthy individuals with high capital gains. Likely to invest in Bitcoins & Cryptos .",
        5: "Cluster 5: Low-income earners with part-time jobs. Prefer affordable savings plans or basic credit solutions."
    }
    return descriptions.get(cluster_number, "Cluster description not available.")

# Function to read image and convert to Base64
def encode_image_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



encoded_image_landing = encode_image_to_base64(  os.path.join(BASE_DIR, "Images", "landing_page.jpg"))
encoded_image_main = encode_image_to_base64(os.path.join(BASE_DIR, "Images", "Main_page_landing_2.jpg"))

# Load and preprocess data
data = load_data()

# Train model
model, preprocessor, X_train = train_model(data)

# Perform clustering
kmeans, clusters = perform_clustering(data, preprocessor)
data['cluster'] = clusters


# Login page
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

## Adding CSS Styling
if not st.session_state.logged_in:
    st.markdown(
        f"""
            <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded_image_landing}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.9); /* Slight transparency */
        }}

         h1 {{
            color: pink; /* Inner text color */
            -webkit-text-stroke: 1px black; /* Outline thickness and color */
            text-stroke: 1px black; /* Fallback for other browsers */
            font-weight: bold; /* Optional: Make the text bold */
            text-align: center; /* Center-align the text */
        }}


        button[data-testid="stButton"] > div {{
            color: white !important; /* Button text color black */
            background-color: white !important; /* Button background white */
            border: 2px solid black !important; /* Black border */
            padding: 10px 20px !important; /* Adjust padding */
            font-size: 16px !important; /* Adjust font size */
            border-radius: 5px !important; /* Rounded corners */
            transition: background-color 0.3s ease !important; /* Smooth transition */
        }}
        button[data-testid="stButton"] > div:hover {{
            background-color: #f0f0f0 !important; /* Slightly darker hover color */
        }}
        </style>
            """,
        unsafe_allow_html=True,
    )
    st.title("Welcome to the Secure Financial Recommendation System - Patron")


    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("Successfully logged in!")
        else:
            st.error("Invalid username or password.")
    st.stop()

# App main page
st.title("Welcome to the Secure Financial Recommendation System - Patron")

st.markdown(
    f"""

           <style>
       [data-testid="stAppViewContainer"] {{
           background-image: url("data:image/jpeg;base64,{encoded_image_main}");
           background-size: cover;
           background-position: center;
           background-attachment: fixed;
       }}
       /* Style the sidebar container */
    [data-testid="stSidebar"] {{
        color: white; /* Default sidebar text color */
    }}

    /* Style sidebar titles */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6 {{
        color: WHITE !important; /* Make sidebar headings black */
    }}

    /* Style sidebar labels and text inputs */
    [data-testid="stSidebar"] label {{
        color: white !important; /* Make sidebar labels black */
    }}

    /* Style other sidebar text */
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] .css-145kmo2 {{
        color: white !important; /* Force black text for other elements */
    }}
        
        </style>
            """,
        unsafe_allow_html=True,
    )

# User inputs
st.sidebar.header("Enter your details:")

user_input = {}
education_options = [
    f"{row['education_level']} - {row['education-num']}"
    for _, row in data[['education_level', 'education-num']].iterrows()
]
user_input['education_combined'] = st.sidebar.selectbox("Education Levels", list(set(education_options)))

# Parse combined input back to education_level and education-num for prediction
selected_education = user_input['education_combined'].split(" - ")
user_input['education_level'] = selected_education[0]
user_input['education-num'] = int(float(selected_education[1]))

for col in X_train.columns:
    if col not in ['education_level', 'education-num', 'education_combined']:
        if col in data.select_dtypes(include=['object']).columns:
            user_input[col] = st.sidebar.selectbox(col, data[col].unique())
        else:
            user_input[col] = st.sidebar.number_input(col, value=float(data[col].median()))

# Convert inputs to DataFrame
user_input_df = pd.DataFrame([user_input])

# Make predictions and assign cluster
if st.button("Predict"):
    # Predict income
    income_prediction = model.predict(user_input_df)[0]
    income_result = ">80K" if income_prediction == 1 else "<=80K"

    # Predict cluster
    user_input_preprocessed = preprocessor.transform(user_input_df)
    cluster_prediction = kmeans.predict(user_input_preprocessed)[0]

    # Get cluster description
    cluster_description = get_cluster_description(cluster_prediction)

    # Map financial products
    product, image_file, product_url = map_financial_products(cluster_prediction, income_prediction)

    # Display results
    st.subheader("Results")
    st.write(f"**Predicted Income**: {income_result}")
    st.write(f"**Assigned Cluster**: {cluster_prediction}")
    st.write(f"**Cluster Description**: {cluster_description}")

    # Display recommended financial product
    st.write(f"**Recommended Product**: {product}")
    st.image(image_file, caption=product, use_container_width=True)
    st.markdown(f"[Learn More About {product}]({product_url})")
