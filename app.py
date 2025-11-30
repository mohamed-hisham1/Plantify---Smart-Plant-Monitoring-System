import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
import io
import pickle
import joblib
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

# --- Setup & Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'Crop_Dataset_updated.csv')
lgb_model_path = os.path.join(current_dir, 'model.pkl')
cv_model_path = os.path.join(current_dir, 'Color_Images.h5')

# Configure Gemini (Replace with your method of loading secrets or env vars)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAwUaYy4P4zJJfZYhEuQ5s0YEERKuPCpS4")
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True
else:
    GEMINI_ENABLED = False

# ==========================================
# --- 1. CROP RECOMMENDER LOGIC ---
# ==========================================
CROP_MODEL_LOADED = False
rf_model = None
crop_data = None
ideal_conditions = None

crop_descriptions = {
    'rice': "A staple crop thriving in wet conditions.", 'maize': "Versatile cereal grain.", 
    'chickpea': "Drought-resistant legume.", 'kidneybeans': "Prefers warm weather.",
    'pigeonpeas': "Tropical legume.", 'mungbean': "Matures quickly.", 
    'mothbeans': "Drought-resistant.", 'blackgram': "Nutritious urad bean.", 
    'lentil': "Ancient nutritious legume.", 'pomegranate': "Fruit-bearing shrub.", 
    'banana': "Thrives in hot, humid climates.", 'mango': "Tropical tree.", 
    'grapes': "Grows best in temperate climates.", 'watermelon': "Vine-like plant.", 
    'muskmelon': "Thrives in hot, dry climates.", 'apple': "Prefers cool climates.", 
    'orange': "Citrus tree.", 'papaya': "Tropical fruit.", 
    'coconut': "Thrives in coastal areas.", 'cotton': "Fiber crop.", 
    'jute': "Vegetable fiber crop.", 'coffee': "Prefers tropical climate."
}

try:
    with open(lgb_model_path, "rb") as f:
        rf_model = pickle.load(f)
    crop_data = pd.read_csv(data_path)

    ideal_conditions = crop_data.groupby('label')[
        ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    ].mean()

    CROP_MODEL_LOADED = True

except Exception as e:
    CROP_MODEL_LOADED = False
    st.error(f"Model load error: {e}")


def predict_crop(N, P, K, ph, temp, humid, rain, top_k):
    if not CROP_MODEL_LOADED:
        return "Model not found.", None, None

    # Logic
    user_input = pd.DataFrame([{'N': N, 'P': P, 'K': K, 'temperature': temp, 'humidity': humid, 'ph': ph, 'rainfall': rain}])
    probs = rf_model.predict_proba(user_input)[0]
    topk_idx = probs.argsort()[-top_k:][::-1]
    
    # Prepare Output 1: Probabilities for Label
    top_crops_dict = {rf_model.classes_[i]: float(probs[i]) for i in topk_idx}
    
    # Prepare Output 2: Comparison Plot (Best Match)
    best_crop = rf_model.classes_[topk_idx[0]]
    ideal_vals = ideal_conditions.loc[best_crop]
    input_vals = user_input.iloc[0]
    
    diff_series = ((input_vals - ideal_vals).abs() / ideal_vals) * 100
    diff_df = diff_series.reset_index().rename(columns={'index': 'Parameter', 0: 'Difference (%)'})
    diff_df = diff_df.sort_values(by='Difference (%)', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(diff_df['Parameter'], diff_df['Difference (%)'], color='orange')
    ax.set_xlabel("Percentage Difference from Ideal")
    ax.set_title(f"Deviation from Ideal Conditions for {best_crop}")
    plt.tight_layout()
    
    description = f"**Top Recommendation:** {best_crop}\n\n{crop_descriptions.get(best_crop, '')}"
    
    return top_crops_dict, fig, description

# ==========================================
# --- 2. COMPUTER VISION LOGIC ---
# ==========================================
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", 
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_leaf_spot)", 
    "Grape___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"
]

def build_model_architecture(num_classes):
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=IMG_SIZE+(3,))
    base.trainable = False
    inputs = Input(shape=IMG_SIZE+(3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

cv_model = None
if os.path.exists(cv_model_path):
    try:
        cv_model = build_model_architecture(len(CLASS_NAMES))
        cv_model.load_weights(cv_model_path)
    except Exception as e:
        print(f"CV Model Error: {e}")

def predict_disease(image):
    if cv_model is None:
        return {"Error": 0.0}
    if image is None:
        return None
        
    # Preprocess
    img = Image.fromarray(image).resize(IMG_SIZE)
    img_array = np.array(img)
    if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = cv_model.predict(img_array)
    probs = predictions[0]
    
    # Return dictionary for Gradio Label
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

# ==========================================
# --- 3. DATA ANALYSIS LOGIC ---
# ==========================================
def load_dataset(file):
    if file is None: return None
    try:
        df = pd.read_csv(file.name)
        return df
    except:
        return None

def get_data_summary(df):
    if df is None: return "No data uploaded."
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def plot_data(df, plot_type, col1, col2):
    if df is None: return None
    
    if plot_type == "Histogram" and col1:
        return px.histogram(df, x=col1, title=f"Histogram of {col1}")
    elif plot_type == "Box Plot" and col1:
        return px.box(df, y=col1, title=f"Box Plot of {col1}")
    elif plot_type == "Scatter Plot" and col1 and col2:
        return px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
    elif plot_type == "Bar Chart" and col1 and col2:
        grouped = df.groupby(col1)[col2].mean().reset_index()
        return px.bar(grouped, x=col1, y=col2, title=f"Avg {col2} by {col1}")
    return None

def update_columns(file):
    df = load_dataset(file)
    if df is not None:
        cols = df.columns.tolist()
        # Return updates for dropdowns
        return df, gr.Dropdown(choices=cols, value=cols[0]), gr.Dropdown(choices=cols, value=cols[1] if len(cols)>1 else cols[0])
    return None, gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

def ai_chat(message, history, df_state):
    if not GEMINI_ENABLED:
        return "Gemini API key missing."
    if df_state is None:
        return "Please upload a dataset first."
        
    summary = get_data_summary(df_state)
    prompt = f"""You are a data analyst. Dataset summary:
    {summary}
    ---
    User Question: "{message}"
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # Updated model name
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# --- UI CONSTRUCTION (Gradio Blocks) ---
# ==========================================
# theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald")

with gr.Blocks(title="Smart Plant Monitoring") as demo:
    gr.Markdown("# 🌿 Smart Plant Monitoring System")
    
    # State to hold the uploaded dataframe
    df_state = gr.State()

    with gr.Tabs():
        # --- TAB 1: CROP RECOMMENDATION ---
        with gr.Tab("🌾 Crop Recommendation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 Soil & Climate Data")
                    n_in = gr.Number(label="Nitrogen (N)", value=50)
                    p_in = gr.Number(label="Phosphorus (P)", value=50)
                    k_in = gr.Number(label="Potassium (K)", value=50)
                    ph_in = gr.Number(label="pH Level", value=6.5)
                    temp_in = gr.Number(label="Temperature (°C)", value=25)
                    hum_in = gr.Number(label="Humidity (%)", value=60)
                    rain_in = gr.Number(label="Rainfall (mm)", value=800)
                    top_k_in = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top K Results")
                    rec_btn = gr.Button("🚀 Recommend Crops", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    crop_labels = gr.Label(num_top_classes=5, label="Top Recommendations")
                    crop_desc = gr.Markdown()
                    crop_plot = gr.Plot(label="Parameter Deviation")

            rec_btn.click(
                fn=predict_crop,
                inputs=[n_in, p_in, k_in, ph_in, temp_in, hum_in, rain_in, top_k_in],
                outputs=[crop_labels, crop_plot, crop_desc]
            )

        # --- TAB 2: DISEASE DETECTION ---
        with gr.Tab("🦠 Disease Detection"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="Upload Leaf Image", sources=["upload", "webcam"])
                    detect_btn = gr.Button("🔍 Analyze Leaf", variant="primary")
                with gr.Column():
                    disease_labels = gr.Label(label="Predictions", num_top_classes=3)
            
            detect_btn.click(fn=predict_disease, inputs=img_in, outputs=disease_labels)

        # --- TAB 3: DATA ANALYSIS (Researcher) ---
        with gr.Tab("🔬 Scientific Data Analysis"):
            gr.Markdown("Upload a CSV file to analyze data, plot graphs, or chat with AI.")
            file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
            
            with gr.Tabs():
                with gr.Tab("Data Overview"):
                    data_preview = gr.Dataframe(label="Head")
                    data_info = gr.Textbox(label="Info", lines=10)
                
                with gr.Tab("Plotting"):
                    with gr.Row():
                        plot_type = gr.Dropdown(["Histogram", "Box Plot", "Scatter Plot", "Bar Chart"], label="Plot Type", value="Histogram")
                        col1 = gr.Dropdown(label="Column X")
                        col2 = gr.Dropdown(label="Column Y (Optional)")
                    plot_btn = gr.Button("Generate Plot")
                    main_plot = gr.Plot()
                    
                with gr.Tab("Chat with AI"):
                    chatbot = gr.ChatInterface(
                        fn=ai_chat, 
                        additional_inputs=[df_state],
                        description="Ask questions about your uploaded dataset."
                    )

            # --- Event Listeners for Data Tab ---
            # When file is uploaded, update state and dropdown columns
            file_upload.change(
                fn=update_columns,
                inputs=file_upload,
                outputs=[df_state, col1, col2]
            ).then(
                fn=lambda df: (df.head() if df is not None else None, get_data_summary(df)),
                inputs=df_state,
                outputs=[data_preview, data_info]
            )
            
            # Plot button listener
            plot_btn.click(
                fn=plot_data,
                inputs=[df_state, plot_type, col1, col2],
                outputs=main_plot
            )

if __name__ == "__main__":
    demo.launch()