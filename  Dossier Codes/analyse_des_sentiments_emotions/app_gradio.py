import gradio as gr
import plotly.express as px
import pandas as pd

"""
def analyse_sentiment(input_text):
    # Exemple : Retourne le sentiment et l'émotion (à remplacer par votre propre logique)
    sentiment = "Positive"  # Exemple simplifié
    emotion = "Joy"  # Exemple simplifié
    return sentiment, emotion

def generate_plotly_chart():
    # Exemple de visualisation Plotly (à adapter avec votre propre logique de visualisation)
    df = pd.DataFrame({
        "Emotion": ["Joy", "Sadness", "Anger"],
        "Frequency": [5, 3, 1]
    })
    fig = px.bar(df, x="Emotion", y="Frequency", title="Emotion Distribution")
    return fig

def process_input(input_text):
    sentiment, emotion = analyse_sentiment(input_text)ss
    fig = generate_plotly_chart()
    return sentiment, emotion, fig

iface = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),
    outputs=[gr.Textbox(label="Sentiment"), gr.Textbox(label="Emotion"), gr.Plot()],
    title="Text Analysis with Plotly Visualization"
)

# Ceci lance l'interface Gradio localement et affiche l'URL pour y accéder
if __name__ == "__main__":
    iface.launch(share=True)

"""
import gradio as gr
import plotly.express as px
import pandas as pd
import random

# Load your dataset
data = pd.read_csv(r'C:\Users\alyas\Desktop\TER\test_sentiment_analysis\donnees_et_visus_analyse_sentiment_emotion/all_emo.csv')

# Function to update the dashboard
def update_dashboard(search_text, sentiment, emotion):
    # Filter data based on user input
    filtered_data = data
    if search_text:
        filtered_data = filtered_data[filtered_data['Text'].str.contains(search_text, case=False, na=False)]
    if sentiment != "All":
        filtered_data = filtered_data[filtered_data['Sentiment'] == sentiment]
    if emotion != "All":
        filtered_data = filtered_data[filtered_data['Emotion'] == emotion]

    # Select a random tweet from the filtered dataset
    if len(filtered_data) > 0:
        sample_tweet = filtered_data.sample(1)
        sample_text = sample_tweet['Text'].iloc[0]
    else:
        sample_text = "No tweets matching the criteria."

    # Generate visualizations for the sample tweet
    if len(filtered_data) > 0:
        # Pie chart for emotions
        emotion_distribution = sample_tweet[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']].iloc[0]
        fig_pie = px.pie(names=emotion_distribution.index, values=emotion_distribution.values, title="Emotion Distribution")

        # Bar chart for sentiments
        sentiment_scores = sample_tweet[['positive', 'neutral', 'negative']].iloc[0]
        fig_bar = px.bar(x=sentiment_scores.index, y=sentiment_scores.values, title="Sentiment Scores")
    else:
        fig_pie = px.pie(title="No data for Emotion Distribution")
        fig_bar = px.bar(title="No data for Sentiment Scores")

    # Convert filtered data to HTML table and return all outputs
    filtered_html = filtered_data.head(10).to_html()  # Display only the first 10 rows
    
    return sample_text, fig_pie, fig_bar, filtered_html

# Create Gradio interface
interface = gr.Interface(
    fn=update_dashboard,
    inputs=[
        gr.Textbox(label="Search Text"),
        gr.Dropdown(choices=["All"] + sorted(data['Sentiment'].unique()), label="Sentiment"),
        gr.Dropdown(choices=["All"] + sorted(data['Emotion'].unique()), label="Emotion")
    ],
    outputs=[
        gr.Textbox(label="Sample Tweet"),
        gr.Plot(label="Emotion Distribution (Pie Chart)"),
        gr.Plot(label="Sentiment Scores (Bar Chart)"),
        gr.HTML(label="Filtered Data")
    ],
    title="Sentiment and Emotion Analysis Dashboard",
    description="Filter and search the dataset based on text, sentiment, and emotion. Displays a sample tweet and corresponding visualizations."
)

# Run the app
if __name__ == "__main__":
    interface.launch(share=True)
