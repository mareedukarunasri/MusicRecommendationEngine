# =====================================================
# MoodMate - Emotion Detection + Music Recommendation
# =====================================================

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------
# STEP 1: Load Emotion Dataset
# -----------------------------------------------------
# emotion_dataset.csv must contain:
# text,emotion

emotion_file = "data/emotion_dataset.csv"
emotion_df = pd.read_csv(emotion_file)

print("Emotion dataset loaded!")

# -----------------------------------------------------
# STEP 2: Train Emotion Detection Model
# -----------------------------------------------------

emotion_vectorizer = TfidfVectorizer()
X = emotion_vectorizer.fit_transform(emotion_df["text"])
y = emotion_df["emotion"]

emotion_model = LogisticRegression()
emotion_model.fit(X, y)

print("Emotion model trained successfully!")

# -----------------------------------------------------
# STEP 3: Load Music Dataset
# -----------------------------------------------------
# music_dataset.csv must contain:
# song,artist,tags

music_file = "data/music_dataset.csv"
music_df = pd.read_csv(music_file)

print("Music dataset loaded!")

# -----------------------------------------------------
# STEP 4: Convert Music Tags → TF-IDF
# -----------------------------------------------------

music_vectorizer = TfidfVectorizer()
tag_matrix = music_vectorizer.fit_transform(music_df["tags"])

# -----------------------------------------------------
# STEP 5: Emotion → Music Mapping
# -----------------------------------------------------

emotion_to_tags = {
    "happy": "happy energetic dance pop",
    "sad": "sad calm slow emotional",
    "angry": "rock intense energetic",
    "calm": "soft instrumental peaceful",
    "surprise": "upbeat electronic fun"
}

# -----------------------------------------------------
# STEP 6: Emotion Prediction Function
# -----------------------------------------------------

def predict_emotion(text):
    text_vec = emotion_vectorizer.transform([text])
    prediction = emotion_model.predict(text_vec)
    return prediction[0]

# -----------------------------------------------------
# STEP 7: Recommendation Function
# -----------------------------------------------------

def recommend_music(emotion, top_n=5):

    emotion = emotion.lower()

    if emotion not in emotion_to_tags:
        print("Emotion not supported")
        return

    emotion_vector = music_vectorizer.transform(
        [emotion_to_tags[emotion]]
    )

    similarity_scores = cosine_similarity(
        emotion_vector, tag_matrix
    )

    scores = similarity_scores.flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    recommendations = music_df.iloc[top_indices][["song","artist"]]

    print("\n🎵 Recommended Songs:")
    for _, row in recommendations.iterrows():
        print(f"{row['song']} - {row['artist']}")

# -----------------------------------------------------
# STEP 8: Run Full Pipeline
# -----------------------------------------------------

if __name__ == "__main__":

    print("\n===== MoodMate System =====")
    user_text = input("How are you feeling today? : ")

    detected_emotion = predict_emotion(user_text)

    print("\nDetected Emotion:", detected_emotion)

    recommend_music(detected_emotion)