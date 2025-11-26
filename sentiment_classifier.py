import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def build_dataset():
   
    data = {
    "text": [
        "This tutoring session helped me a lot",
        "I learned nothing today",
        "Great experience, thank you",
        "The topic was confusing",
        "It was okay, nothing special",
        "Amazing instructor",
        "I am frustrated and lost",
        "Very clear explanation",
        "The class was boring",
        "I feel more confident now",
        "Terrible explanation and hard to follow",
        "Loved the energy of the instructor",
        "Not helpful at all",
        "Pretty decent overall",
        "I'm very confused by today's lesson",
        "This was incredibly helpful",
        "I enjoyed the session",
        "The assignment was too difficult",
        "Everything made sense thanks to the tutor",
        "I would not recommend this class",
        "This was a positive experience",
        "I don’t think I learned anything",
        "The material was simple",
        "I didn’t like the pacing",
        "The slides were clear and well organized",
        "The tutor didn’t explain enough",
        "Good session, I understand the topic now",
        "Neutral experience overall",
        "I feel lost and overwhelmed",
        "Really fun and informative",
        "I feel motivated after the session",
        "The example problems were too hard",
        "I love how the tutor explained everything",
        "The class moved way too quickly",
        "Very professional and helpful",
        "The content was outdated",
        "The instructor seemed impatient",
        "The session boosted my understanding",
        "Nothing was explained clearly",
        "I liked the interactive examples",
        "It was a waste of time",
        "The tutor answered all my questions",
        "I still don’t get the topic",
        "Overall it was okay",
        "This helped me feel prepared",
        "The noise made it hard to focus",
        "Instructor was friendly",
        "I felt ignored",
        "The session was productive",
        "I didn’t find this helpful",
        "I learned so much today",
        "Too many distractions",
        "The tutor made me feel confident",
        "The instructions were unclear",
        "Very patient instructor",
        "This session was unnecessary",
        "I liked the examples used",
        "It didn’t help me understand",
        "Good pacing and structure",
        "The assignment confused me more",
        "I feel proud of my progress",
        "The session felt rushed",
        "Great communication",
        "The material was overwhelming",
        "This clarified so many things",
        "I wanted more detailed explanations",
        "Helpful but could be better",
        "I feel like I wasted my time",
        "Very effective session",
        "The tutor barely explained anything",
        "I feel encouraged after this",
        "Not the best, but okay",
        "The tutor was very knowledgeable",
        "I didn’t learn anything new",
        "I feel more comfortable with the topic",
        "This was disappointing",
        "I appreciated the step by step help",
        "The tutor talked too fast",
        "The session improved my skills",
        "I don’t feel prepared at all",
        "I enjoyed this experience",
        "This made things worse",
        "Clear examples and explanations",
        "The content was too basic",
        "Very engaging instructor",
        "The session dragged on",
        "I finally understand the concept",
        "The tutor didn’t answer my questions",
        "This made the topic easier",
        "I feel mentally drained",
        "Great atmosphere and teaching",
        "I still feel confused",
        "This really helped me improve",
        "Instructor was unorganized",
        "I feel good about the assignment now",
        "The lesson was hard to follow",
        "This made me confident for the test",
        "I don't think I benefited from this",
        "The tutor was super supportive",
        "I feel like I actually learned today",
        "The instructions were too vague",
        "Amazing teaching style",
        "I still feel unsure about everything"
    ],
        "label": [
        "positive", "negative", "positive", "negative", "neutral",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "neutral", "negative",
        "positive", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "neutral", "negative", "positive",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "negative", "positive", "negative", "positive",
        "negative", "neutral", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "neutral", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "positive", "negative", "positive", "negative",
        "negative", "positive", "negative", "positive", "positive",
        "negative", "positive", "negative",
    ]

}


    return pd.DataFrame(data)


def train_model(df: pd.DataFrame):
    """
    Train a sentiment classifier using TF-IDF and Logistic Regression.
    Returns the vectorizer, trained model, and test performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Convert text to numeric features
    vectorizer = TfidfVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectors, y_train)

    # Evaluate
    predictions = model.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return vectorizer, model, accuracy, report


def interactive_loop(vectorizer, model):
    
    print("\ Sentiment Classifier Ready!")
    print("Type a comment to analyze sentiment, or 'quit' to exit.")

    while True:
        user_input = input("\nYour comment: ")
        if user_input.strip().lower() == "quit":
            print("Thank you for analyzing")
            break

        user_vector = vectorizer.transform([user_input])
        result = model.predict(user_vector)[0]
        print(f"Prediction: {result}")


def main():
    print("Building dataset...")
    df = build_dataset()
    print("Training model...")
    vectorizer, model, accuracy, report = train_model(df)

    print("\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    # Start interactive prediction loop
    interactive_loop(vectorizer, model)


if __name__ == "__main__":
    main()
