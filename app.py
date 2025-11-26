import streamlit as st
from sentiment_classifier import build_dataset, train_model


@st.cache_resource
def load_model():
    """
    Load dataset and train the model once.
    Streamlit will cache this so it does not retrain on every interaction.
    """
    df = build_dataset()
    vectorizer, model, accuracy, report = train_model(df)
    return vectorizer, model, accuracy, report


def get_reaction(label: str):
    """
    Map sentiment label to a friendly reaction.
    """
    if label == "positive":
        return "✅ **Positive** — This feedback sounds encouraging! 😄"
    elif label == "negative":
        return "❌ **Negative** — This feedback shows frustration or issues. 😞"
    elif label == "neutral":
        return "➖ **Neutral** — This feedback is more mixed or flat. 😐"
    else:
        return "🤔 Could not determine sentiment."


def main():
    st.set_page_config(page_title="Student Feedback Sentiment Checker", layout="centered")

    st.title("🎓 Student Feedback Sentiment Checker")
    st.write(
        "Type in a student comment and this app will predict whether the sentiment "
        "is **positive**, **negative**, or **neutral** based on a trained ML model."
    )

    # Load model
    with st.spinner("Loading model..."):
        vectorizer, model, accuracy, report = load_model()

    st.markdown(f"**Model Accuracy (test set):** `{accuracy:.2f}`")

    st.markdown("---")

    # Text input
    feedback = st.text_area(
        "Enter a student feedback comment:",
        placeholder="Example: The tutor explained everything clearly and I feel more confident now."
    )

    if st.button("Analyze Sentiment"):
        if not feedback.strip():
            st.warning("Please enter a comment first.")
        else:
            # Transform input and predict
            user_vector = vectorizer.transform([feedback])
            prediction = model.predict(user_vector)[0]

            st.subheader("Prediction")
            st.markdown(get_reaction(prediction))

            # (Optional) show raw label
            st.caption(f"Raw model label: `{prediction}`")

    st.markdown("---")
    st.caption("Built by Jordon Ammons — Student Feedback Tracker")


if __name__ == "__main__":
    main()
