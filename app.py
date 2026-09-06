

import gradio as gr
import spacy
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from spacy import displacy
import os

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def named_entity_recognition(text):
    if not text.strip():
        return "Please enter some text."
    doc = nlp(text)
    if not doc.ents:
        return "No named entities found."
    result = []
    for ent in doc.ents:
        result.append(
            f"{ent.text} → {ent.label_} "
            f"({spacy.explain(ent.label_)})"
        )
    return "\n".join(result)

def pos_tagging(text):
    if not text.strip():
        return "Please enter some text."
    doc = nlp(text)
    result = []
    for token in doc:
        if not token.is_space:
            result.append(
                f"{token.text} → "
                f"POS: {token.pos_}, "
                f"TAG: {token.tag_}"
            )
    return "\n".join(result)

def pos_distribution(text):
    if not text.strip():
        return None
    doc = nlp(text)
    pos_counts = Counter(
        token.pos_
        for token in doc
        if not token.is_space and not token.is_punct
    )
    if not pos_counts:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        pos_counts.keys(),
        pos_counts.values()
    )
    ax.set_title("POS Distribution")
    ax.set_xlabel("Part of Speech")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def lemmatization(text):
    if not text.strip():
        return "Please enter some text."
    doc = nlp(text)
    result = []
    for token in doc:
        if not token.is_space:
            result.append(
                f"{token.text} → {token.lemma_}"
            )
    return "\n".join(result)

def stemming(text):
    if not text.strip():
        return "Please enter some text."
    doc = nlp(text)
    result = []
    for token in doc:
        if not token.is_space and not token.is_punct:
            result.append(
                f"{token.text} → {stemmer.stem(token.text)}"
            )
    return "\n".join(result)

def morphology(text):
    if not text.strip():
        return "Please enter some text."
    doc = nlp(text)
    result = []
    for token in doc:
        if not token.is_space:
            result.append(
                f"{token.text} → {token.morph}"
            )
    return "\n".join(result)

def dependency_parsing(text):
    if not text.strip():
        return "<p>Please enter some text.</p>"
    doc = nlp(text)
    html = displacy.render(
        doc,
        style="dep",
        jupyter=False,
        options={
            "compact": True,
            "distance": 100
        }
    )
    return html

def run_nlp(text, operation):
    if operation == "Named Entity Recognition":
        return named_entity_recognition(text)
    elif operation == "POS Tagging":
        return pos_tagging(text)
    elif operation == "POS Distribution":
        return pos_distribution(text)
    elif operation == "Lemmatization":
        return lemmatization(text)
    elif operation == "Stemming":
        return stemming(text)
    elif operation == "Morphology":
        return morphology(text)
    elif operation == "Dependencies":
        return dependency_parsing(text)
demo = gr.Interface(
    fn=run_nlp,
    inputs=[
        gr.Textbox(
            label="Enter Text",
            placeholder="Enter your sentence or paragraph here...",
            lines=6
        ),
        gr.Dropdown(
            choices=[
                "Named Entity Recognition",
                "POS Tagging",
                "POS Distribution",
                "Lemmatization",
                "Stemming",
                "Morphology",
                "Dependencies"
            ],
            label="Select NLP Operation",
            value="Named Entity Recognition"
        )
    ],
    outputs=gr.HTML(),
    title="NLP Analysis Dashboard",
    description=(
        "Perform Named Entity Recognition, POS Tagging, "
        "POS Distribution, Lemmatization, Stemming, "
        "Morphology and Dependency Parsing."
    )
)
demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
