import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import re

 
# Load Model
 
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

 
# Load Role Catalog and Compute Embeddings
 
@st.cache_data
def load_roles():
    # df = pd.read_csv("roles_catalog.csv")
    df = pd.read_csv("roles_catalog_large.csv", quotechar='"', on_bad_lines='skip')
    df.fillna("", inplace=True)
    role_texts = (df["role_title"] + ". " + df["role_description"]).tolist()
    embeddings = model.encode(role_texts, convert_to_numpy=True, show_progress_bar=False)
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)), metric="cosine")
    nn.fit(embeddings)
    return df, embeddings, nn

roles_df, role_embeddings, nn = load_roles()

 
# Robust Skill Extractor
 
def extract_skills(text, vocab=None):
    if vocab is None:
        vocab = [
            "python","java","c++","react","node","django","flask","sql",
            "tensorflow","pytorch","nlp","cloud","aws","docker","kubernetes",
            "git","html","css","javascript","linux","azure","pandas","numpy"
        ]
    text_low = text.lower()
    found = []
    for skill in vocab:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_low):
            found.append(skill)
    return found

 
# Generate 30/60/90 Learning Plan
 
def generate_learning_plan(role_title, missing_skills):
    plan = {
        "30 Days": [],
        "60 Days": [],
        "90 Days": []
    }
    if not missing_skills:
        plan["30 Days"].append("Revise existing skills and practice small projects.")
        plan["60 Days"].append("Work on intermediate-level projects in your role domain.")
        plan["90 Days"].append("Prepare for interviews and apply for jobs.")
    else:
        for i, skill in enumerate(missing_skills):
            if i % 3 == 0:
                plan["30 Days"].append(f"Learn basics of {skill} (online tutorials).")
            elif i % 3 == 1:
                plan["60 Days"].append(f"Do a mini-project using {skill}.")
            else:
                plan["90 Days"].append(f"Master {skill} and apply it in a portfolio project.")
    return plan

 
# Streamlit UI
 
st.title("🎯 Personalized Career & Skills Advisor")
st.write("An AI-powered tool that maps your skills, recommends career paths, and prepares you for the evolving job market.")

input_type = st.radio("Choose Input Type:", ["Paste Resume", "Enter Skills"])

if input_type == "Paste Resume":
    resume_text = st.text_area("Paste your resume or profile text here")
    user_skills = extract_skills(resume_text)
elif input_type == "Enter Skills":
    user_skills = [s.strip() for s in st.text_input("Enter your skills (comma-separated)").split(",")]

if st.button("Find Career Paths"):
    if not user_skills or user_skills == [""]:
        st.warning("⚠️ Please provide resume text or skills.")
    else:
        skill_sentence = ", ".join(user_skills)
        user_emb = model.encode([skill_sentence], convert_to_numpy=True)

        distances, idxs = nn.kneighbors(user_emb, n_neighbors=min(5, len(role_embeddings)))
        st.subheader("🔍 Top Career Matches")
        for dist, idx in zip(distances[0], idxs[0]):
            score = 1 - float(dist)
            role = roles_df.iloc[idx]
            required = [s.strip().lower() for s in str(role.get("required_skills","")).split(",") if s.strip()]
            missing = [s for s in required if s not in [x.lower().strip() for x in user_skills]]

            with st.expander(f"{role['role_title']} (Score: {round(score,3)})"):
                st.write("**Role Description:**", role["role_description"])
                st.write("**Your Skills:**", ", ".join(user_skills))
                st.write("**Missing Skills:**", ", ".join(missing) if missing else "None 🎉")
                
                # Generate Learning Plan
                st.markdown("### 📅 Personalized 30/60/90 Day Plan")
                plan = generate_learning_plan(role["role_title"], missing)
                for k, v in plan.items():
                    st.markdown(f"**{k}:**")
                    for item in v:
                        st.markdown(f"- {item}")
                st.progress(int(score*100))

        st.success("✅ Career recommendations generated!")
