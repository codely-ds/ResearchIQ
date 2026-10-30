
import streamlit as st

from nvidia_client import get_embedding
from storage import add_note, load_notes, search_notes

st.set_page_config(
    page_title="ResearchIQ",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 ResearchIQ")
st.subheader("AI-Powered Semantic Research Search")

st.write(
    "Save research notes and discover related information "
    "using NVIDIA Nemotron-3-Embed-1B."
)

notes = load_notes()

col1, col2 = st.columns(2)

with col1:
    st.metric("Saved Notes", len(notes))

with col2:
    st.metric("AI Model", "Nemotron 1B")

st.divider()

st.header("📝 Add a Research Note")

with st.form("add_note_form"):
    title = st.text_input("Note title")
    topic = st.text_input("Topic or category")

    content = st.text_area(
        "Research content",
        height=180,
        placeholder="Write your research note here..."
    )

    submitted = st.form_submit_button(
        "Generate Embedding & Save",
        type="primary"
    )

if submitted:
    if not title.strip() or not content.strip():
        st.warning("Please enter both a title and content.")

    else:
        try:
            with st.spinner("Generating NVIDIA embedding..."):
                embedding = get_embedding(
                    content.strip(),
                    input_type="passage"
                )

                add_note(
                    title.strip(),
                    topic.strip() or "General",
                    content.strip(),
                    embedding
                )

            st.success("Research note saved successfully!")
            st.rerun()

        except Exception as error:
            st.error(f"Could not save note: {error}")

st.divider()

st.header("🔍 Search Your Research")

query = st.text_input(
    "What do you want to find?",
    placeholder="e.g. How can AI predict student performance?"
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=3
)

if st.button("Search Research", type="primary"):
    if not query.strip():
        st.warning("Enter a search query.")

    elif not load_notes():
        st.info("Add at least one research note first.")

    else:
        try:
            with st.spinner("Searching by meaning..."):
                query_embedding = get_embedding(
                    query.strip(),
                    input_type="query"
                )

                results = search_notes(
                    query_embedding,
                    top_k
                )

            st.subheader("🎯 Most Relevant Notes")

            for rank, note in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(f"### {rank}. {note['title']}")
                    st.caption(f"Topic: {note['topic']}")
                    st.write(note["content"])

                    st.progress(
                        min(
                            1.0,
                            max(0.0, note["similarity"])
                        )
                    )

                    st.caption(
                        "Cosine similarity: "
                        f"{note['similarity']:.3f}"
                    )

        except Exception as error:
            st.error(f"Search failed: {error}")

st.divider()

st.caption(
    "ResearchIQ | Python • Streamlit • NVIDIA Embeddings"
)
