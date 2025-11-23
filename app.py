import streamlit as st
from backend.llm import generate_output_schema, generate_input_schema, generate_mapping
from backend.rag import retrieve_similar_api, embed_text
from backend.schemas import pretty_json

st.set_page_config(page_title="AutoSpec POC", layout="wide")

st.title("🔗 AI-Driven API Mapper-AutoSpec POC")

# ----------------------------
#  Step 1: Requirement Input
# ----------------------------
st.header("1. Enter Custom API Requirement")

requirement_text = st.text_area(
    "Describe the API you want to build", 
    placeholder="Example: I need an API that returns flight status, gate info, passenger name based on PNR..."
)

if st.button("Generate Output Schema"):
    with st.spinner("Generating output schema..."):
        output_schema = generate_output_schema(requirement_text)
        st.session_state["output_schema"] = output_schema

if "output_schema" in st.session_state:
    st.subheader("Generated Output Schema")
    st.code(pretty_json(st.session_state["output_schema"]), language="json")

# ----------------------------
#  Step 2: Retrieve Similar API (RAG)
# ----------------------------
if "output_schema" in st.session_state:
    st.header("2. Finding Closest Matching Existing API")

    if st.button("Search Matching API"):
        with st.spinner("Searching vector DB..."):
            api_match = retrieve_similar_api(st.session_state["output_schema"])
            st.session_state["api_match"] = api_match

if "api_match" in st.session_state:
    st.subheader("Matched API")
    st.json(st.session_state["api_match"])

# ----------------------------
#  Step 3: Input Schema for the Matched API
# ----------------------------
if "api_match" in st.session_state:
    st.header("3. Generate Input Schema")

    if st.button("Generate Input Schema"):
        with st.spinner("Generating input schema..."):
            input_schema = generate_input_schema(st.session_state["api_match"])
            st.session_state["input_schema"] = input_schema

if "input_schema" in st.session_state:
    st.subheader("Generated Input Schema")
    st.code(pretty_json(st.session_state["input_schema"]), language="json")

# ----------------------------
#  Step 4: Generate Field Mapping
# ----------------------------
#if "input_schema" in st.session_state:
#    st.header("4. Generate Mapping Between Custom Schema & Existing API")
if "output_schema" in st.session_state and "api_match" in st.session_state:
    st.header("4. Generate Mapping Between Custom Schema & Existing API")

    if st.button("Generate Mapping"):
        with st.spinner("Creating mapping..."):
            mapping = generate_mapping(
                st.session_state["output_schema"],
                st.session_state["api_match"]["output_schema"] 
            )
            st.session_state["mapping"] = mapping

if "mapping" in st.session_state:
    st.subheader("Generated Field Mapping")
    st.code(pretty_json(st.session_state["mapping"]), language="json")
