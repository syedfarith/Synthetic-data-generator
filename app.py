import streamlit as st
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
from io import StringIO

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")

# Check if API key is loaded correctly
if not api_key:
    st.error("API key is missing! Please check your .env file.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# Function to get field suggestions from LLM
def get_field_suggestions(user_input):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": f"Extract dataset fields and their types (text, numeric, binary) from: {user_input}. Suggest at least 5 fields and make them realistic."}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        
        response = chat_completion.choices[0].message.content
        print("Raw API Response:", response)  # Debugging output

        # Parse response into field list
        cleaned_response = [line.strip() for line in response.split("\n") if ":" in line]
        fields = [{"name": field.split(":")[0].strip(), "type": field.split(":")[1].strip()} for field in cleaned_response]
        return fields

    except Exception as e:
        st.error(f"Error in fetching fields: {e}")
        return []

# Function to generate synthetic data using LLM
def generate_data_with_llm(user_input, fields, rows=50):
    field_str = ", ".join([f"{field['name']} ({field['type']})" for field in fields])
    prompt = f"Generate a dataset with {rows} rows using the following fields: {field_str} for the {user_input} task. Return data in CSV format. No notes, only dataset. Strictly no duplicates."

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        st.error(f"Error in dataset generation: {e}")
        return ""

# Streamlit UI
st.title("🛠 Synthetic Dataset Generator")

# Initialize session state for dynamic fields
if "fields" not in st.session_state:
    st.session_state.fields = []

# User input for dataset description
user_input = st.text_area("📝 Describe your dataset requirements:")

# Generate fields using LLM button
if st.button("🔍 Generate Fields using LLM"):
    if user_input:
        suggested_fields = get_field_suggestions(user_input)
        if suggested_fields:
            st.session_state.fields = suggested_fields
        else:
            st.warning("No fields were generated. Try again or modify your prompt.")
    else:
        st.warning("Please enter dataset requirements before generating fields.")

# Add field manually
if st.button("➕ Add Field"):
    st.session_state.fields.append({"name": "", "type": "Text"})

# Display dynamic fields with delete buttons
valid_field_types = ["Text", "Numeric", "Binary"]

for i, field in enumerate(st.session_state.fields):
    col1, col2, col3 = st.columns([3, 2, 1])  # Name, Type, Delete button
    with col1:
        field_name = st.text_input(f"Field {i+1} Name:", field["name"])
    with col2:
        field_type = st.selectbox(f"Field {i+1} Type:", valid_field_types, index=valid_field_types.index(field["type"]) if field["type"] in valid_field_types else 0)
    with col3:
        if st.button(f"❌ Delete {i+1}", key=f"delete_{i}"):
            st.session_state.fields.pop(i)
            st.rerun()  

    # Update session state
    if i < len(st.session_state.fields):
        st.session_state.fields[i] = {"name": field_name, "type": field_type}

# Input for number of rows
num_rows = st.number_input("📊 Number of rows to generate:", min_value=1, max_value=1000, value=50)

# Generate dataset button
if st.button("🚀 Generate Dataset"):
    if user_input and st.session_state.fields:
        dataset_csv = generate_data_with_llm(user_input, st.session_state.fields, rows=num_rows)
        
        if dataset_csv:
            # Convert CSV string to DataFrame
            dataset = pd.read_csv(StringIO(dataset_csv))
            st.dataframe(dataset)

            # Provide CSV download option
            csv = dataset_csv.encode('utf-8')
            st.download_button("📥 Download CSV", csv, "synthetic_dataset.csv", "text/csv")
        else:
            st.error("Failed to generate dataset. Please try again.")

    else:
        st.warning("Please enter dataset requirements and at least one field!")
