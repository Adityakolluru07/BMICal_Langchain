import json
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain.globals import set_debug
import time
from collections import deque
from dotenv import load_dotenv
import os
load_dotenv()

# Loading Groq API
groq_api_key = os.getenv("GROQ_API_KEY")


set_debug(False)

class BMI(BaseModel):
    bmi_category: str = Field(description="BMI category of the person")

# Initialize the language model
# llm = Ollama(model="gemma")
# llm = Ollama(model="llama2")
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0.1)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile", temperature=0.1)
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.1)

# Initialize a deque to store response times for moving average
response_times = deque(maxlen=10)  # Store the last 10 response times

# Create a prompt template
template = """
Answer the user query.\n{format_instructions}\n
Given the following information about a person:
Height: {height} cm
Weight: {weight} kg
Age: {age} years
Gender: {gender}
BMI: {bmi}

According to the World Health Organization (WHO) BMI categories, provide BMI category.
"""

parser = JsonOutputParser(pydantic_object=BMI)
# print(parser.get_format_instructions())

prompt = PromptTemplate(
    partial_variables={"format_instructions": parser.get_format_instructions()},
    input_variables=["height", "weight", "age", "gender", "bmi"],
    template=template
)

# Create an LLMChain
chain = prompt | llm | parser

def assess_health(height, weight, age, gender, bmi)->BMI:

    # Run the chain and print debugger output
    response = chain.invoke({"height": height, "weight": weight, "age": age, "gender": gender, "bmi": bmi})    
    
    if isinstance(response, dict) and "bmi_category" in response:
        return BMI(bmi_category=response["bmi_category"])
    else:
        raise ValueError("Please try again")

# Streamlit UI
st.title('Health Assessment App')

height_input = st.text_input('Enter height (ft\'in")', value="5'10\"", placeholder="e.g., 5'11\"")
if height_input:
    try:
        ft, inches = height_input.replace('"', '').split("'")
        height_cm = (int(ft) * 30.48) + (int(inches) * 2.54)
        height = round(height_cm, 2)  # Round to 2 decimal places
    except ValueError:
        st.error("Please enter height in the format: ft'in\" (e.g., 5'11\")")
        height = None  # Default height for 5'10" in cm
else:
    height = None

weight = st.number_input('Enter weight (kg)', min_value=0.0, max_value=500.0, value=75.0, placeholder="Enter your weight")
age = st.number_input('Enter age (years)', min_value=0, max_value=150, value=27, placeholder="Enter your age")
gender = st.selectbox('Select gender', ['Male', 'Female'])

if st.button('Assess Health'):
    try:
        bmi = weight / (height/100)**2
        result = assess_health(height, weight, age, gender, bmi)
        st.write('Assessment Result:\n')
        st.write(f"BMI Value: {bmi:.2f}")
        st.write(f"BMI Category: {result.bmi_category}")
    except ValueError as e:
        st.error(e)

# To run this code:
# 1. Make sure you have installed all required dependencies (langchain, langchain_community, streamlit)
# 2. Make sure you have Ollama installed and the Gemma model available
# 3. Save this file as app.py
# 4. Open a terminal and navigate to the directory containing app.py
# 5. Run the command: streamlit run app.py