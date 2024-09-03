import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

# Loading Groq API
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.1)

def assess_health(healthy_weight, good_blood_pressure, normal_cholesterol, no_other_issues):
    # Create a prompt template
    template = """
    Given the following health information:
    Healthy Weight Range: {healthy_weight}
    Good Blood Pressure: {good_blood_pressure}
    Normal Cholesterol Level: {normal_cholesterol}
    No Other Health Issues: {no_other_issues}

    Please provide a "GOOD" or "NOT GOOD" response based on these factors.
    DO NOT give any disclaimer or additional information.
    Just the response "GOOD" or "NOT GOOD" in one line.
    Provide a brief suggestion for improvement if needed.
    
    Response format should be like:

    You are in Good health
    or
    You are not in Good health

    """

    prompt = PromptTemplate(
        input_variables=["healthy_weight", "good_blood_pressure", "normal_cholesterol", "no_other_issues"],
        template=template
    )

    # Create an LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    response = chain.run(healthy_weight=healthy_weight, good_blood_pressure=good_blood_pressure, 
                         normal_cholesterol=normal_cholesterol, no_other_issues=no_other_issues)

    return response.strip()

# Streamlit UI
st.title('Health Assessment App')
st.write('Healthy weight')
healthy_weight = st.radio('Are you in a healthy weight range for your height and age today?', ('True', 'False'), key='healthy_weight', horizontal=True)

st.write('Blood pressure')
good_blood_pressure = st.radio('Do you have good blood pressure today?', ('True', 'False'), key='good_blood_pressure', horizontal=True)

st.write('Cholesterol')
normal_cholesterol = st.radio('Is your cholesterol level normal today?', ('True', 'False'), key='normal_cholesterol', horizontal=True)

st.write('Other Health Issues')
no_other_issues = st.radio('Are you clear of other health issues not addressed above?', ('True', 'False'), key='no_other_issues', horizontal=True)

if st.button('Assess Health'):
    result = assess_health(healthy_weight, good_blood_pressure, normal_cholesterol, no_other_issues)
    st.write('Assessment Result:')
    st.write(result)

# To run this code:
# 1. Make sure you have installed all required dependencies (langchain, langchain_groq, streamlit)
# 2. Save this file as app2.py
# 3. Open a terminal and navigate to the directory containing app2.py
# 4. Run the command: streamlit run app2.py
