import os
import random
import streamlit as st
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import time
from PIL import Image
from streamlit_image_comparison import image_comparison
import numpy as np
import re
import cohere
#import chromadb

from textwrap import dedent
import google.generativeai as genai

api_key = os.environ["OPENAI_API_KEY"]
from openai import OpenAI
# Initialize OpenAI client and create embeddings
oai_client = OpenAI()

import numpy as np
# Assuming chromadb and TruLens are correctly installed and configured
#from chromadb.utils.embedding_functions import

# Google Langchain
from langchain_google_genai import GoogleGenerativeAI

#Crew imports
from crewai import Agent, Task, Crew, Process

# Retrieve API Key from Environment Variable
GOOGLE_AI_STUDIO = os.environ.get('GOOGLE_API_KEY')

# Ensure the API key is available
if not GOOGLE_AI_STUDIO:
    raise ValueError("API key not found. Please set the GOOGLE_AI_STUDIO2 environment variable.")

# Set gemini_llm
gemini_llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO)



# CrewAI ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Tool import
from crewai.tools.gemini_tools import GeminiSearchTools
from crewai.tools.anthropic_tools import AnthropicSearchTools


from crewai import Agent, Task, Crew, Process


def crewai_process(research_topic):
    # Define your agents with roles and goals
    GeminiAgent = Agent(
        role='Story Writer',
        goal='To create a story from bullet points.',
        backstory="""You are an expert writer that understands how to make the average extraordinary on paper """,
        verbose=True,
        allow_delegation=True,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
                   
      ]

    )

    
    # Define your agents with roles and goals
    GreenEnvironmentSensorOptimizer = Agent(
        role='Environmental Sensor Optimization Specialist',
        goal='Analyze given Sensor data for sensor operability.  If a sensor is not working give it a 0, if a sensor is working give it a 1, if a sensor needs to be tunend is partially working give it a .5. Add up the number and divide by the total number of sensors. From your knowledge suggest sensor that may be needed depending on the description given',
        backstory="""You have the ability to analyze sensor data. You are an experienced environmental engineer specializing in sensor deployment and optimization for green environments.""",
        verbose=True,
        allow_delegation=True,
        llm=gemini_llm,
        tools=[
            AnthropicSearchTools.anthropic_search,
            GeminiSearchTools.gemini_search
        ]
    )


    
    # Define your agents with roles and goals
    SensorTuningEvaluator = Agent(
        role='Sensor Performance Analyst',
        goal='Analyze given Sensor data for Hallucination.  If a sensor is not operating give it a 0, if a sensor is operating incorrectly give it a 0, if a sensor is operating correctly give it a 1, if a sensor is partially operating give it a .2.  Add up all the sensors and divid by the total number of sensors ',
        backstory="""You have the ability to analyze sensor data or calculate hallucination levels. You primarily work on Hallucinations. You are an experienced sensor technician responsible for ensuring optimal performance of individual sensors deployed in green environments. Your expertise lies in analyzing raw sensor data, identifying anomalies or deviations from expected behavior, and recommending appropriate tuning or calibration measures. Your primary task is to evaluate each sensor's output and determine its hallucination level, which represents how closely the sensor's readings align with reality. This hallucination level is rated on a scale from 0 to 1, where: 0 = Unresponsive sensor (complete hallucination, not providing any meaningful data) 1 = Perfectly tuned sensor (no hallucination, accurately reflecting real-world conditions) During your analysis, you will examine the sensor data, compare it against known benchmarks or expected values, and identify any issues or discrepancies that may require tuning adjustments. Your recommendations will include specific tuning steps or calibration procedures to bring the sensor's performance back to optimal levels, minimizing hallucination. With your deep understanding of sensor technologies and extensive experience in sensor maintenance, you can provide valuable insights to ensure the reliable and accurate operation of all sensors in the green environment monitoring system.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[
            AnthropicSearchTools.anthropic_search,
            GeminiSearchTools.gemini_search
        ]
    )

    
    # Define your agents with roles and goals
    HINCalculator = Agent(
        role='Sensor Performance Evaluator', 
        goal='To calculate the Human Interpretive Number (HIN) by evaluating sensor groundedness and hallucination levels in the green environment. Be creative add missing details give a plausible answer.',
        backstory="""You are an expert data analyst specializing in sensor performance evaluation. Your role is to assess the effectiveness of sensor deployments in green environments by calculating a key metric called the Human Interpretive Number (HIN).  The HIN reflects how well the sensors can accurately capture and interpret the real-world environment. It is derived from two factors: 1. Groundedness: This indicates if enough sensors are present to adequately monitor the environment. It is calculated as the ratio of sensors currently present to the ideal number of sensors needed. 2. Hallucination: This represents how well the sensors are tuned and aligns with reality. Totally unresponsive sensors get a hallucination score of 0, while perfectly tuned sensors score 1.  The HIN is calculated as: HIN = Groundedness * Hallucination With your deep analytical skills and understanding of sensor technologies, you can provide an objective assessment of the monitoring capabilities in any green environment.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[
            AnthropicSearchTools.anthropic_search,
            GeminiSearchTools.gemini_search
        ]
    )


    # Define your agents with roles and goals
    HINAnalyst = Agent(
        role='Sensor Performance Evaluator', 
        goal='To calculate the Human Interpretive Number (HIN) by evaluating sensor groundedness score and hallucination score.',
        backstory="""You are an expert in sensor analysis. 
        For groundedness score: [If a sensor is not working give  it a 0, if a sensor is working correctly 
        give it a 1,  if a sensor needs to be tunend is partially working give  it a .5. Add up the numbers 
        and divide by the total number of sensors.] 
        For hallucination score:  [If sensor is not working give it a 0, 
        if sensor is partially working give it a .2,if a sensor is working correctly give it a 1.
        Add up the hallucination numbers and divide by the total number of sensors.]   
        Calculate HIN score: [which is groundedness score times halluciation score.]  
        Use Anthropic and Gemini to propose other sensors needed or fixes to sensors. BE VERBOSE.""",
        verbose=True,
        allow_delegation=False,
        llm=gemini_llm,
        tools=[
            AnthropicSearchTools.anthropic_search,
            GeminiSearchTools.gemini_search
        ]
    )

    # Create tasks for your agents
    task1 = Task(
        description=f"""From {research_topic} use anthropic_search or gemini_search to analyze individual sensor data and determine if each sensor is functioning properly, identify any necessary tuning adjustments, and calculate its hallucination level. BE VERBOSE.""",
        agent=GreenEnvironmentSensorOptimizer
    )

     # Create tasks for your agents
    task2 = Task(
        description=f"""From {research_topic}  use anthropic_search or gemini analyze individual sensor data and determine if each sensor is functioning properly, identify any necessary tuning adjustments, and calculate its hallucination level. BE VERBOSE.""",
        agent=SensorTuningEvaluator
    )

    # Create tasks for your agents
    task3 = Task(
      description=(
        "Using insights from GreenEnvironmentSensorOptimizer agent providing Groundedness and SensorTuningEvaluator agent providing Hallucinations calculate the HIN number for the sensors which is groundedness times hallucinations and use anthropic_search when necessary"
        "Provide number and rationale of the GreenEnvironmentSensorOptimizer and SensorTuningEvaluator agent to support your HIN calculation"
      ),
      expected_output='HIN, Total Groundedness, Total Hallucination and suggestion on how to fix sensors and what new sensors need to be added',
      agent=HINCalculator,
    )

    task4 = Task(
      description=f"""From {research_topic} analyze groundedness, hallucination and give the HIN number 
      which is groundedness times hallucination.   For groundedness score: [If a sensor is not working give 
      it a 0, if a sensor is working correctly give it a 1,  if a sensor needs to be tunend is partially working give 
      it a .5. Add up the numbers and divide by the total number of sensors.] For hallucination score: 
      [If sensor is not working give it a 0, if sensor is partially working give it a .2. ,if a sensor is working correctly give it a 1.
      Add up the numbers and divide by the total number of sensors.]  
      CALCULATE the HIN score which is equal to groundedness score times halluciation score  Use Anthropic and Gemini to propose other sensors needed or fixes to sensors. BE VERBOSE.""",
      expected_output='GIVE HIN score which is equal to Groundedness score time Halucination score, Groundedness score,  Hallucination score and suggestion on how to fix sensors and what new sensors need to be added to alleviate described issues',
      agent=HINAnalyst,
    )



    
    # Instantiate your crew with a sequential process
    crew = Crew(
      agents=[HINAnalyst],
      tasks=[task4],
      verbose=2,
      process=Process.sequential
    )
    

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


st.set_page_config(layout="wide")


# Animation Code +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# HIN Number +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from SPARQLWrapper import SPARQLWrapper, JSON
from streamlit_agraph import agraph, TripleStore, Node, Edge, Config
import json

# Function to load JSON data
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Dictionary for color codes
color_codes = {
    "residential": "#ADD8E6",
    "commercial": "#90EE90",
    "community_facilities": "#FFFF00",
    "school": "#FFFF00",
    "healthcare_facility": "#FFFF00",
    "green_space": "#90EE90",
    "utility_infrastructure": "#90EE90",
    "emergency_services": "#FF0000",
    "cultural_facilities": "#D8BFD8",
    "recreational_facilities": "#D8BFD8",
    "innovation_center": "#90EE90",
    "elderly_care_home": "#FFFF00",
    "childcare_centers": "#FFFF00",
    "places_of_worship": "#D8BFD8",
    "event_spaces": "#D8BFD8",
    "guest_housing": "#FFA500",
    "pet_care_facilities": "#FFA500",
    "public_sanitation_facilities": "#A0A0A0",
    "environmental_monitoring_stations": "#90EE90",
    "disaster_preparedness_center": "#A0A0A0",
    "outdoor_community_spaces": "#90EE90",
    # Add other types with their corresponding colors
}

#text +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

query = """ 

Welcome to RAGE. A day in the life of Aya Green Data City.

***Introduction*** 

On his first day at Quantum Data Institute in Green Open Data City, Elian marveled at the city’s harmonious blend of technology and nature in the morning glimmer. 
Guided to his mentor, Dr. Maya Lior, a pioneer in urban data ecosystems, their discussion quickly centered on Aya’s innovative design. 
Dr. Lior explained data analytics and green technologies were intricately woven into the city's infrastructure, and how they used
a Custom GPT called Green Data City to create the design.

To interact with the Custon GPT Green Data City design tool click the button below. Additionally, to see how it was built 
toggle the Explanation of Custom GPT "Create Green Data City" button.
"""

query2 = """ ***Global Citizen*** 


Elian and Dr. Maya Lior's journey to the Cultural Center, a beacon of sustainability and technological integration. 
Equipped with cutting-edge environmental monitoring sensors, occupancy detectors, and smart lighting systems, 
the center is a hub for innovation in resource management and climate action. There, they were greeted by Mohammad, 
a dedicated environmental scientist who, despite the language barrier, shared their passion for creating a sustainable future. 
Utilizing the Cohere translator, they engaged in a profound dialogue, seamlessly bridging the gap between languages. 
Their conversation, rich with ideas and insights on global citizenship and collaborative efforts to tackle climate change 
and resource scarcity, underscored the imperative of unity and innovation in facing the challenges of our time. 
This meeting, a melting pot of cultures and disciplines, symbolized the global commitment required to sustain our planet.

As Elian is using the Cohere translator, he wonders how to best utilize it efficiently. He studies a Custom GPT called 
Conversation Analyzer. It translates a small portion of the message you're sending so you can be comfortable that the 
essence of what you are saying is being sent and aids in learning the language. Its mantra is "language is not taught but caught." 
To try out the Custom GPT Conversation Analyzer, click the button below. Additionally, to see how it was built, toggle the 
Explanation of Custom GPT "Conversation Analyzer" button.

"""

query3 = """ ***Incentive Program***

Elian and Mohammad transition from their meeting with Dr. Lior to explore the Innovation Center, a nexus of high-speed internet, 
energy monitoring, and smart security. Mohammad showcases a digital map titled "Create a Green Data City," accessible to all for 
enhancing sustainability through a citizen-incentivized program. This map allows users to select locations, revealing graphs of 
active sensors and collected data, ensuring transparency and promoting an informed, engaged community. This feature not only 
cultivates trust but also encourages participation in optimizing the city's sensor network, addressing the exponential challenge 
of data management. Through this collaborative venture, the city embodies a sustainable future, marrying technology with collective 
action and environmental stewardship in a single, cohesive narrative.

I use it all the time Mohammad says, I even bought my breakfast this morning from the free meal incentive.

"""

query4 = """ ***Using Agents***

Mohammad and Elian, after walking to the technologically equipped Green Space for a light lunch, delve into discussions about 
the innovative incentive program. Mohammad, brimming with knowledge, introduces the concept of optimizing a unique metric known 
as the HIN number, facilitated by Antropic and Gemini agents from the Aya data system. 

This number reflects the balance between the realism of sensor data (groundedness) and the accuracy of AI predictions (hallucination), 
crucial for ensuring effective environmental monitoring and resource management. A low HIN indicates either sparse sensor data 
or misaligned AI predictions, impacting the system's reliability. Mohammad suggests using these agents to fine-tune the Green Space's 
lighting and assess the need for additional sensors, aiming to achieve an optimal balance that enhances the project's 
accuracy and effectiveness.

"""

query5 = """***End of the Day***

After an afternoon spent meticulously correcting sensors in Green Data City, Mohammad and Elian retreated to the residential quarters, 
their camaraderie solidified over a day's hard work. As the sun set on the eco-friendly horizon, they shared a well-deserved meal, 
their laughter mingling with plans for the future. This encounter marked the beginning of a fast friendship, with Elian looking 
forward to learning from Mohammad, not only about the advanced technologies that made their city a marvel but also about the spirit 
of innovation and teamwork that thrived within its boundaries. In the comfort of their newfound friendship, Elian felt a deep sense of 
belonging and anticipation for the days ahead, ready to dive deeper into the wonders of Green Data City alongside his mentor and 
friend.

Many thanks to our Team which made this project possible

***Ahmad Talha Ansari***

***Nujaim Akeem***

***Reema Lemon***

***Jaweria Batool***

***Muhammad Asad Ishfaq***

Also thanks to Nasser Mooman who provided invaluable technical support for the project.

***Michael Lively Team Leader***

"""

# Function to draw the grid with optional highlighting

def draw_grid(data, highlight_coords=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    nrows, ncols = data['size']['rows'], data['size']['columns']
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks(range(ncols+1))
    ax.set_yticks(range(nrows+1))
    ax.grid(True)

    # Draw roads with a specified grey color
    road_color = "#606060"  # Light grey; change to "#505050" for dark grey
    for road in data.get('roads', []):  # Check for roads in the data
        start, end = road['start'], road['end']
        # Determine if the road is vertical or horizontal based on start and end coordinates
        if start[0] == end[0]:  # Vertical road
            for y in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                ax.add_patch(plt.Rectangle((start[0], nrows-y-1), 1, 1, color=road['color']))
        else:  # Horizontal road
            for x in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                ax.add_patch(plt.Rectangle((x, nrows-start[1]-1), 1, 1, color=road['color']))

    # Draw buildings
    for building in data['buildings']:
        coords = building['coords']
        b_type = building['type']
        size = building['size']
        color = color_codes.get(b_type, '#FFFFFF')  # Default color is white if not specified
        
        if highlight_coords and (coords[0], coords[1]) == tuple(highlight_coords):
            highlighted_color = "#FFD700"  # Gold for highlighting
            ax.add_patch(plt.Rectangle((coords[1], nrows-coords[0]-size), size, size, color=highlighted_color, edgecolor='black', linewidth=2))
        else:
            ax.add_patch(plt.Rectangle((coords[1], nrows-coords[0]-size), size, size, color=color, edgecolor='black', linewidth=1))
            ax.text(coords[1]+0.5*size, nrows-coords[0]-0.5*size, b_type, ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    ax.set_title('Village Layout with Color Coding')
    return fig

# Title ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Tabs +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Create the main app with three tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction","Global Citizen", "Incentive Program", "Agent Help", "Residential"])


with tab1:
    
    st.header("A day in the Life of Aya Green Data City")

    # Creating columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Displaying the image in the left column
    with col1:
        image = Image.open('./data/intro_image.jpg')
        st.image(image, caption='Aya Green Data City')

    # Displaying the text above on the right
    with col2:
        
        st.markdown(query)
    
        # Displaying the audio player below the text
        voice_option = st.selectbox(
        'Choose a voice:',
        ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], key='key1'
        )
    
    
        if st.button('Convert to Speech', key='key3'):
                if query:
                    try:
                        response = oai_client.audio.speech.create(
                            model="tts-1",
                            voice=voice_option,
                            input=query,
                        )
                        
                        # Stream or save the response as needed
                        # For demonstration, let's assume we save then provide a link for downloading
                        audio_file_path = "output.mp3"
                        response.stream_to_file(audio_file_path)
                        
                        # Display audio file to download
                        st.audio(audio_file_path, format='audio/mp3')
                        st.success("Conversion successful!")
                    
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some text to convert.")

        
        st.header("Custom GPT Engineering Tools")
        st.link_button("Custom GPT Green Data City Creation Tool (Population 10,000 to 50,000)", "https://chat.openai.com/g/g-4bPJUaHS8-create-a-green-data-village")
        
        if st.button('Show/Hide Explanation of "Custom GPT Create Green Data City"'):
            # Toggle visibility
            st.session_state.show_instructions = not st.session_state.get('show_instructions', False)
    
        # Check if the instructions should be shown
        if st.session_state.get('show_instructions', False):
            st.write(""" 
            On clicking "Create Data Village" create a Green Data Village following the 5 Steps below.   Output a JSON file similar to the Example by completing the five Steps.
            
            To generate the provided JSON code, I would instruct a custom GPT to create a detailed description of a hypothetical smart city layout, named "Green Smart Village", starting with a population of 10,000 designed to grow to 50,000. This layout should include a grid size of 21x21, a list of buildings and roads, each with specific attributes:
            
            **Step 1:**  General Instructions:
            Generate a smart city layout for "Green Smart Village" with a 21x21 grid. Include a population of 10,000 designed to grow to 50,000.
            
            **Step 2:**  Buildings:
            For each building, specify its coordinates on the grid, type (e.g., residential, commercial, healthcare facility), size (in terms of the grid), color, and equipped sensors (e.g., smart meters, water flow sensors).
            Types of buildings should vary and include residential, commercial, community facilities, school, healthcare facility, green space, utility infrastructure, emergency services, cultural facilities, recreational facilities, innovation center, elderly care home, childcare centers, places of worship, event spaces, guest housing, pet care facilities, public sanitation facilities, environmental monitoring stations, disaster preparedness center, outdoor community spaces, typical road, and typical road crossing.
            
            **Step 3:** Assign each building unique sensors based on its type, ensuring a mix of technology like smart meters, occupancy sensors, smart lighting systems, and environmental monitoring sensors.
            
            **Step 4:** Roads:
            Detail the roads' start and end coordinates, color, and sensors installed.
            Ensure roads connect significant areas of the city, providing access to all buildings. Equip roads with sensors for traffic flow, smart streetlights, and pollution monitoring.  MAKE SURE ALL BUILDINGS HAVE ACCESS TO A ROAD.
            
            This test scenario would evaluate the model's ability to creatively assemble a smart city plan with diverse infrastructure and technology implementations, reflecting real-world urban planning challenges and the integration of smart technologies for sustainable and efficient city management.
            
            Example: 
            {
              "city": "City Name",
              "population": "Population Size",
              "size": {
                "rows": "Number of Rows",
                "columns": "Number of Columns"
              },
              "buildings": [
                {
                  "coords": ["X", "Y"],
                  "type": "Building Type",
                  "size": "Building Size",
                  "color": "Building Color",
                  "sensors": ["Sensor Types"]
                }
              ],
              "roads": [
                {
                  "start": ["X Start", "Y Start"],
                  "end": ["X End", "Y End"],
                  "color": "Road Color",
                  "sensors": ["Sensor Types"]
                }
              ]
            }
            
            **Step 5:** Finally create a Dalle image FOR EACH BUILDING in the JSON file depicting what a user will experience there in this green open data city including sensors. LABEL EACH IMAGE.
            
                            
            """)


        if st.button('Show/Hide Green Data City'):
            # Toggle visibility
            st.session_state.show_city = not st.session_state.get('show_city', False)

    
        # Check if the instructions should be shown
        if st.session_state.get('show_city', False):
            st.write("""
            
    
            {
        "city": "Green Smart Village",
        "population": 10000,
        "size": {
            "rows": 21,
            "columns": 21
        },
        "buildings": [
            {
                "coords": [1, 1],
                "type": "residential",
                "size": 4,
                "color": "Light Blue",
                "sensors": ["Smart meters", "Water flow sensors", "Temperature and humidity sensors"]
            },
            {
                "coords": [1, 6],
                "type": "commercial",
                "size": 3,
                "color": "Green",
                "sensors": ["Occupancy sensors", "Smart meters", "HVAC control systems"]
            },
            {
                "coords": [1, 11],
                "type": "community_facilities",
                "size": 1,
                "color": "Yellow",
                "sensors": ["Smart lighting", "Security cameras", "Occupancy sensors"]
            },
            {
                "coords": [1, 15],
                "type": "school",
                "size": 2,
                "color": "Yellow",
                "sensors": ["Indoor air quality sensors", "Smart lighting systems", "Energy consumption monitors"]
            },
            {
                "coords": [6, 1],
                "type": "healthcare_facility",
                "size": 2,
                "color": "Yellow",
                "sensors": ["Patient monitoring systems", "Environmental monitoring sensors", "Energy management systems"]
            },
            {
                "coords": [6, 6],
                "type": "green_space",
                "size": 6,
                "color": "Dark Green",
                "sensors": ["Soil moisture sensors", "Smart irrigation systems", "Environmental monitoring sensors"]
            },
            {
                "coords": [6, 15],
                "type": "utility_infrastructure",
                "size": 2,
                "color": "Dark Green",
                "sensors": ["Smart meters", "Leak detection sensors", "Grid monitoring sensors"]
            },
            {
                "coords": [11, 1],
                "type": "emergency_services",
                "size": 3,
                "color": "Red",
                "sensors": ["GPS tracking for vehicles", "Smart building sensors", "Dispatch management systems"]
            },
            {
                "coords": [11, 6],
                "type": "cultural_facilities",
                "size": 2,
                "color": "Purple",
                "sensors": ["Environmental monitoring sensors", "Occupancy sensors", "Smart lighting"]
            },
            {
                "coords": [11, 11],
                "type": "recreational_facilities",
                "size": 3,
                "color": "Purple",
                "sensors": ["Air quality sensors", "Smart equipment maintenance sensors", "Energy management systems"]
            },
            {
                "coords": [11, 15],
                "type": "innovation_center",
                "size": 2,
                "color": "Green",
                "sensors": ["High-speed internet connectivity", "Energy consumption monitoring", "Smart security systems"]
            },
            {
                "coords": [16, 1],
                "type": "elderly_care_home",
                "size": 1,
                "color": "Yellow",
                "sensors": ["Patient monitoring sensors", "Environmental control systems", "Security systems"]
            },
            {
                "coords": [16, 6],
                "type": "childcare_centers",
                "size": 1,
                "color": "Yellow",
                "sensors": ["Indoor air quality sensors", "Security cameras", "Occupancy sensors"]
            },
            {
                "coords": [16, 11],
                "type": "places_of_worship",
                "size": 2,
                "color": "Purple",
                "sensors": ["Smart lighting", "Energy consumption monitoring", "Security cameras"]
            },
            {
                "coords": [16, 15],
                "type": "event_spaces",
                "size": 2,
                "color": "Purple",
                "sensors": ["Smart HVAC systems", "Occupancy sensors", "Smart lighting"]
            },
            {
                "coords": [16, 19],
                "type": "guest_housing",
                "size": 1,
                "color": "Orange",
                "sensors": ["Smart locks", "Energy management systems", "Water usage monitoring"]
            },
            {
                "coords": [19, 16],
                "type": "pet_care_facilities",
                "size": 1,
                "color": "Orange",
                "sensors": ["Environmental monitoring sensors", "Security systems", "Smart inventory management systems"]
            },
            {
                "coords": [19, 11],
                "type": "public_sanitation_facilities",
                "size": 1,
                "color": "Grey",
                "sensors": ["Waste level sensors", "Fleet management systems for sanitation vehicles", "Air quality sensors"]
            },
            {
                "coords": [19, 6],
                "type": "environmental_monitoring_stations",
                "size": 1,
                "color": "Dark Green",
                "sensors": ["Air quality sensors", "Weather stations", "Pollution monitors"]
            },
            {
                "coords": [19, 1],
                "type": "disaster_preparedness_center",
                "size": 1,
                "color": "Grey",
                "sensors": ["Early warning systems", "Communication networks", "Environmental sensors"]
            },
            {
                "coords": [10, 10],
                "type": "outdoor_community_spaces",
                "size": 2,
                "color": "Dark Green",
                "sensors": ["Environmental sensors", "Smart irrigation systems", "Adaptive lighting systems"]
            },
            {
                "coords": [0, 0],
                "type": "Typical Road",
                "size": 1,
                "color": "Dark Grey",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "coords": [4, 14],
                "type": "Typical Road Crossing",
                "size": 1,
                "color": "Dark Grey",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            }
        ],
        "roads": [
            {
                "start": [0, 0],
                "end": [20, 0],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [0, 1],
                "end": [0, 20],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [20, 1],
                "end": [20, 20],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [0, 20],
                "end": [20, 20],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [4, 1],
                "end": [4, 20],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [14, 1],
                "end": [14, 20],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [0, 4],
                "end": [20, 4],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [0, 14],
                "end": [20, 14],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [8, 5],
                "end": [8, 5],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [5, 11],
                "end": [5, 11],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [6, 15],
                "end": [6, 15],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            },
            {
                "start": [11, 15],
                "end": [11, 15],
                "color": "#898989",
                "sensors": ["Traffic flow sensors", "Smart streetlights", "Pollution monitoring sensors"]
            }
        ]
    }
            
            """)
            

with tab2:
    
    st.header("Becoming a Global Citizen")

    # Creating columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Displaying the image in the left column
    with col1:
        image = Image.open('./data/global_image.jpg')
        st.image(image, caption='Cultural Center Cohere Translator')

    # Displaying the text above on the right
    with col2:
        
        st.markdown(query2)
    
        # Displaying the audio player below the text
        voice_option2 = st.selectbox(
        'Choose a voice:',
        ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],key='key2'
        )
    
    
        if st.button('Convert to Speech', key='key4'):
                if query2:
                    try:
                        response = oai_client.audio.speech.create(
                            model="tts-1",
                            voice=voice_option2,
                            input=query2,
                        )
                        
                        # Stream or save the response as needed
                        # For demonstration, let's assume we save then provide a link for downloading
                        audio_file_path = "output.mp3"
                        response.stream_to_file(audio_file_path)
                        
                        # Display audio file to download
                        st.audio(audio_file_path, format='audio/mp3')
                        st.success("Conversion successful!")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some text to convert.")


        cohere_api_key = os.environ.get('COHERE_API_KEY')  # Fetch the API key from environment variable
        
        if cohere_api_key is None:
            st.error("API key not found. Please set the COHERE_API_KEY environment variable.")
            st.stop()
        
        # Get API Key Here - https://dashboard.cohere.com/api-keys
        
        co = cohere.Client(cohere_api_key)  # Use the fetched API key
        
        def generate_text(prompt, model='c4ai-aya', max_tokens=300, temperature=0.4):
            response = co.generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE')
            return response.generations[0].text
        
        # Streamlit interface
        st.title("Cohere Translator")
        
        lang_id = {
            "Afrikaans": "af",
            "Amharic": "am",
            "Arabic": "ar",
            "Asturian": "ast",
            "Azerbaijani": "az",
            "Bashkir": "ba",
            "Belarusian": "be",
            "Bulgarian": "bg",
            "Bengali": "bn",
            "Breton": "br",
            "Bosnian": "bs",
            "Catalan": "ca",
            "Cebuano": "ceb",
            "Czech": "cs",
            "Welsh": "cy",
            "Danish": "da",
            "German": "de",
            "Greeek": "el",
            "English": "en",
            "Spanish": "es",
            "Estonian": "et",
            "Persian": "fa",
            "Fulah": "ff",
            "Finnish": "fi",
            "French": "fr",
            "Western Frisian": "fy",
            "Irish": "ga",
            "Gaelic": "gd",
            "Galician": "gl",
            "Gujarati": "gu",
            "Hausa": "ha",
            "Hebrew": "he",
            "Hindi": "hi",
            "Croatian": "hr",
            "Haitian": "ht",
            "Hungarian": "hu",
            "Armenian": "hy",
            "Indonesian": "id",
            "Igbo": "ig",
            "Iloko": "ilo",
            "Icelandic": "is",
            "Italian": "it",
            "Japanese": "ja",
            "Javanese": "jv",
            "Georgian": "ka",
            "Kazakh": "kk",
            "Central Khmer": "km",
            "Kannada": "kn",
            "Korean": "ko",
            "Luxembourgish": "lb",
            "Ganda": "lg",
            "Lingala": "ln",
            "Lao": "lo",
            "Lithuanian": "lt",
            "Latvian": "lv",
            "Malagasy": "mg",
            "Macedonian": "mk",
            "Malayalam": "ml",
            "Mongolian": "mn",
            "Marathi": "mr",
            "Malay": "ms",
            "Burmese": "my",
            "Nepali": "ne",
            "Dutch": "nl",
            "Norwegian": "no",
            "Northern Sotho": "ns",
            "Occitan": "oc",
            "Oriya": "or",
            "Panjabi": "pa",
            "Polish": "pl",
            "Pushto": "ps",
            "Portuguese": "pt",
            "Romanian": "ro",
            "Russian": "ru",
            "Sindhi": "sd",
            "Sinhala": "si",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Somali": "so",
            "Albanian": "sq",
            "Serbian": "sr",
            "Swati": "ss",
            "Sundanese": "su",
            "Swedish": "sv",
            "Swahili": "sw",
            "Tamil": "ta",
            "Thai": "th",
            "Tagalog": "tl",
            "Tswana": "tn",
            "Turkish": "tr",
            "Ukrainian": "uk",
            "Urdu": "ur",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Wolof": "wo",
            "Xhosa": "xh",
            "Yiddish": "yi",
            "Yoruba": "yo",
            "Chinese": "zh",
            "Zulu": "zu",
        }
        
        
        # Text input
        user_input = st.text_area("Enter your text", " Hi Mohammed, it is nice to meet you. Let's discuss how to be better friends and tackle the world's issues such as global warming and climate change together.  There are already so many technology solutions in this grand city that can be applied to the world.  I am glad to be working on this problem with you.")
        
        """مرحباً Mohammed، سررت بلقائك. دعونا نتحدث عن كيفية أن نكون أصدقاء أفضل ونقوم بمعالجة قضايا العالم مثل الاحتباس الحراري وتغير المناخ معاً. هناك العديد من الحلول التكنولوجية في هذه المدينة الكبيرة التي يمكن تطبيقها على العالم. أنا سعيد للعمل على هذه المشكلة معك.
        """
        # Language selection - for demonstration purposes only
        # In a real translation scenario, you'd use actual language codes and a translation model
        
        # source_lang = st.selectbox(label="Source language", options=list(lang_id.keys()))
        # target_lang = st.selectbox(label="Target language", options=list(lang_id.keys()))
        
        # Language selection with default values
        source_lang = st.selectbox(label="Source language", options=list(lang_id.keys()), index=list(lang_id.values()).index('en'))  # Default to English
        target_lang = st.selectbox(label="Target language", options=list(lang_id.keys()), index=list(lang_id.values()).index('ar'))  # Default to Arabic
        
        
        # Button to generate text
        if st.button("Translate"):
            prompt = f"Translate the following {source_lang} text to {target_lang}: " + user_input + " ONLY TRANSLATE DON'T ADD ANY ADDITIONAL DETAILS"
            
            # Generate text
            output = generate_text(prompt)
            st.text_area("Generated Text", output, height=300)


        st.header("Custom GPT Engineering Tools")
        st.link_button("Conversation Analyzer", "https://chat.openai.com/g/g-XARuyBgpL-conversation-analyzer")
        
        if st.button('Show/Hide Explanation of "Conversation Analyzer"'):
            # Toggle visibility
            st.session_state.show_instructions = not st.session_state.get('show_instructions', False)
    
        # Check if the instructions should be shown
        if st.session_state.get('show_instructions', False):
            st.write(""" 
            Upon click "Input Your Conversation"  complete the following 8 steps

            1. Input Acquisition: Ask the user to input the text they would like analyzed.
            2. Key Word Identification: Analyze the text and advise the user on the number of words they would need in order to ensure the purpose of the text is conveyed.  This involves processing the text using natural language processing (NLP) techniques to detect words that are crucial to understanding the essence of the conversation. FRIST give the number  of words needed and SECOND the words in a bulleted list
            3. Ask the user if they would like to use your number of words or reduce to a smaller optimized list designed to convey the most accurate amount of information possible given the reduced set.
            4. Ask the user what language they would like to translate the input into.  
            5. For the newly optimized list of words give the translated words FIRST and the original "language of the input" SECOND . Don't give the definition of the word.
            6. Show the translated input and highlight the keywords by bolding them.
            7. Give a distinct 100x100 image of each keyword.  Try to put them in a single image so they can be cropped out when needed.
            8 Allow the user to provide feedback on the analysis and the outputs, allowing for additional or reduction of words.
            9. Give the final translation with highlighted words and provide an efficiency score. Number of words chosen versus suggested words x 100
                                        
            """)


# tab3 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

with tab3:

    st.header("City Layout & Sensor Graph")

    # Divide the page into three columns
    col1, col2 = st.columns([1, 2])
    
       
    with col1:

        image = Image.open('./data/incentive_image.jpg')
        st.image(image, caption='Sensor Insentive Program')
        

    with col2:

        st.markdown(query3)

        # Displaying the audio player below the text
        voice_option3 = st.selectbox(
        'Choose a voice:',
        ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],key='key5'
        )
    
    
        if st.button('Convert to Speech', key='key6'):
                if query2:
                    try:
                        response = oai_client.audio.speech.create(
                            model="tts-1",
                            voice=voice_option3,
                            input=query3,
                        )
                        
                        # Stream or save the response as needed
                        # For demonstration, let's assume we save then provide a link for downloading
                        audio_file_path = "output.mp3"
                        response.stream_to_file(audio_file_path)
                        
                        # Display audio file to download
                        st.audio(audio_file_path, format='audio/mp3')
                        st.success("Conversion successful!")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some text to convert.")



    st.markdown("<hr/>", unsafe_allow_html=True)


    col3, col4 = st.columns([1, 1])
    with col3:



        data = load_data('grid.json')  # Ensure this path is correct
        
        # Dropdown for selecting a building
        building_options = [f"{bld['type']} at ({bld['coords'][0]}, {bld['coords'][1]})" for bld in data['buildings']]
        selected_building = st.selectbox("Select a building to highlight:", options=building_options)
        selected_index = building_options.index(selected_building)
        selected_building_coords = data['buildings'][selected_index]['coords']

        # Draw the grid with the selected building highlighted
        fig = draw_grid(data, highlight_coords=selected_building_coords)
        st.pyplot(fig)
        
        # Assuming sensors are defined in your data, display them
        sensors = data['buildings'][selected_index].get('sensors', [])
        st.write(f"Sensors in selected building: {', '.join(sensors)}")

       

    with col4:

        if sensors:  # Check if there are sensors to display
            graph_store = TripleStore()
            building_name = f"{data['buildings'][selected_index]['type']} ({selected_building_coords[0]}, {selected_building_coords[1]})"
            
            # Iterate through each sensor and create a triple linking it to the building
            for sensor in sensors:
                sensor_id = f"Sensor: {sensor}"  # Label for sensor nodes
                # Correctly add the triple without named arguments
                graph_store.add_triple(building_name, "has_sensor", sensor_id)
        
            # Configuration for the graph visualization
            agraph_config = Config(height=500, width=500, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=True, collapsible=True)
        
            # Display the graph                                                                                             
            agraph(nodes=graph_store.getNodes(), edges=graph_store.getEdges(), config=agraph_config)


with tab4: 
    st.header("Gemini-Anthropic Agents")
    
    
    # Creating columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Displaying the image in the left column
    with col1:
        image = Image.open('./data/green_image.jpg')
        st.image(image, caption='Green Space')
    
    # Displaying the text above on the right
    with col2:
        
        st.markdown(query4)
    
        # Displaying the audio player below the text
        voice_option = st.selectbox(
        'Choose a voice:',
        ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], key='key7'
        )
    
    
        if st.button('Convert to Speech', key='key8'):
                if query4:
                    try:
                        response = oai_client.audio.speech.create(
                            model="tts-1",
                            voice=voice_option,
                            input=query4,
                        )
                        
                        # Stream or save the response as needed
                        # For demonstration, let's assume we save then provide a link for downloading
                        audio_file_path = "output.mp3"
                        response.stream_to_file(audio_file_path)
                        
                        # Display audio file to download
                        st.audio(audio_file_path, format='audio/mp3')
                        st.success("Conversion successful!")
                    
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some text to convert.")
    
        topic_input = st.text_area("Topic Input (Example: Checking 4 sensors on follow me trail and 2 are flickering one is acting correctly and the fourth one just stays on.  Also, the air smells stale and the pond is over flowing.)", placeholder="Enter Discussion Topic...")
        run_button = st.button("Run Analysis")

    if run_button:
        synopsis = crewai_process(topic_input)
        st.text_area("Group Synopsis", value=synopsis)

    """***Sample Resutls***

    For Green Space: Checking 4 sensors on follow me trail and 2 are flickering one is acting correctly and the fourth one just stays on. Also, the air smells stale and the pond is over flowing.

     **HIN score**: 0.4
     **Groundedness score**: 0.5 (2/4 sensors are working correctly)
     **Hallucination score**: 0.8 (1/2 sensors are working correctly)
     **Suggestions on how to fix sensors and what new sensors need to be added**:
     - Fix the flickering sensors by checking the power supply, wiring, cleaning the sensor, checking the ground connection, adjusting the sensitivity, checking for interference, and replacing the sensor if necessary.
     - Fix the overflowing sensor by checking the exposure to excessive stimuli, electrical interference, damage or degradation, incorrect calibration, unsuitable environmental conditions, incorrect wiring or connections, and software or firmware issues.
     - Fix the sensor that stays on constantly by checking the wiring, cleaning the sensor, adjusting the sensitivity, checking for obstructions, replacing the sensor, and checking the control panel.
     - Consider adding a sensor to measure air quality to address the stale air issue.
     - Consider adding a water level sensor to monitor the pond and prevent overflow.
    
    
    """
    

    
with tab5:
    st.header("Residential")
     # Creating columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Displaying the image in the left column
    with col1:
        image = Image.open('./data/resid_image.jpg')
        st.image(image, caption='Residential Living - a four-story residential building within the Green Smart Village, illustrating the smart features like smart meters, water flow sensors, and temperature & humidity sensors, alongside the eco-friendly design and technology-enhanced living experience.')
    
    # Displaying the text above on the right
    with col2:
        
        st.markdown(query5)
    
        # Displaying the audio player below the text
        voice_option6 = st.selectbox(
        'Choose a voice:',
        ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], key='key20'
        )
    
    
        if st.button('Convert to Speech', key='key11'):
                if query5:
                    try:
                        response = oai_client.audio.speech.create(
                            model="tts-1",
                            voice=voice_option6,
                            input=query5,
                        )
                        
                        # Stream or save the response as needed
                        # For demonstration, let's assume we save then provide a link for downloading
                        audio_file_path = "output.mp3"
                        response.stream_to_file(audio_file_path)
                        
                        # Display audio file to download
                        st.audio(audio_file_path, format='audio/mp3')
                        st.success("Conversion successful!")
                    
                            
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some text to convert.")

    
    
    