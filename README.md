# DESCRIPTION
This repository is a part of the CSE6242 Project for team-011. The purpose of this is to visualize and analyze natural disaster trends across the US. It integrates data obtained from NOAA and NRI metrics, and a few ML-based projections for 2025, to provide a better view of climate-risks. The project is targeted for general public and authorities to visualize natural disaster trends, and aid in decision making/policy drafting activities.

The application features a colored heat-map of all US states, allowing users to toggle between historic data and future predictions. Users can filter at event-level(hurricanes, storms, etc) and view state-wise and year-wise statistics. The dashboard has interactive charts that provide easier understanding of the impact of each disaster. Under the predictions drop-down, there is an integrated ChatBot API which provides context aware answer to questions around the predicted data and mitigation measures.

The technology used is Python-Flask backend to serve as a layer between the frontend and data, as well as provide the API for the chatbot. Vanilla HTML, CSS and JavaScript is used for rendering, with the support of web technologies like Chart.js and Plotly. Data processing is done using Python, primararily to convert csv into json files to load into the system.


# INSTALLATION
1. Prerequisites:   <br>
   Python 3.8+  <br>
   pip(packet manager for Python) <br>
   Git  <br>
2. Fork and clone the repository <br>
   git clone <forked_repository_url> <br>
   cd cs6242_team11 <br>
3. Install all requirements: pip install -r requirements.txt <br>
4. Get an API Key for Groq: <br>
   Get a free API key from [console.groq.com](https://console.groq.com/). <br>
   Set the env variable: <br>
   On Windows: $env:GROQ_API_KEY = "your_api_key_here" <br>
   On MacOS/Linux: export GROQ_API_KEY="your_api_key_here" <br>
5. Data Setup: <br>
   Currently all required json files are within the datasets folder, so **users can skip this step**. <br>
   Use the DataInstaller.ipynb file to collect and preprocess NOAA data. NRI Data can be found [here](https://hazards.fema.gov/nri/data-resources). <br>
   Run these commands to generate json files in the datasets directory(Reminding again, if datasets contain the necessary 4 json files, this step is not required) <br>
   - python preprocessing/preprocess_noaa_data.py 
   - python preprocessing/preprocess_nri_data.py
   - python preprocessing/preprocess_predictions.py
# EXECUTION
Run 'python serve_dashboard.py' in the terminal, and watch the magic unfold! <br>
A running instance should be hosted at http://localhost:8000

# ONLINE HOSTED INSTANCES
Please check out https://cse6242.online/ for accessing the dashboard.

