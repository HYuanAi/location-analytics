# A Machine Learning Based Indoor Positioning for IoT System: Data Analysis for Indoor Location Data

The project consists of mainly jupyter notebook files that were used in the process of location analytics. 

## Breakdown of Directories and Files

- Data
    - TrainingData.csv and ValidationData.csv are unmodified and can be downloaded from [Kaggle](https://www.kaggle.com/giantuji/UjiIndoorLoc)
    - AllData.csv: combined from TrainingData.csv and ValidationData.csv
    - UJI_B012_floorplan.png: Image of floorplan of three buildings in Jaume I University, where locations were detected. Retrieved from [Google Maps](https://www.google.com/maps/place/Jaume+I+University/@39.9945872,-0.0776551,15z/data=!4m5!3m4!1s0xd5ffe0fca9b5147:0x1368bf53b3a7fb3f!8m2!3d39.9945711!4d-0.0689003)
    - UJI_B2F3_floorplan.png: Image of Floor 3 of the Ciencia y Tecnologia Building. Retrieved from [Google Maps](https://www.google.com/maps/place/Ciencia+y+Tecnologia,+Avinguda+de+Vicent+Sos+Baynat,+12006+Castell%C3%B3+de+la+Plana,+Castell%C3%B3,+Spain/@39.9923426,-0.0666954,18z/data=!4m12!1m6!3m5!1s0xd5ffe0fca9b5147:0x1368bf53b3a7fb3f!2sJaume+I+University!8m2!3d39.9945711!4d-0.0689003!3m4!1s0xd5ffe0ff0333497:0xdb593066a82b15e2!8m2!3d39.992209!4d-0.0661433)
    - UJI_B2F3_binary.jpg: Binary Transformation of Floor 3 of the Ciencia y Tecnologia Building.
- Data Analysis Report for Indoor Location Data.pdf
- Jupyter Notebook (.ipynb) files and SQL script (.sql) file: The numbers before each file indicate to which section of the report they correspond to. Please refer to the report for details of implementation and discussion of results. 
- Outputs: Contains sub-directories, whose names correspond to sections of the report.
- Python files: Some functions used in the notebooks to aid in analysis.

## Dependencies    

- Data Manage and Transformation: Pandas, Numpy, json
- Data Visualisation: Matplotlib, seaborn
- Analytics & Machine Learning: scikit-learn
- Image Processing: OpenCV, scikit-image
- Graph and Network: networkx
- GIS & geometries: GeoPandas, pyproj, GeoPy, Shapely
- Linear Optimisation: Google OR-Tools 
- Others: itertools, datetime, math

## Usage

Refer to the Data Analysis Report for comprehensive description of the analysis methodologies and outcomes. Refer to individual jupyter notebooks for how they were implemented and for further improvements. 


## Contributors

Ho Yuan Ai (Student ID: 29566061) is the main contributor of the project. 

## Special Thanks

Dr. Tan Chee Keong for supervising the project and providing some useful suggestions. Dr. Soon Lay Ki for overseeing the project progress and guiding us throughout the planning and implementation phase of the project. 