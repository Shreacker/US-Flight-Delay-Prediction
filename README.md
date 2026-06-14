```text
            +---------------------------------------------------------------------------+
            |                                --+--                                      |
            |                                  |                                        |
            |                              .-"""""-.                                    |
            |                            .'_________'.                                  |
            |                           /_ /__   __\ _\                                 |
            |                          ;'-._\_| |_/_.-';                                |
            |         ,----------------|    `-. .-'    |----------------,               |
            |          ``""--..__      |   / .-'._.'-. \   |      __..--""``            |
            |                    `"-.\`_.\/.-'_| |-'\_/._/.-"`                          |
            |                          ____||    '-._.-'    ||____                      |
            |                        .'_ _ _|               |_ _ _'.                    |
            |                        `--------'           `--------'                    |
            |                           FLIGHT DELAY PREDICTION                         |
            +---------------------------------------------------------------------------+
```

# Flight Delay Prediction

This repository contains the code and data pipeline for predicting flight delays.
Please check out our Dashboard on certain visual illustration of our Dataset: [Flight Delay Dashboard](https://shreacker.github.io/US-Flight-Delay-Prediction/flight_dashboard/)

## Getting Started

Follow these instructions to set up the project on your local machine.

### 1. Pull the Repository
Clone the repository to your local machine using Git:
```bash
git clone "https://github.com/Shreacker/US-Flight-Delay-Prediction"
cd "Flight Delay"
```

### 2. Setup Environment
It's recommended to use a virtual environment. You can set it up and install the required dependencies as follows:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv/Scripts/Activate
pip install -r requirements.txt
pip install dvc dvc-gdrive
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Pull the DVC Data
This project uses Data Version Control (DVC) to manage large datasets. To download the data tracked by DVC, run:
```bash
dvc remote modify --local myremote gdrive_client_id "530206577614-fiau6klarp9gcfsqcijueeo0f3vbmh7i.apps.googleusercontent.com"
dvc remote modify --local myremote gdrive_client_secret "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"
```

Before you run `dvc pull`, you must link the data to your Google Account:

1. Open this link in your web browser: [GoogleDrive](https://drive.google.com/drive/folders/1uIADZmd4MbNl26Ih_y53SX9o9GBjJKf-?usp=sharing)  
2. At the top of the screen next to the folder name, click the little **dropdown arrow** (or click **Organize**).  
3. Click **Add** shortcut.  
4. Select **My Drive** and click **Add**.  
5. Once the shortcut is in your Drive, go to your terminal and run `dvc pull`.  

**If you still have any problems with gaining access to the data through `dvc pull`, please manually download the data from my Kaggle:** [US Flight Delay Dataset](https://www.kaggle.com/datasets/williamlewistran/us-flight-delay-dataset)  
**And move the folder to `root`**

### 4. Running the Pipeline
Once the environment is set up and the data is pulled, you can run the main scripts in the following order. Make sure your virtual environment is activated.

#### Feature Engineering
To perform feature engineering on the raw data, run:
```bash
python src/feat-engineering.py
```

#### Training the Model
To train the model on the engineered features, run:
```bash
python src/main.py
```

#### Evaluation
To evaluate the trained model, run:
```bash
python src/eval.py
```

### Provided Utilities/Modules
At the same time, we provide our hand-crafted modules for data preprocessing and utilities for common tasks.
Check out `preprocessing/` and `utils/`

## Dataset
Our dataset will be organized as follows:
```text
data/
|
|_engineered/ # Data after being engineered with additional features
|   |
|   |_train_engineered.csv
|   |_val_engineered.csv
|   |_test_engineered.csv
|
|_processed/ # Processed final data -> Ready for training
|   |
|   |_train.csv
|   |_val.csv
|   |_test.csv
|   
|_raw/
|   |
|   |_post_weather_crawl/ # After crawling weather data from Meteo API
|   |   |
|   |   |_flight_24_weather.csv
|   |   |_flight_25_weather.csv
|   |
|   |_pre_weather_crawl/ # Completely raw data
|       |
|       |_flight_24.csv
|       |_flight_25.csv
|
|_flight_data_2024_data_dictionary.csv
```

## Contributors
The dataset was conducted from 2 sources:
* 2024 Dataset: From Kaggle - [2024 US Flight Delay Dataset](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)
* 2025 Dataset: Crawl directly from [BTS Transtat](https://transtats.bts.gov/)