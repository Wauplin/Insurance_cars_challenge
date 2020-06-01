
## Load data

```
from constants import DATA_PATH
from utils.data_processing import load_data, train_test_split

# Load raw dataframe
df = load_data(DATA_PATH)

# Add a train/test column to dataframe
df = train_test_split(df)
```

## Train VehiculeModelAggregator

```

from src.vehicule_model_aggregator import VehiculeModelAggregator

# Built mapping from submodel list
vma = VehiculeModelAggregator.from_series(df["Blind_Submodel"], threshold=1000)

# Map aggregated car model
df["Aggregate_Car_Model"] = vma.map(df["Blind_Submodel"])

# Get mapping for a make/model/submodel
aggregate_car_model = vma.model_mapping['Y']
aggregate_car_model = vma.model_mapping['Y.29']
aggregate_car_model = vma.model_mapping['Y.29.0']
```

## Train DamagePreprocessor
```
from src.damage_preprocessor import DamagePreprocessor

# Build preprocessor (will build VehiculeModelAggregator itself)
preprocessor = DamagePreprocessor.from_dataframe(df)

# Preprocess dataframe for training/inference
preprocessed_df = preprocessor.preprocess(df)
```

## Train DamagePredictor
```

from src.damage_predictor import DamagePredictor

# Train models from initial DataFrame (will build DamagePreprocessor itself)
predictor = DamagePredictor.from_dataframe(df)

# Get train and test scores
# -> precision, recall, f1-score, confusion_matrix for classification task
# -> MSE + datapoints from regression task
scores = predictor.scores()

# Predict damage for a given vehicule
pred = predictor.predict_vehicule(2000, "AAA.AAA.AAA")
# pred['has_damage'] -> boolean value
# pred['damage_amount'] -> float value
```

## Save trained objects
```
# Save mapping in a config file
# JSON format is used for readability
from constants import VMA_CONFIG_PATH
vma.to_config_file(VMA_CONFIG_PATH)

# Save preprocessor in a config file
# JSON format is used for readability
from constants import DAMAGE_PREPROCESSOR_CONFIG_PATH
preprocessor.to_config_file(DAMAGE_PREPROCESSOR_CONFIG_PATH)

# Save models in a config file
# Pickle format is used for practicality
from constants import DAMAGE_PREDICTOR_PICKLE_PATH
predictor.to_pickle(DAMAGE_PREDICTOR_PICKLE_PATH)
```

## Load trained objects
```
# Load mapping from config file
vma = VehiculeModelAggregator.from_config_file(VMA_CONFIG_PATH)

# Load preprocessor from config file
preprocessor = DamagePreprocessor.from_config_file(DAMAGE_PREPROCESSOR_CONFIG_PATH)

# Load models from config file
predictor = DamagePredictor.from_pickle(DAMAGE_PREDICTOR_PICKLE_PATH)
```

## Load and save data from/to MySQL
```
"""
Depending on the use case, the SQL code could change.
Especially, we need to chose a policy if the table already exists.
Here, I chose to replace the table entirely. We might want to append the data
to the existing table or even raise an error if already exists.
"""

from utils.sql import df_to_sql, sql_to_df

# Save raw dataframe to MySQL
df_to_sql(table_name='podatki', df=df)

# Save preprocessed dataframe to MySQL
df_to_sql(table_name='features', df=preprocessed_df)

# Load raw dataframe from MySQL
df_from_sql = sql_to_df(table_name='podatki')

# Load preprocessed dataframe from MySQL
preprocessed_df_from_sql = sql_to_df(table_name='features')

# Drop all tables /!\ very unsafe /!\
from utils.sql import db_connection, drop_all_tables
drop_all_tables(db_connection)
```
