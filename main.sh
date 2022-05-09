
# This step processes the raw data files to create the processed data for modeling
python extractData.py

# Baseline models
python baselineModels.py

# CNN Modeling
python cnn_triplet.py
