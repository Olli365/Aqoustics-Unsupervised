import pandas as pd

# Load the spreadsheet
file_path = r'C:\Users\aliwh\Desktop\Work\Aqoustics-Work\Hope-Spots.xlsx'  # Update this with the actual path to your file
df = pd.read_excel(file_path, engine='openpyxl')

# Convert the dataframe to a list of dictionaries
hope_spots_dict_list = [{row['HOPE SPOTS']: row['LINK']} for index, row in df.iterrows()]

# Print the result
for entry in hope_spots_dict_list:
    print(str(entry) + ',')
