import pandas as pd
import numpy as np
import re
import nltk
import ssl
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.impute import SimpleImputer
from fuzzywuzzy import process
from fancyimpute import IterativeImputer
import sqlite3

def clean_dataset(dataset_path):
    # Initialize NLTK
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('stopwords')

    # Set stopwords for English
    stop_words = set(stopwords.words('english'))

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Define function to normalize characters and remove stopwords
    def normalize_chars(text):
        lower_text = text.lower()
        no_number_text = re.sub(r'\d+', '', lower_text)
        no_punc_text = re.sub(r'[^\w\s]', '', no_number_text)
        no_wspace_text = no_punc_text.strip()
        lst_text = no_wspace_text.split()
        filtered_text = [word for word in lst_text if word not in stop_words]
        normalized_text = ' '.join([word.capitalize() for word in filtered_text])
        return normalized_text

    # Normalize Column Names
    df.columns = [normalize_chars(col) for col in df.columns]

    # Nesting Columns
    df['Name'] = df['First Name'] + ' ' + df['Middle Name'].fillna('') + ' ' + df['Last Name']
    df.drop(columns=['First Name', 'Middle Name', 'Last Name'], inplace=True)

    # Create a Mapping Dictionary
    mapping_dict = {
        "Admission Number": "admission_number",
        # Rest of the mapping dictionary...
    }
    df.rename(columns=mapping_dict, inplace=True)

    # Cleaning Rows, Step 1: Cleaning 'Admission Number'
    df.drop_duplicates(subset=['admission_number'], inplace=True)
    df.sort_values(by='admission_number', inplace=True)

    # Cleaning Rows, Step 2: Normalize Certain Column Names
    df['name'] = df['name'].apply(normalize_chars)
    df['father_name'] = df['father_name'].apply(normalize_chars)
    df['mother_name'] = df['mother_name'].apply(normalize_chars)

    # Cleaning Rows, Step 3: Clean 'roll number'
    df['roll_number'] = df['roll_number'].apply(lambda x: re.sub(r'\s+|[^0-9.]', '', str(x)))

    # Cleaning Rows, Step 4: Standardising and Validating class

    # Point 1: Check for consistency and remove special characters
    df['class'] = df['class'].str.upper().str.strip()  # Convert to uppercase and remove extra spaces
    df['class'] = df['class'].apply(lambda x: re.sub(r'[^\w\s]', '', x) if pd.notna(x) else x)

    # Point 2: Convert to categorical
    df['class'] = pd.Categorical(df['class'])

    # Point 4: Remove Extra Spaces and standardize format
    df['class'] = df['class'].str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with single space

    # Point 5: Validate and standardize entries
    expected_classes = [f"V {section}" for section in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]
    df['class'] = df['class'].apply(lambda x: x if x in expected_classes else None)  # Filter out unexpected values

    # Replace missing sections with the most occurring class
    most_occuring_class = df[df['class'].notna()]['class'].mode().iloc[0]  # Get the most occurring class excluding missing values
    df['class'] = df['class'].fillna(most_occuring_class)  # Fill missing values

    # Cleaning Rows, Step 5: Standardising the date format to dd/mm/yyyy
    def normalise_date(date_str):
        # Regular expression pattern to match 'dd/mm/yyyy' format
        dd_mm_yyyy_pattern = r'(\d{1,2})/(\d{1,2})/(\d{2,4})'
        
        # Regular expression pattern to match 'day month year' format
        day_month_year_pattern = r'(\d{1,2})\s+(\w+)\s+(\d{2,4})'
        
        # Extracting date, month, and year from the date string for 'dd/mm/yyyy' format
        match_dd_mm_yyyy = re.match(dd_mm_yyyy_pattern, date_str)
        if match_dd_mm_yyyy:
            day, month, year = match_dd_mm_yyyy.groups()
            # Pad single-digit day and month with leading zero
            day = day.zfill(2)
            month = month.zfill(2)
            # Rearranging date to the desired format
            if len(year) == 2:
                year = '20' + year  # Assuming all years are in the 21st century
            return f'{day}/{month}/{year}'
        
        # Extracting date, month, and year from the date string for 'day month year' format
        match_day_month_year = re.match(day_month_year_pattern, date_str)
        if match_day_month_year:
            day, month, year = match_day_month_year.groups()
            # Dictionary to map month names to their corresponding numbers
            month_mapping = {
                'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
                'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
            }
            # Normalising month name to its corresponding number
            month_num = month_mapping.get(month[:3].lower())
            if month_num:
                month = month_num.zfill(2)
            # Pad single-digit day with leading zero
            day = day.zfill(2)
            # Rearranging date to the desired format
            if len(year) == 2:
                year = '20' + year  # Assuming all years are in the 21st century
            return f'{day}/{month}/{year}'
        
        # Return None if the date string does not match any of the expected formats
        return None

    # Apply the function to the 'date of birth' column
    df['date_of_birth'] = df['date_of_birth'].apply(normalise_date)


    # Cleaning Rows, Step 6: Calculate the age based on the date of birth
    from datetime import datetime

    def calculate_age(date_of_birth):
        # Convert date_of_birth to a datetime object
        dob = datetime.strptime(date_of_birth, '%d/%m/%Y')
        
        # Get the current date
        current_date = datetime.now()
        
        # Calculate the difference in years
        age = current_date.year - dob.year
        
        # Adjust age if the birthday hasn't occurred yet this year
        if current_date.month < dob.month or (current_date.month == dob.month and current_date.day < dob.day):
            age -= 1
        
        return age

    # Apply the function to the 'date of birth' column
    df['age'] = df['date_of_birth'].apply(calculate_age)

    # Cleaning Rows, Step 7: Clean address column and unnest it into address and pin_code column
    # Remove multiple spaces in the address column and replace with a single space
    df['address'] = df['address'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['address'] = df['address'].str.title()

    # Extract pin code from the address column (including whitespace)
    df['pin_code'] = df['address'].str.extract(r'(Bhopal)\s*\d{6}', expand=False)

    # Remove everything after "Bhopal" from the address column
    df['address'] = df['address'].str.replace(r'(Bhopal).*$', r'\1', regex=True)

    # Read the pin code dataset containing location and pin code
    pin_code_data = pd.read_csv('pin_code_dataset.csv')

    # Iterate over each row in the main dataset
    for index, row in df.iterrows():
        if row['pin_code']:
            # Extract location from the address before 'Bhopal'
            location = row['address'].split('Bhopal')[0].strip()
        
            # Perform fuzzy matching to find the closest matching location in the pin code dataset
            match = process.extractOne(location, pin_code_data['location'])
            
            # If a match is found and the score is above the threshold, update the pin code in the main dataset
            if match[1] >= 80:  # Adjust the threshold as needed
                correct_pin_code = pin_code_data.loc[pin_code_data['location'] == match[0], 'pin_code'].values
                df.at[index, 'pin_code'] = correct_pin_code[0]

    # Move pin_code column next to address column
    address_column_index = df.columns.get_loc('address')
    df.insert(address_column_index + 1, 'pin_code', df.pop('pin_code'))


    # Cleaning Rows, Step 8: Clean mobile number
    # Remove non-numeric characters from mobile numbers
    df['mobile_number'] = df['mobile_number'].apply(lambda x: ''.join(filter(str.isdigit, str(x))))

    # Remove all white spaces from mobile numbers
    df['mobile_number'] = df['mobile_number'].str.replace(' ', '')

    # Remove leading zeros from mobile numbers
    df['mobile_number'] = df['mobile_number'].str.lstrip('0')

    # Ensure mobile numbers are 10 digits long
    df['mobile_number'] = df['mobile_number'].apply(lambda x: x[-10:] if len(x) > 10 else x)

    # Identify duplicate mobile numbers with different father names or mother names
    mask = df.duplicated(subset=['mobile_number'], keep=False) & ~(df.duplicated(subset=['mobile_number', 'father_name'], keep=False) & df.duplicated(subset=['mobile_number', 'mother_name'], keep=False))

    # Replace duplicate values with None
    df.loc[mask, 'mobile_number'] = None

    # Cleaning Rows, Step 9: Clean weight
    # Remove whitespaces from the weight' column
    df['weight'] = df['weight'].str.replace(r'\s+', '', regex=True)

    # Clean 'weight' column
    df['weight'] = df['weight'].str.rstrip('kg')  # Remove 'kg' sign at the end of values
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')  # Convert to numeric, coerce errors to NaN

    # Clip values to range [25, 55]
    df['weight'] = df['weight'].clip(lower=25, upper=55)

    # Handle missing values using SimpleImputer with mean strategy
    imputer = SimpleImputer(strategy='mean')
    df['weight'] = imputer.fit_transform(df[['weight']])

    # Round off to whole numbers
    df['weight'] = df['weight'].round().astype(int)

    # Add 'kg' sign at the end of values
    df['weight'] = df['weight'].astype(str) + 'kg'


    # Cleaning Rows, Step 10: Clean height
    # Define a function to convert height to cm
    def convert_height_to_cm(height):
        if pd.isna(height) or height == '' or len(height.strip()) == 0:  # Handle missing or empty strings
            return np.nan
        elif "'" in height:  # Convert feet and inches to cm
            try:
                # Regular expression pattern to match feet and inches format
                pattern = r'(\d+)\'(\d+)\"'
                # Match the pattern in the string
                match = re.match(pattern, height)
                if match:
                    feet, inches = map(int, match.groups())
                    # Convert feet and inches to centimeters
                    cm = feet * 30.48 + inches * 2.54
                    return cm
                else:
                    return None  # Return None if the input string does not match the expected format
            except ValueError:
                return np.nan  # Return NaN for invalid height values
        elif 'cm' in height:  # Convert 'cm' to cm
            try:
                return float(height.strip(' cm'))
            except ValueError:
                return np.nan  # Return NaN for invalid height values
        elif 'm' in height:  # Convert meters to cm
            try:
                return float(height.strip(' m')) * 100  # Convert to float and multiply by 100
            except ValueError:
                return np.nan  # Return NaN for invalid height values
        elif '.' in height:  # Convert unitless value with decimal to meters
            try:
                return float(height) * 100  # Convert to float and multiply by 100
            except ValueError:
                return np.nan  # Return NaN for invalid height values
        else:  # Assume unitless value without decimal is in cm
            try:
                return int(height)  # Convert to integer
            except ValueError:
                return np.nan  # Return NaN for invalid height values


    # Remove whitespaces from the 'height' column
    df['height'] = df['height'].str.replace(r'\s+', '', regex=True)

    # Clean the 'height' column
    df['height'] = df['height'].apply(convert_height_to_cm)

    # Clip values to range [120, 160]
    df['height'] = df['height'].clip(lower=120, upper=160)

    # Handle missing values using SimpleImputer with median strategy
    imputer = SimpleImputer(strategy='median')
    df['height'] = imputer.fit_transform(df[['height']])

    # Round off to whole numbers
    df['height'] = df['height'].round().astype(int)

    # Add 'cm' sign at the end of values
    df['height'] = df['height'].astype(str) + 'cm'

    # Cleaning Rows, Step 11: Clean attendance
    # Remove whitespaces from the 'attendance' column
    df['attendance'] = df['attendance'].str.replace(r'\s+', '', regex=True)

    # Remove '%' sign at the end of values
    df['attendance'] = df['attendance'].str.rstrip('%')

    # Convert values to numeric, coerce errors to NaN
    df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')

    # Handle missing values and values greater than 100 using SimpleImputer with mode strategy
    imputer = SimpleImputer(strategy='most_frequent')
    df['attendance'] = imputer.fit_transform(df[['attendance']])

    # Clip values to range [75, 100]
    df['attendance'] = df['attendance'].clip(lower=75, upper=100)

    # Round off to whole numbers
    df['attendance'] = df['attendance'].round().astype(int)

    # Add '%' sign again after rounding off
    df['attendance'] = df['attendance'].astype(str) + '%'

    # Cleaning Rows, Step 12: Clean physical_health and communication_skills
    # Convert all values in the physical_health and communication_skills column to uppercase
    df['physical_health'] = df['physical_health'].str.upper()
    df['communication_skills'] = df['communication_skills'].str.upper()

    # Remove whitespaces from the columns
    df['physical_health'] = df['physical_health'].str.strip()
    df['communication_skills'] = df['communication_skills'].str.strip()

    # Remove special characters and numeric values, retain only alphabetic characters
    df['physical_health'] = df['physical_health'].str.replace(r'[^a-zA-Z]', '', regex=True)
    df['communication_skills'] = df['communication_skills'].str.replace(r'[^a-zA-Z]', '', regex=True)

    # Retain only the first character if it's a valid grade, otherwise consider it as a missing value
    valid_grades = ['A', 'B', 'C', 'D', 'E']
    df['physical_health'] = df['physical_health'].apply(lambda x: x[0] if isinstance(x, str) and len(x) == 1 and x in valid_grades else np.nan)
    df['communication_skills'] = df['communication_skills'].apply(lambda x: x[0] if isinstance(x, str) and len(x) == 1 and x in valid_grades else np.nan)

    # Handle missing values in 'physical_health' column using backward fill (backfill)
    df['physical_health'] = df['physical_health'].fillna(method='bfill')

    # Handle missing values in 'communication_skills' column using forward fill (ffill)
    df['communication_skills'] = df['communication_skills'].fillna(method='ffill')

    # Cleaning Rows, Step 13: Clean subjects column
    # Selecting multiple columns and creating a new DataFrame
    dmf = df[['admission_number','english', 'hindi', 'mathematics', 'science', 'social_science','sanskrit']]

    # Remove white spaces, special characters, and alphabets from all columns
    dmf = dmf.apply(lambda x: re.sub(r'\s+|[^0-9.]', '', str(x)))

    # Convert columns to numeric data type
    dmf=dmf.apply(pd.to_numeric, errors='coerce')

    # Replace values greater than 100 and less than 33 with NaN using clip except admission_number
    dmf[['english', 'hindi', 'mathematics', 'science', 'social_science','sanskrit']] = dmf[['english', 'hindi', 'mathematics', 'science', 'social_science','sanskrit']].clip(lower=33, upper=100)

    # Sort the DataFrame index in ascending order
    dmf = dmf.sort_index()

    # Linear interpolation for column 'english'
    dmf['english'] = dmf['english'].interpolate(method='linear')

    # Nearest neighbor interpolation for column 'hindi'
    dmf['hindi'] = dmf['hindi'].interpolate(method='nearest')

    # Piecewise interpolation for column 'mathematics' (custom implementation)
    dmf['mathematics'] = dmf['mathematics'].interpolate(method='linear', limit_direction='forward')

    # Polynomial interpolation for column 'science'
    df['science'] = df['science'].interpolate(method='polynomial', order=2, limit_direction='both', limit_area='inside')

    # Spline interpolation for column 'social_science' in the sorted DataFrame
    dmf['social_science'] = dmf['social_science'].interpolate(method='spline', order=2, limit_direction='both', limit_area='inside')

    # Multiple imputation for column 'sanskrit'
    from fancyimpute import IterativeImputer

    # Create an IterativeImputer object
    imputer = IterativeImputer(max_iter=10,random_state=0)

    # Impute missing values in the 'sanskrit' column
    dmf['sanskrit'] = imputer.fit_transform(dmf[['sanskrit']])

    # Round off all values in dmf to whole numbers
    dmf = dmf.round()

    # Sort by Admission number
    dmf=dmf.sort_values(by='admission_number', ascending=True)

    # substitute back to original data frame
    df[['admission_number','english', 'hindi', 'mathematics', 'science', 'social_science','sanskrit']]=dmf[['admission_number','english', 'hindi', 'mathematics', 'science', 'social_science','sanskrit']]


    # Cleaning Rows, Step 14: Cleaning 'total_marks', 'percentage', 'final_grade', 'result', 'rank', 'remark'
    # Remove whitespace and special characters from the specified columns
    columns_to_clean = ['total_marks', 'percentage', 'final_grade', 'result', 'rank', 'remark']
    for col in columns_to_clean:
        df[col] = df[col].str.replace(r'\s+', '', regex=True)  # Remove whitespace
        df[col] = df[col].str.replace(r'[^a-zA-Z0-9]+', '', regex=True)  # Remove special characters
        
        
    # Remove digits from 'final_grade', 'result', 'remark'
    df['final_grade'] = df['final_grade'].str.replace(r'\d+', '')
    df['result'] = df['result'].str.replace(r'\d+', '')
    df['remark'] = df['remark'].str.replace(r'\d+', '')

    # Remove alphabets from 'rank', 'total_marks', 'percentage'
    df['rank'] = df['rank'].str.replace(r'[a-zA-Z]+', '')
    df['total_marks'] = df['total_marks'].str.replace(r'[a-zA-Z]+', '')
    df['percentage'] = df['percentage'].str.replace(r'[a-zA-Z]+', '')


    # Calculate total_marks
    df['total_marks'] = df[['english', 'hindi', 'mathematics', 'science', 'social_science', 'sanskrit']].sum(axis=1)

    # Calculate percentage and round off to two decimal places
    df['percentage'] = (df['total_marks'] / 600) * 100
    df['percentage'] = df['percentage'].round(2)

    # Define a function to assign grades based on percentage
    def assign_grade(percentage):
        if percentage >= 90:
            return 'A+'
        elif 80 <= percentage < 90:
            return 'A'
        elif 70 <= percentage < 80:
            return 'B+'
        elif 60 <= percentage < 70:
            return 'B'
        elif 50 <= percentage < 60:
            return 'C+'
        elif 40 <= percentage < 50:
            return 'C'
        elif 33 <= percentage < 40:
            return 'D'
        else:
            return 'E'

    # Apply the function to create the grade column
    df['final_grade'] = df['percentage'].apply(assign_grade)


    # Check if the 'result' column has any missing values or values other than 'Pass'
    if df['result'].isnull().any() or not all(df['result'] == 'Pass'):
        # Replace missing values or values other than 'Pass' with 'Pass'
        df.loc[df['result'].isnull() | (df['result'] != 'Pass'), 'result'] = 'Pass'




    # First, sort the DataFrame by 'class', 'total_marks', and 'roll_number'
    df.sort_values(by=['class', 'total_marks', 'roll_number'], ascending=[True, False, True], inplace=True)

    # Then, use the rank function to assign ranks
    df['rank'] = df.groupby('class')['total_marks'].rank(method='first', ascending=False)

    # Convert rank to integer type
    df['rank'] = df['rank'].astype(int)


    # Define a dictionary mapping final grades to remarks
    grade_remarks = {
        'A': 'Excellent',
        'B': 'Very Good',
        'C': 'Good',
        'D': 'Satisfactory',
        'E': 'Needs Improvement'
    }

    # Map final grades to remarks and fill the "remark" column
    df['remark'] = df['final_grade'].map(grade_remarks)

    # If there are any missing final grades, fill the remark with 'Needs Grading'
    df['remark'].fillna('Needs Grading', inplace=True)


    # Cleaning Rows, Step 15: Change datatype to reduce memory
    # Identify non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # Convert non-numeric columns to categorical dtype
    df[non_numeric_columns] = df[non_numeric_columns].astype('category')

    # Convert numeric columns (excluding 'admission_number','total_marks','percentage') to int8 dtype
    numeric_columns = df.select_dtypes(include=[np.number]).columns.difference([ 'admission_number','total_marks','percentage'])
    df[numeric_columns] = df[numeric_columns].astype('int8')

    # Cleaning Rows, Step 16: Saving the final Dataset
    df.to_excel('final_dataset.xlsx', index=False)
    df.to_csv('output.csv', index=False)
    df.to_json('output.json', orient='records')
    
    # Connect to a SQLite database
    conn = sqlite3.connect('output.db')
    # Save DataFrame to the database
    df.to_sql('output_table', conn, if_exists='replace', index=False)

    return df
