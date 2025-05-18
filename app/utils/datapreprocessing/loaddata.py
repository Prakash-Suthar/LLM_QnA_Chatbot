import pandas as pd 

data_path = r"E:\codified\DDReg\LLM_QnA_Chatbot\assets\dataset.csv"

# Load the CSV
df = pd.read_csv(data_path)

print(df.head())

# Step 2: Drop rows with missing question or answer
df.dropna(subset=['question', 'answer'], inplace=True)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Strip leading/trailing whitespace and remove empty strings
df['question'] = df['question'].str.strip()
df['answer'] = df['answer'].str.strip()

#  Remove rows with empty strings after strip
df = df[(df['question'] != '') & (df['answer'] != '')]

# Drop duplicate questions
df = df.drop_duplicates(subset="question", keep="first").reset_index(drop=True)
# Save cleaned data
df.to_csv(r"E:\codified\DDReg\LLM_QnA_Chatbot\assets\unique_questions.csv", index=False)
#  Optional sanity check – check for very short or gibberish entries
print("\nSuspiciously short entries (if any):")
print(df[df['question'].str.len() < 5])
print(df[df['answer'].str.len() < 5])


