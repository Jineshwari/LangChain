from google import genai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Ask Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Suggest top food places in Pune"
)

# Print result
print(response.text)