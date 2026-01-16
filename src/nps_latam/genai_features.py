import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the data structure we want robustly
class TextAnalysis(BaseModel):
    sentiment: str = Field(description="Sentiment of the text: 'Positive', 'Neutral', or 'Negative'")
    intent: str = Field(description="Primary intent: 'Booking', 'Complaint', 'Inquiry', 'Feedback', 'Other'")
    keywords: list[str] = Field(description="List of up to 3 key topics mentioned (e.g. 'Wifi', 'Food', 'Delay')")

def get_llm():
    """Initializes the Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )

def analyze_feedback_batch(texts):
    """
    Analyzes a list of text feedback items using GenAI to extract features.
    Returns a DataFrame with the original text and extracted columns.
    """
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=TextAnalysis)

    prompt = PromptTemplate(
        template="""
        You are an expert data analyst for an airline. 
        Analyze the following customer text and extract structured features.
        
        Text: "{text}"
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    results = []
    print(f"Analyzing {len(texts)} text items with GenAI...")
    
    for text in texts:
        try:
            # For demonstration/cost, analyzing one by one. 
            # In production, batching would be better if API supports it natively or via map operations.
            if not text or not isinstance(text, str):
                results.append({"sentiment": "N/A", "intent": "N/A", "keywords": []})
                continue
                
            output = chain.invoke({"text": text})
            results.append(output)
        except Exception as e:
            print(f"Error analyzing text '{text[:30]}...': {e}")
            results.append({"sentiment": "Error", "intent": "Error", "keywords": []})

    # Create DataFrame from results
    features_df = pd.DataFrame(results)
    features_df['original_text'] = texts
    
    # Simple mapping for Sentiment to numeric if needed
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    features_df['sentiment_score'] = features_df['sentiment'].map(sentiment_map).fillna(0)
    
    return features_df

def generate_synthetic_feedback(row):
    """
    Generates a synthetic feedback string based on structured row data.
    Used for demonstrating feature engineering pipeline when real text is missing.
    """
    # Simple conditional generation logic (simulating what a user *might* write)
    issues = []
    if row.get('Wifi_a_bordo', 5) <= 2:
        issues.append("wifi was the worst")
    if row.get('Comida_Bebida', 5) <= 2:
        issues.append("food was cold")
    if row.get('Limpieza', 5) <= 2:
        issues.append("plane was dirty")
    if row.get('Distancia_Vuelo', 0) > 1000 and row.get('Espacio_Piernas', 5) <= 2:
        issues.append("legroom was terrible for such a long flight")
    
    if not issues and row.get('Satisfaccion') == 'satisfied':
        return "Great flight, everything went smoothly."
    elif not issues:
        return "It was an okay flight, nothing special."
    else:
        return "I was unhappy because " + " and ".join(issues) + "."

if __name__ == "__main__":
    # Test execution
    sample_texts = [
        "The flight was on time but the food was terrible.",
        "I need to change my flight date to next Tuesday.",
        "Great service from the crew, very attentive!",
        "My luggage is lost, can you help?"
    ]
    
    try:
        df = analyze_feedback_batch(sample_texts)
        print("Feature Engineering Results:")
        print(df)
    except Exception as e:
        print(f"Test failed: {e}")
