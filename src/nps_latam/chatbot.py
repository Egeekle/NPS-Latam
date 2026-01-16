import os
import csv
import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

class FlightChatbot:
    def __init__(self, log_file: str = "Data/chatbot_logs.csv"):
        """
        Initialize the Flight Chatbot.
        
        Args:
            log_file (str): Path to the CSV file where logs will be stored.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=self.api_key)
        self.log_file = log_file
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self):
        """Creates the log file with headers if it doesn't exist."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "user_query", "bot_response"])

    def _log_interaction(self, user_query: str, bot_response: str):
        """Logs the user query and bot response to the CSV file."""
        timestamp = datetime.datetime.now().isoformat()
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, user_query, bot_response])

    def respond(self, user_input: str) -> str:
        """
        Generates a response to the user input using the LLM and logs the interaction.
        
        Args:
            user_input (str): The query from the customer.
            
        Returns:
            str: The response from the chatbot.
        """
        # Define the system context
        messages = [
            SystemMessage(content="You are a helpful and polite airline customer service assistant. "
                                  "Your goal is to assist customers with questions about their flight experience, "
                                  "services (wifi, food, comfort), and satisfaction. "
                                  "Keep your answers concise and professional."),
            HumanMessage(content=user_input)
        ]
        
        try:
            response_msg = self.llm.invoke(messages)
            bot_response = response_msg.content
            
            # Log the interaction
            self._log_interaction(user_input, bot_response)
            
            return bot_response
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error processing your request: {str(e)}"
            return error_msg

if __name__ == "__main__":
    # Simple test execution
    try:
        print("Initializing Chatbot...")
        # Assuming run from root, adjust path if running directly from src
        # Construct absolute path for safety during test run
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_path = os.path.join(base_dir, "Data", "chatbot_logs.csv")
        
        bot = FlightChatbot(log_file=log_path)
        
        test_query = "¿Tienen opciones de comida vegetariana en vuelos internacionales?"
        print(f"User: {test_query}")
        
        response = bot.respond(test_query)
        print(f"Bot: {response}")
        print(f"Interaction logged to {log_path}")
        
    except ValueError as val_err:
        print(f"Configuration Error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
