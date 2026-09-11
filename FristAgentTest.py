import requests
import os
from tavily import TavilyClient
from openai import OpenAI
from dotenv import load_dotenv
import re


load_dotenv()


def get_weather(city:str)->str:
    """
    Query real weather information by calling the wttr.in API.
    """
    # API endpoint, we request data in JSON format
    url = f"http://wttr.in/{city}?format=j1"

    try:
        # Make network request
        response = requests.get(url)
        # Check if request was successful
        response.raise_for_status()
        # Parse returned JSON data
        data = response.json()

        # Extract current weather conditions
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        # Format as natural language return
        return f"{city} current weather: {weather_desc}, temperatue {temp_c} degress Celsius."

    except requests.exceptions.RequestException as e:
        # Handle network errors
        return f"Error: Network problem encontered while querying weather data. - {e}"
    except (KeyError, IndexError) as e:
        # Handel data parsing errors
        return f"Error: Failed to parse weather data, citi name may be invalid. - {e}"
    

def get_attraction(city:str, weather:str)->str:
    """
    Base on city and weather, use Tavily API to search for recommended tourist attractions.
    """
    # 1.Read API key from envirnoment variable
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: Tavily API key not found."
    
    # 2.Initialize TavilyClient
    tavily = TavilyClient(api_key=api_key)
    
    # 3.Construct a precise query
    query = f"'{city}' most worthwhile tourist and reasons  in '{weather} weather'"
    
    try:
        # 4.call Tavily API
        response = tavily.search(query=query,search_type="basic",include_answer=True)
        
        # 5.Tavily's returned results are already very clean and can be used directly
        # response['answer'] is a summary answer based on all search results
        if response.get("answer"):
            return response["answer"]
        
        # If there's no comprehensive answer, format raw results
        formatted_results = []
        for result in response.get("results",[]):
            formatted_results.append(f"- {result['title']} : {result['content']}")
            
        if not formatted_results:
            return "Sorry, no relevant results tourist attractions recommendations found."
        
        return "Based on search, found the following information for you:\n" + "\n".join(formatted_results)
    
    except Exception as e:
        # Handle Tavily API errors
        return f"Error: Problem occurred when excutting Tavily search. - {e}"
        
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}

        
class OpenAICompatibleClient:
    """
    A client for calling any LLM service compatible with the OpenAI interface.
    """
    
    def __init__(self, model:str, api_key:str, base_url:str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def generate(self, prompt:str, system_prompt:str) ->str:
        """Call LLM API to generate response"""
        print("Calling LLM...")
        try:
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                stream=False,
            )
            answer = response.choices[0].message.content
            print("Large language model responseded successfully.")
            return answer
        except Exception as e:
            print(f"Error: Large language model failed to generate response. - {e}")
            return "Error: Large language model failed to generate response."
     
     
     
AGENT_SYSTEM_PROMPT = """
You are an intelligent travel assistant. Your task is to analyze user requests and use available tools to solve problems step by step.

# Available Tools:
- `get_weather(city: str)`: Query real-time weather for a specified city.
- `get_attraction(city: str, weather: str)`: Search for recommended tourist attractions based on city and weather.

# Output Format Requirements:
Each response must strictly follow this format, containing one Thought-Action pair:

Thought: [Your thinking process and next step plan]
Action: [The specific action you want to execute]

Action format must be one of the following:
1. Call a tool: function_name(arg_name="arg_value")
2. Finish task: Finish[final answer]

# Important Notes:
- Output only one Thought-Action pair each time
- Action must be on the same line, do not break lines
- When you have collected enough information to answer the user's question, you must use Action: Finish[final answer] format to end

Let's begin!
"""


# --- 1. Configure LLM client ---
# Please replace this with the corresponding credentials and address for the service you use   
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. Initialize ---
user_prompt = "Hello, please help me check the today's weather in Berjing, and then recommend a suitable tourist attraction based on the weather."
prompt_history = [f"User request: {user_prompt}"]

print(f"User input: {user_prompt}\n" + "="*40)

# --- 3. Start agent loop ---
for i in range(5):
    print(f"--- Loop {i+1} ---\n")
    
    # 3.1. Build Prompt
    full_prompt = "\n".join(prompt_history)
     # 3.2. Call LLM for thinking
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # Truncate extra Thought-Action pairs that the model may generate
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', 
                    llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("Truncated extra Thought-Action pairs")
    print(f"Model output:\n{llm_output}\n")
    
    # 3.3. Parse and execute action
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "Error: No action found. Please explicitly use Action: finish(...) or other actions."
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"Task completed, final answer: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"Error: Undefined tool '{tool_name}'"
        
    # 3.4. Record observation_str = f"Observation: {observation}"
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
