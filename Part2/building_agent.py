import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from serpapi import SerpApiClient
import re

load_dotenv()

# ReAcr Prompt Template
REACT_PROMPT_TEMPLATE = """
Please note that you are an intelligent assistant capable of calling external tools.

Available tools are as follows:
{tools}

Please respond strictly in the following format:

Thought: Your thinking process, uesd to analyze problems, decompose, tasks and plan the next action.
Action: The action you decide to take, must be in one of the following formats:
- {{tool_name}}[{{tool_input}}]: Call an available tool.
- Finish[final answer]: When you believe you have obtained the final answer.
- When you have collected enough information to answer the user's final question, you must use `Finish[final answer]`after the Action: field to output the final answer.

Now, please start solving the following problem:
Question: {question}
History: {history}
"""


class HelloAgentsLLM:
    """
    A customized LLM client for the book "Hello agent"
    It is used to call any service compatible with the Open interface and uses streaming responses by default.
    """

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        """
        Initialize the client. Prioritize passed paramenters; if not provided, load from environment variables.
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")

        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError(
                "Model ID, API key, and service address must be provided or defined in the .env file."
            )

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, message: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        Call the large language model to think and return its response.
        """
        print(f"🧠Calling {self.model} model...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                stream=True,
            )

            #  Handle streaming response
            print("✅ Large language model response successful:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # Newline after streaming output ends
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ Error occurred when calling LLM API: {e}")
            return None


class ToolExecutor:
    """
    A tool executor responsible for managing and executor tools.
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        Register a new tool in the toolbox.
        """
        if name in self.tools:
            print(f"Warning: Tool '{name}' already exists and will be overwritten.")
        self.tools[name] = {"description": description, "func": func}
        print(f"Tool '{name}' registered.")

    def getTool(self, name: str) -> callable:
        """
        Get a tool's execution function by name
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        Get a formatted description string of all available tools
        """
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )


class ReActAgent:
    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        Run the ReAct angent to answer a questions.
        """
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- Step {current_step} ---")

            # 1. Format prompt
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )
            #  2. Call LLM to think
            message = [{"role": "system", "content": prompt}]
            response_text = self.llm_client.think(message)

            if not response_text:
                print("Error:LLM failed to return a valid response")
                break

            # 3. Parse LLM output
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"Thought: {thought}")

            if not action:
                print("Warning: Failed to parse valid Action, process terminated.")
                break

            # 4. Execute Action
            if action.startswith("Finish"):
                # If it's a Finish instruction, extract the final answer and end
                final_answer = re.match(r"Finish\[(.*)\]", action, re.DOTALL).group(1)
                print(f"🎉Final Answer: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... Handle invalid Action format ...
                self.history.append(f"Error: tool_name or tool_input is invalid")
                continue

            print(f"🎬Action: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"Error: Tool named '{tool_name}' not found."
            else:
                observation = tool_function(tool_input)  # Call real tool

            print(f"👀 Observation: {observation}")

            # Add this round's Action and Observation to history
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # Loop ends
        print("Maximum steps reached, prosecc termunated")
        return None

    def _parse_output(self, text: str):
        """
        Parse LLM output to extract Thought and Action.
        """

        # Thought: match untill Action: or end of text
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: match untill end of text
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        Parse Action string to extract tool name and input.
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


def search(query: str) -> str:
    """
    A practial web search engine tool base on SerpApi.
    It intelligentli parses search results, prioritizing direct answers or knowledge graph information.
    """
    print(f"🔍 Executing [SerpApi] web search: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error:SERPAPI_API_KEY not configured in .env file."

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }
        client = SerpApiClient(params)
        results = client.get_dict()
        # Intelligent parsing: prioritize finding the most direct answer
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # If no direct answer, return summarise of the first three organic results
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"Sorry, no information found about '{query}'."

    except Exception as e:
        return f"Error:{e}"


# --- Client Usage Example ---
if __name__ == "__main__":
    try:
        llm_client = HelloAgentsLLM()
        tool_executor = ToolExecutor()
        search_description = "A web search engine. Use this tool when you need to answer questions about current events, facts and information not found in your knowledge base."
        tool_executor.registerTool("WebSearch", search_description, search)
        re_act = ReActAgent(llm_client, tool_executor)
        question = "Huawei latest phone model and main selling points"
        question = "华为最新的手机是什么，卖点又是什么"
        re_act.run(question=question)

    except Exception as e:
        print(e)
