from serpapi import SerpApiClient
from typing import Dict, Any

load_dotenv()


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
        print(f"Error:{e}")


if __name__ == "__main__":
    # 1. Initialize tool executor
    toolExecutor = ToolExecutor()

    # 2. Register our practical search tool
    search_description = "A web search engine. Use this tool when you need to answer questions about current events, facts and information not found in your knowledge base."
    toolExecutor.registerTool("WebSearch", search_description, search)

    # 3. Print available tools
    print("\n--- Available Tools ---")
    print(toolExecutor.getAvailableTools())

    # 4. Agent's Action call, this time we ask a real-time question
    tool_name = "WebSearch"
    tool_input = "What is NVIDIA's latest GPU model"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- Observation ---")
        print(observation)
    else:
        print(f"Error: Tool named '{tool_name}' not found.")
