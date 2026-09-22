from dotenv import load_dotenv
from serpapi import SerpApiClient

load_dotenv()


class Tools:
    def search(self, query: str) -> str:
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
            if (
                "knowledge_graph" in results
                and "description" in results["knowledge_graph"]
            ):
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
