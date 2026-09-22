from llm_call_function import Tools
import json

tool = Tools()
result = tool.search("Coffee")
with open("Part2/data.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
