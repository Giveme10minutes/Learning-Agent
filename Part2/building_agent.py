import ast
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
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

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner:
    def __init__(self, llm_client:HelloAgentsLLM):
        self.llm_client = llm_client
        
    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划
        """
        prompt = PLANNER_PROMPT_TEMPLATE
        
        # 为了生成计划，我们构建一个简单的消息列表
        message = [{"role":"user","content":prompt}]
        
        print("--- 正在生成计划 ---")
        # 使用流式输出来获取完整计划
        response_text = self.llm_client.think(message=message) or ""
        
        print(f"✅ 计划已生成:\n{response_text}")
        
        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal+eval来安全地执行字符串
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor:
    def __init__(self,llm_client:HelloAgentsLLM):
        self.llm_client = llm_client
        
    def execute(self, question:str,plan:list[str]) ->str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = ""
        print("\n--- 正在执行计划")
        for i,step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}：{step}")
            
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history is history else "无"
                current_step=step
            )
            
            message = [{"role":"user","content":prompt}]
            response_text =self.llm_client.think(message=message) or ""
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")
            
        final_answer = response_text
        return final_answer
        
class PlanAndSolbeAgent:
    def __init__(self,llm_client:HelloAgentsLLM):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client=llm_client
        self.planner=Planner(llm_client)
        self.executor = Executor(llm_client)
        
    def run(self, question:str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题：{question}")
        plan = self.planner.plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
        final_answer = self.executor.execute(question,plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
    
class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """

    def __init__(self):
        """
        初始化一个空列表来存储所有记录。
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        如果不存在，则返回 None。
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None
    
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""


REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent:
    def __init__(self, llm_clien:HelloAgentsLLM, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code


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
