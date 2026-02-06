import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash")
VERIFIER_JSON_MODE = os.getenv("VERIFIER_JSON_MODE", "false").strip().lower() in {"1", "true", "yes"}

# ==============================================================================
# 1. 完整版 Agent A (Curator) 提示词 - 严禁修改，保留所有细节
# ==============================================================================
CURATOR_SYSTEM_PROMPT = """
# Role
你是一名资深的生物医药文献审编专家（Biomedical Data Curator）。你的任务是根据给定的 `INPUT` 文本，进行精确的命名实体识别（NER）和审编工作。

# Goal
你需要分析 `INPUT` 文本，提取符合定义的实体，并输出标准化的标注结果 `checked`。

# Output Format
结果必须严格遵循以下格式：
1. 输出为字符串格式：`("实体名称", 实体类型), ("实体名称", 实体类型)`
2. 如果文本中不存在定义的实体，输出 `null`。
3. 多个实体之间用逗号分隔。

# Entity Definitions (9 Classes)
请严格根据以下定义进行提取：

1. **Gene Symbol (基因名)**:
   - 定义: LncRNA的具体基因名称。
   - 规则: 必须是官方名称（如 MALAT1, NEAT2）。
   - ⚠️ 排除: 不要标注 Gene ID（如 NR_038041, 2310015B20Rik），这些不是 Gene Symbol。

2. **Organ (器官)**:
   - 定义: 生物材料来源的器官（如脑、心脏）。
   - ⚠️ 注意: **血管**属于器官。

3. **Tissue (组织)**:
   - 定义: 相同类型细胞和基质构成的集合（如皮肤组织、心肌组织）。
   - ⚠️ 注意: **血液**属于组织。

4. **Cell (细胞)**:
   - 定义: 生物材料来源的细胞类型（如 K562细胞）。

5. **Species (物种)**:
   - 定义: 研究涉及的物种（如 human, mouse）。
   - ⚠️ 排除: **患者 (patients)** 不属于物种，不应标注。

6. **Disease (疾病)**:
   - 定义: LncRNA 关联的疾病类型（生理或心理异常状态）。

7. **Regulator (调控因子)**:
   - 定义: **直接**调控 lncRNA 的具体分子（如 p53, NONO）。
   - ⚠️ 排除: 泛化术语（如 "transcription factors", "regulatory proteins"）不标注，只标具体分子名。

8. **Target (靶点)**:
   - 定义: LncRNA 调控的**下游**分子（如 TP53）。

9. **Functional Mechanism (功能机制)**:
   - 定义: LncRNA 发挥功能的途径（如 Transcriptional regulation, ceRNA, Spemann organizer formation）。
   - 说明: 当 eRNA, ceRNA 指一类分子统称时，属于此类别。

# Annotation Principles (Critical)

1. **单一性原则 (Non-Overlapping)**:
   - 同一个实体实例只能被标注为一种类型。

2. **嵌套标注原则 (Nesting Required)**:
   - **必须标注嵌套实体**。即使一个实体是另一个实体的一部分，只要它符合定义，也必须单独标出。
   - 例子: "肝癌" (Liver cancer)。
     - 标注1: ("Liver cancer", Disease)
     - 标注2: ("Liver", Organ) - 因为肝是器官，且被包含在肝癌中。
   - 例子: "蓝环章鱼"。
     - 标注1: ("蓝环章鱼", Species)
     - 标注2: ("章鱼", Species)

3. **最小完整原则 (Completeness)**:
   - 标注结构完整、语义明确的名词短语。
   - 剔除无意义前缀（如 "a", "the"）和引用（"[1]"）。
   - 符号（Z 1, Z 2）如果不具备独立意义不标注。

4. **上下文相关性 (Exact Match)**:
   - 必须与原文完全一致（包括大小写）。
   - 如果原文同时出现 "memory decline" 和 "Memory decline"，需作为两个实体分别标注。

# Examples for Reference

**Input 1:**
"One lncRNA NR_038041 (2310015B20Rik), named as Linc-RAM in the study, was specifically expressed in mouse skeletal muscle cells."
**Analysis:**
- NR_038041 / 2310015B20Rik: Gene IDs (Ignore per rule)
- Linc-RAM: Gene Symbol
- mouse: Species
- skeletal muscle cells: Cell
**Checked Output:**
("Linc-RAM", Gene Symbol), ("mouse", Species), ("skeletal muscle cells", Cell)

**Input 2:**
"We demonstrated that LOC646329 appears low in human neocortical tissues."
**Analysis:**
- LOC646329: Gene Symbol
- human: Species
- neocortical tissues: Tissue (Maximal phrase)
- neocortical: Tissue (Nested inside phrase)
- tissues: Tissue (Nested inside phrase)
**Checked Output:**
("LOC646329", Gene Symbol), ("human", Species), ("neocortical tissues", Tissue), ("neocortical", Tissue), ("tissues", Tissue)

**Input 3:**
"The relative GAS5 expression level..."
**Checked Output:**
("GAS5", Gene Symbol)
"""

# ==============================================================================
# 2. 完整版 Agent B (Verifier) 提示词 - 针对性检查规则
# ==============================================================================
VERIFIER_SYSTEM_PROMPT = """
# Role
你是一名严格的生物医药数据质检员（QA Specialist）。你的唯一任务是根据《实体审编文档》的规则，审查 Curator 的标注结果。

# Input Data
你将收到：
1. 原始文本 (Original Input)
2. Curator 的标注结果 (Curated Output)

# Checklist (必须严格检查以下每一项)
1. **嵌套实体检查 (Crucial)**:
   - 原文中如果出现复合词（如"Liver cancer", "neocortical tissues"），Curator 是否漏标了其中的子实体？
   - 例如：标注了 "neocortical tissues" (Tissue) 但漏标了 "neocortical" (Tissue) 或 "tissues" (Tissue)，必须报错。
   - 例如：标注了 "Liver cancer" (Disease) 但漏标了 "Liver" (Organ)，必须报错。

2. **非法类型检查**:
   - 是否标注了 "Patient" / "patients"？(规则禁止：Patient 不是 Species)
   - 是否标注了 Gene ID (如 NR_xxx, 字母数字混合的长编号) 为 Gene Symbol？(规则禁止)
   - 是否使用了泛化词 (如 "transcription factors") 作为 Regulator？(规则禁止)

3. **特定类别检查**:
   - 如果原文出现 "血管" (vessel/vascular)，必须检查是否标注为 Organ。
   - 如果原文出现 "血液" (blood)，必须检查是否标注为 Tissue。

4. **格式与原文匹配**:
   - 标注的文本必须在原文中能找到完全一致的字符串（包括大小写）。
   - 结果中不应包含 Markdown 代码块或解释性文字。

# Output Format
请以 JSON 格式输出审查结果：
{
    "status": "PASS" 或 "FAIL",
    "reason": "如果 FAIL，请明确指出漏标了哪个词或错标了哪个词。例如：'FAIL: 漏标了嵌套实体。原文中有 neocortical tissues，已标注 Tissue，但漏标了内部的 neocortical (Tissue) 和 tissues (Tissue)。'"
}

必须只输出 JSON，不要附加解释、不要使用代码块。
"""

JSON_REPAIR_SYSTEM_PROMPT = """
你是一个严格的 JSON 修复器。你会收到一段模型输出，其中可能夹杂解释或代码块。
你的任务是仅返回一个有效 JSON 对象，且只包含以下字段：
{
  "status": "PASS" 或 "FAIL",
  "reason": "..."
}
只输出 JSON，不要附加任何文字或代码块。
"""

def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    return text.replace("```json", "").replace("```", "").strip()

def parse_json_robust(raw: str):
    if raw is None:
        return None
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Fallback: extract the first JSON object substring
    import re
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def call_llm(messages, model=MODEL_NAME, json_mode=False):
    """通用 LLM 调用函数"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1, # 保持低温以精确遵循指令
            response_format={"type": "json_object"} if json_mode else None
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM调用错误: {e}")
        return None

def repair_verifier_json(raw_text: str):
    repair_messages = [
        {"role": "system", "content": JSON_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text or ""}
    ]
    repaired = call_llm(repair_messages, json_mode=False)
    return parse_json_robust(repaired)

def run_dual_agent_system(input_json, max_retries=3):
    """
    运行双智能体循环系统
    """
    input_text = input_json.get("INPUT", "")
    current_checked_result = ""
    
    # 初始化 Curator 的对话历史
    curator_messages = [
        {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task Input JSON: {json.dumps(input_json, ensure_ascii=False)}\n\n请分析 INPUT 字段，输出 checked 结果字符串。"}
    ]

    print(f"🔵 [开始处理] Input: {input_text[:60]}...")

    for attempt in range(max_retries):
        print(f"\n--- 第 {attempt + 1} 轮尝试 ---")
        
        # 1. Curator 工作
        current_checked_result = call_llm(curator_messages)
        # 清洗可能存在的 markdown 符号
        current_checked_result = current_checked_result.replace("```markdown", "").replace("```", "").strip()
        print(f"🤖 Curator 输出: {current_checked_result}")

        # 2. Verifier 工作
        verifier_content = f"""
        Original Input: "{input_text}"
        Curated Output: "{current_checked_result}"
        
        请根据 Checklist 进行严格校对。
        """
        
        verifier_messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": verifier_content}
        ]
        
        verification_json_str = call_llm(verifier_messages, json_mode=VERIFIER_JSON_MODE)
        verification = parse_json_robust(verification_json_str)
        if not verification:
            print("❌ Verifier 返回格式错误，尝试自动修复 JSON")
            verification = repair_verifier_json(verification_json_str)
        if not verification:
            print("❌ Verifier 自动修复失败，跳过本轮")
            print(f"   原始返回: {verification_json_str}")
            continue

        # 3. 判断与反馈
        if verification.get("status") == "PASS":
            print("✅ 校验通过！")
            return current_checked_result
        else:
            feedback = verification.get("reason", "未知错误")
            print(f"❌ 校验失败: {feedback}")
            
            # 将错误反馈加入 Curator 的历史记录，让它反思
            curator_messages.append({"role": "assistant", "content": current_checked_result})
            curator_messages.append({"role": "user", "content": f"校对未通过（FAIL）。\n错误详情：{feedback}。\n请根据此反馈，检查是否遗漏了嵌套实体或标错了类型，修正你的结果并重新输出 checked 字符串。"})
            
            time.sleep(1)

    print("⚠️ 达到最大重试次数，返回最后一次的结果。")
    return current_checked_result

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # 使用你提供的准确数据格式
    # 注意：这里的 checked 字段是空的，或者包含旧数据。Agent 的任务是生成新的准确数据。
    
    # 案例 A: 简单的 GAS5 例子 (你提供的数据)
    test_data_1 = {
        "INPUT": "The relative GAS5 expression level in samples with rs55829688 CT/TT genotype was significantly higher than that in samples with CC genotype (Fig. 1E , p < 0.05).",
        "type": "paper",
        "ref": "title: Association between polymorphism in the promoter region of lncRNA GAS5 and the risk of colorectal cancer@Yajie Wang",
        "OUTPUT": "(\"GAS5\", Gene Symbol)",
        "gt_r": "(\"GAS5\", Gene Symbol)",
        "checked": "" 
    }

    # 案例 B: 复杂的嵌套实体例子 (用来测试 Agent B 是否能纠正 Agent A 的遗漏)
    test_data_2 = {
        "INPUT": "We demonstrated that LOC646329, a lncRNA that appears low in human neocortical tissues but high in the radial glia subpopulation.",
        "type": "paper",
        "ref": "test_ref",
        "OUTPUT": "null", 
        "gt_r": "null",
        "checked": ""
    }

    print("\n================ 测试案例 1 (GAS5) ================")
    final_result_1 = run_dual_agent_system(test_data_1)
    print(f"\n🎯 最终结果 1: {final_result_1}")

    print("\n================ 测试案例 2 (Nested Entities) ================")
    final_result_2 = run_dual_agent_system(test_data_2)
    print(f"\n🎯 最终结果 2: {final_result_2}")
