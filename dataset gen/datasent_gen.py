import json
from typing import Optional, Dict, Any, List

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from guardrails import Guard

# Define a Guardrails rail string for rubric generation.
# The expected output is a JSON object with a "rubrics" key whose value is a list of strings.
RUBRIC_GEN_RAIL = """
<rail version="0.1">
<output>
  <object>
    <property name="rubrics" type="array" description="A list of rubric criteria for grading student answers. Each criterion must be a clear, concise sentence ending with (1 Mark)">
      <items type="string" />
    </property>
    <required name="rubrics"/>
  </object>
</output>
</rail>
"""

# Initialize Guardrails with the rail string.
guard = Guard.from_rail_string(RUBRIC_GEN_RAIL)

# Initialize the Mistral model on Ollama.
model = OllamaLLM(model="mistral", temperature=0.4)

def guarded_llm_invoke(prompt: str, debug_label: str) -> Optional[Dict[str, Any]]:
    """
    Invokes the LLM with the provided prompt and validates its output using Guardrails.
    The expected JSON must contain a "rubrics" key with a list of rubric item strings.
    """
    raw_llm_output = model.invoke(prompt)
    print(f"\n[RAW LLM OUTPUT - {debug_label}]\n{raw_llm_output}")
    messages = [{"role": "user", "content": prompt}]
    try:
        outcome = guard.parse(llm_output=raw_llm_output, messages=messages)
        print(f"\n[VALIDATED JSON - {debug_label}]\n{outcome.validated_output}")
        return outcome.validated_output
    except Exception as e:
        print(f"\nGuardrails failed to parse JSON for {debug_label}: {e}")
        return None

def generate_rubrics_for_pair(question: str, ideal_answer: str) -> List[str]:
    """
    Generates rubric items for a given question and its ideal answer.
    The prompt instructs the LLM to produce a JSON object with a "rubrics" key,
    where each rubric item is a concise criterion ending with "(1 Mark)".
    
    Expected Output Format:
    ```json
    {
      "rubrics": [
        "Understanding of Agile Principles (1 Mark)",
        "Incremental Feature Rollout (1 Mark)",
        "Customer/Stakeholder Involvement (1 Mark)",
        ...
      ]
    }
    ```
    """
    prompt_template = PromptTemplate(
        template="""
        You are an expert academic evaluator tasked with generating rubric criteria for grading student answers.
        For the following question and its ideal answer, generate a list of rubric items.
        Each rubric item must be a single, clear sentence that explains what a student's answer must demonstrate, and must end with "(1 Mark)".
        
        IMPORTANT RULES:
        - The sum of all rubric item marks must be exactly 5 marks.
        - You DO NOT have to generate 5 rubric items; for example, you can generate 2 items where one is worth 2 marks and the other 3 marks.
        - Assign marks in parentheses at the end of each rubric item.
        
        ### Example:
            Question:
            "What is the role of a prototype program in problem solving?"

            Ideal Answer:
            "A prototype simulates portions of the software product to test and refine designs."

            Generated Rubrics:
            ```json
            {{
              "rubrics": [
                "Clearly explains that a prototype simulates portions of the software product (2 Marks)",
                "Identifies prototyping as a way to test and refine designs before implementation (3 Marks)"
              ]
            }}
        ### End of Example

        Question:
        {question}

        Ideal Answer:
        {ideal_answer}

        Generate Rubrics:

        Expected Output Format:
        ```json
        {{
         "rubrics": [
            "Clearly explains that a prototype simulates portions of the software product (2 Marks)",
            "Identifies prototyping as a way to test and refine designs before implementation (3 Marks)"
          ]
        }}
        """, 
    input_variables=["question", "ideal_answer"] )
            
    prompt = prompt_template.format(question=question, ideal_answer=ideal_answer)
    validated_output = guarded_llm_invoke(prompt, debug_label="Rubric Generation")
    if validated_output and "rubrics" in validated_output:
        return validated_output["rubrics"]
    else:
        return []

def load_file_lines(file_path: str) -> List[str]: 
    """Loads non-empty lines from a file."""
    with open(file_path, "r", encoding="utf-8") as f: 
        return [line.strip() for line in f if line.strip()]

def main(): 
    # Load questions and ideal answers from text files. 
    questions = load_file_lines("questions.txt") 
    ideal_answers = load_file_lines("answers.txt")
    
    if len(questions) != len(ideal_answers):
        print("Warning: The number of questions and ideal answers do not match!")

    all_rubrics = {}

    for q, ans in zip(questions, ideal_answers):
        print(f"\nGenerating rubrics for Question: {q}")
        rubric_items = generate_rubrics_for_pair(q, ans)
        all_rubrics[q] = rubric_items
        if rubric_items:
            print("Rubric Items:")
            for item in rubric_items:
                print(" -", item)
        else:
            print("No rubric items generated.")
        print("-" * 40)

    # Save the generated rubric items to a JSON file.
    with open("generated_rubrics_dataset.json", "w", encoding="utf-8") as outfile:
        json.dump(all_rubrics, outfile, indent=2)

        
if __name__ == "__main__": 
    main()