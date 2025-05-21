import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Optional
import os
import re

# --- Define state schema (optional for your use case) ---
class CompareState(TypedDict):
    resume: str
    job: str
    output: Optional[str]

# --- Setup model ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=openai_api_key
)

# --- Define tool functions ---
def extract_resume_info(resume: str) -> str:
    """Extracts key information from a resume."""
    return f"{resume[:1000]}"  # Optionally do NLP here

def extract_jd_info(jd: str) -> str:
    """Extracts key information from a job description."""
    return f"{jd[:1000]}"  # Optionally do NLP here

def match_resume_to_jd(resume_info: str, jd_info: str) -> str:
    """Scores the match between the resume and the job description from 0 to 100."""
    prompt = (
        "You are an AI hiring assistant. Based on the resume info and job description, "
        "return a percentage score (0–100) indicating how well the resume matches the job. "
        "Then explain your reasoning. Respond ONLY in the format:\n\n"
        "'MATCH SCORE: <score>% - <reason>'\n\n"
        f"Resume Info:\n{resume_info}\n\nJob Description:\n{jd_info}"
    )
    result = model.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()

# --- Create LangGraph agents ---
resume_agent = create_react_agent(
    model=model,
    tools=[extract_resume_info],
    name="resume_agent",
    prompt="You extract key details from a resume."
)

jd_agent = create_react_agent(
    model=model,
    tools=[extract_jd_info],
    name="jd_agent",
    prompt="You extract key details from a job description."
)

match_agent = create_react_agent(
    model=model,
    tools=[match_resume_to_jd],
    name="match_agent",
    prompt="You determine if the candidate is a match for the job based on extracted details."
)

# --- Supervisor prompt ---
supervisor_prompt = (
    "You are a smart hiring supervisor. First, use resume_agent to analyze the resume. "
    "Then use jd_agent to analyze the job description. Finally, use match_agent to determine if the candidate matches the job. "
    "Return a percentage match score and reasoning in the format: 'MATCH SCORE: <score>% - <reason>'."
)

# --- Create the workflow ---
workflow = create_supervisor(
    agents=[resume_agent, jd_agent, match_agent],
    model=model,
    prompt=supervisor_prompt
)

# --- Job description ---
job_post_content = '''
We are looking for a talented React Native Mobile Developer to join our dynamic development team. You will be responsible for building high-performance, scalable, and user-friendly mobile applications for both iOS and Android platforms using React Native.
'''

# --- Main execution ---
if __name__ == "__main__":
    # Load resume text from PDF
    doc = fitz.open("./JeffPhan_MobileDevelop.pdf")

    resume_text = "\n".join(page.get_text() for page in doc)
    # Compile and run workflow
    app = workflow.compile()
    result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"""Does this candidate match this job?
              Resume:
              {resume_text}

              Job Description:
              {job_post_content}
              """
        }
      ]
    })

    # Output match result
    try:
      content = result.content.strip()
    except AttributeError:
      content = str(result).strip()

      match = re.search(r"MATCH SCORE:\s*(\d+)%", content, re.IGNORECASE)
      if match:
          score = int(match.group(1))
          print(f"MATCH SCORE: {score}%")
          print(content)
      else:
          print("Unable to determine match score.")
          print(content)

    # Export result to ./results/{pdfName}.txt
    pdf_path = "./JeffPhan_MobileDevelop.pdf"
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{pdf_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        if match:
            f.write(f"MATCH SCORE: {score}%\n\n")
        f.write(content)
    print(f"Result exported to {output_path}")
