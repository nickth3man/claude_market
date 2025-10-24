"""
PocketFlow Cookbook Example: Structured Output (Resume Parser)

Difficulty: ☆☆☆ Dummy Level
Source: https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-structured-output

Description:
Extract structured data from resumes using YAML prompting.
Demonstrates:
- Structured LLM output with YAML
- Schema validation with assertions
- Retry logic for parsing errors
- Index-based skill matching
"""

import yaml
from pocketflow import Node, Flow
# from utils import call_llm  # You need to implement this


class ResumeParserNode(Node):
    """Parse resume text into structured YAML format"""

    def prep(self, shared):
        return {
            "resume_text": shared["resume_text"],
            "target_skills": shared.get("target_skills", [])
        }

    def exec(self, prep_res):
        """Extract structured data from resume"""
        resume_text = prep_res["resume_text"]
        target_skills = prep_res["target_skills"]

        # Create skill list with indexes for prompt
        skill_list_for_prompt = "\n".join(
            [f"{i}: {skill}" for i, skill in enumerate(target_skills)]
        )

        prompt = f"""
Analyze the resume below. Output ONLY the requested information in YAML format.

**Resume:**
```
{resume_text}
```

**Target Skills (use these indexes):**
```
{skill_list_for_prompt}
```

**YAML Output Requirements:**
- Extract `name` (string)
- Extract `email` (string)
- Extract `experience` (list of objects with `title` and `company`)
- Extract `skill_indexes` (list of integers found from the Target Skills list)
- **Add a YAML comment (`#`) explaining the source BEFORE each field**

Generate the YAML output now:
"""

        # Get LLM response
        # response = call_llm(prompt)

        # Placeholder response
        response = """
```yaml
# Extracted from header
name: John Smith

# Found in contact section
email: john.smith@email.com

# Work history section
experience:
  - title: Senior Developer
    company: Tech Corp
  - title: Software Engineer
    company: StartupXYZ

# Skills matching target list
skill_indexes: [0, 2, 5]  # Team leadership, Project management, Python
```
"""

        # Parse YAML from response
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        structured_result = yaml.safe_load(yaml_str)

        # Validate structure
        assert structured_result is not None, "Parsed YAML is None"
        assert "name" in structured_result, "Missing 'name'"
        assert "email" in structured_result, "Missing 'email'"
        assert "experience" in structured_result, "Missing 'experience'"
        assert isinstance(structured_result.get("experience"), list), "'experience' is not a list"
        assert "skill_indexes" in structured_result, "Missing 'skill_indexes'"

        return structured_result

    def post(self, shared, prep_res, exec_res):
        """Store and display structured data"""
        shared["structured_data"] = exec_res

        print("\n=== STRUCTURED RESUME DATA ===\n")
        print(yaml.dump(exec_res, sort_keys=False, allow_unicode=True,
                       default_flow_style=None))
        print("\n✅ Extracted resume information.\n")

        return "default"


# Example usage
def run_parser():
    """Run resume parser demo"""

    # Sample resume text
    sample_resume = """
    JOHN SMITH
    Email: john.smith@email.com | Phone: (555) 123-4567

    EXPERIENCE
    Senior Developer - Tech Corp (2020-Present)
    - Led team of 5 developers
    - Built scalable Python applications
    - Managed multiple projects simultaneously

    Software Engineer - StartupXYZ (2018-2020)
    - Developed web applications
    - Collaborated with cross-functional teams
    - Presented technical solutions to stakeholders

    SKILLS
    - Team Leadership & Management
    - Python, JavaScript, SQL
    - Project Management
    - Public Speaking
    - CRM Software
    - Data Analysis
    """

    # Target skills to match
    target_skills = [
        "Team leadership & management",
        "CRM software",
        "Project management",
        "Public speaking",
        "Microsoft Office",
        "Python",
        "Data Analysis"
    ]

    # Prepare shared store
    shared = {
        "resume_text": sample_resume,
        "target_skills": target_skills
    }

    # Create and run flow
    parser_node = ResumeParserNode(max_retries=3, wait=10)
    flow = Flow(start=parser_node)
    flow.run(shared)

    # Display matched skills
    if "structured_data" in shared:
        found_indexes = shared["structured_data"].get("skill_indexes", [])
        if found_indexes:
            print("\n--- Matched Target Skills ---")
            for index in found_indexes:
                if 0 <= index < len(target_skills):
                    print(f"✓ {target_skills[index]} (Index: {index})")


if __name__ == "__main__":
    run_parser()
