import json
import os
from typing import Dict, Any
from openai import OpenAI

# -------------------------
# OpenAI Client
# -------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Helper: Safe JSON Extractor
# -------------------------
def safe_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract JSON even if the model returns added text.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return {"error": "Invalid JSON returned by LLM", "raw": text}


# ============================================================
# 1️⃣ Generate Output Schema from Requirement Text
# ============================================================
def generate_output_schema(requirement_text: str) -> Dict[str, Any]:
    prompt = f"""
You are an expert API architect. Convert the user's API requirement into a clean **JSON Schema**.

Requirements:
- Must be valid JSON.
- Only output the schema object, nothing else.
- Use correct JSON Schema types (string, number, boolean, object, array).
- Infer nested objects if needed.

User Requirement:
\"\"\"{requirement_text}\"\"\"

Return ONLY JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return safe_json_from_response(response.choices[0].message.content)


# ============================================================
# 2️⃣ Generate Input Schema Based on Existing API
# ============================================================
def generate_input_schema(api_details: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are an API integration expert. Use the existing API details below to generate a valid JSON Schema 
for its **input request**.

Existing API Details:
{json.dumps(api_details, indent=2)}

Rules:
- Include all required fields.
- Include optional fields.
- Use JSON Schema format.
- Output only pure JSON.

Return ONLY JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return safe_json_from_response(response.choices[0].message.content)


# ============================================================
# 3️⃣ Generate Field-to-Field Mapping
# ============================================================
def generate_mapping(
    custom_output_schema: Dict[str, Any],
    existing_output_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate field mappings from a NEW custom API output schema
    to an EXISTING API output schema.

    This tells the developer:
        - Which existing output fields should populate
          which custom output fields.
        - What transformations (if any) are needed.
    """

    prompt = f"""
You are a senior API integration expert.

Your task:
Map fields from a NEW custom API’s output schema
TO the fields in an EXISTING API’s output schema.

The goal:
Tell the developer exactly which field in the EXISTING API output
should fill which field in the NEW CUSTOM API output.

Output Format (strict JSON):
{{
  "mappings": [
    {{
      "custom_path": "<path in custom_output_schema>",
      "existing_path": "<path in existing_output_schema>",
      "transformation": "<optional transformation or null>"
    }}
  ]
}}

Rules:
- custom_path must point to a field in custom_output_schema.
- existing_path must point to a field in existing_output_schema.
- If a field matches exactly, transformation = null.
- If any format conversion is required, mention it briefly.
- Return ONLY valid JSON.

Custom Output Schema:
{json.dumps(custom_output_schema, indent=2)}

Existing Output Schema:
{json.dumps(existing_output_schema, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return safe_json_from_response(response.choices[0].message.content)
