import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("HUGGINGFACE_API_KEY")
api_url = os.getenv("HUGGINGFACE_API_URL")

print("🔍 API KEY:", api_key)
print("🔍 API URL:", api_url)

HEADERS = {"Authorization": f"Bearer {api_key}"}

@app.post("/analyze-resume/")
async def analyze_resume(request: Request):
    try:
        body = await request.json()
        resume = body.get("resume")
        job_desc = body.get("job_description")

        combined_input = f"Resume: {resume}. Job Description: {job_desc}"

        payload = {
            "inputs": combined_input,
            "parameters": {
                "candidate_labels": ["match", "partial match", "no match"]
            }
        }

        response = requests.post(api_url, headers=HEADERS, json=payload)
        print("📤 Sent to Hugging Face:", payload)

        result = response.json()
        print("🔥 Hugging Face raw response:", result)

        if "labels" in result and "scores" in result:
            return {
                "labels": result["labels"],
                "scores": result["scores"]
            }
        else:
            return {
                "raw_response": result,
                "error": "Unexpected structure. Here's what we got."
            }

    except Exception as e:
        print("❌ Internal Server Error:", e)
        return {"error": f"Internal Server Error: {str(e)}"}
