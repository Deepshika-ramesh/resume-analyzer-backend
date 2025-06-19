import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()


HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = os.getenv("HUGGINGFACE_API_URL")

print("🧪 .env loaded manually:", open(".env").read())
print("🔍 API KEY:", HUGGINGFACE_API_KEY)
print("🔍 API URL:", API_URL)

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

@app.post("/analyze-resume/")
async def analyze_resume(request: Request):
    try:
        body = await request.json()
        resume = body.get("resume")
        job_desc = body.get("job_description")

        # Combine resume and job desc into one string for classification
        combined_input = f"Resume: {resume}. Job Description: {job_desc}"

        payload = {
            "inputs": combined_input,
            "parameters": {
                "candidate_labels": ["match", "partial match", "no match"]
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)
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