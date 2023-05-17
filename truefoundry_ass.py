from fastapi import FastAPI, HTTPException
import requests
import sys

app = FastAPI()
parser = argparse.ArgumentParser()
parser.add_argument("--hf_pipeline", type=str, help="Hugging Face pipeline name")
parser.add_argument("--model_deployed_url", type=str, help="Deployed endpoint of the Hugging Face model")
args = parser.parse_args()
@app.post("/")
async def convert_and_forward_to_model(input_data: dict,pipeline:str):
    # Convert input_data to V2 inference protocol format
    converted_input_data = convert_to_v2_protocol(input_data,pipeline)

    # Send the converted input to the deployed model
    try:
        response = requests.post(model_deployed_url, json=converted_input_data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return the response received from the model
    return response.json()

def convert_to_v2_protocol(input_data: dict, pipeline: str):
    if pipeline == "zero-shot-classification":
        # Convert input_data to V2 inference protocol for zero-shot classification
        converted_input_data = {
            "inputs": input_data.get("sequences"),
            "parameters": {
                "candidate_labels": input_data.get("candidate_labels"),
            },
        }
    elif pipeline == "object-detection":
        # Convert input_data to V2 inference protocol for object detection
        converted_input_data = {
            "inputs": {
                "image": input_data.get("image"),
            },
            "parameters": {
                "threshold": input_data.get("threshold", 0.5),
            },
        }
    elif pipeline == "text-generation":
        # Convert input_data to V2 inference protocol for text generation
        converted_input_data = {
            "inputs": {
                "text": input_data.get("text"),
            },
            "parameters": {
                "max_length": input_data.get("max_length", 20),
            },
        }
    elif pipeline == "token-classification":
        # Convert input_data to V2 inference protocol for token classification
        converted_input_data = {
            "inputs": input_data.get("text"),
            "parameters": {
                "tags": input_data.get("tags"),
            },
        }
    else:
        raise ValueError("Unsupported pipeline: " + pipeline)

    return converted_input_data

if __name__ == "__main__":
    hf_pipeline = sys.argv[sys.argv.index("--hf_pipeline") + 1]
    model_deployed_url = sys.argv[sys.argv.index("--model_deployed_url") + 1]
    uvicorn.run(app, host="0.0.0.0", port=8000)