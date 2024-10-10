import google.generativeai as genai
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.protos import Part
from src.utils.template import load_system_instruction, load_user_instruction, load_response_schema
from src.utils.io import convert_json_to_jsonl, load_binary_file, save_json
from src.config.logging import logger
from src.config.setup import config
from typing import List, Dict, Any
import json
import time
import os




OUTPUT_DIR = os.path.join(config.DATA_DIR, 'output')
VALIDATION_DIR = os.path.join(config.DATA_DIR, 'validation')



def create_generation_config(response_schema: Dict[str, Any]) -> GenerationConfig:
    """
    Create a GenerationConfig instance.

    Args:
        response_schema (Dict[str, Any]): The schema for the response.

    Returns:
        GenerationConfig: An instance of GenerationConfig with the specified parameters.
    """
    try:
        logger.info("Creating generation configuration")
        config = GenerationConfig(
            temperature=0.0, 
            top_p=0.0, 
            top_k=1, 
            candidate_count=1, 
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        logger.info("Successfully created generation configuration")
        return config
    except Exception as e:
        logger.error(f"Error creating generation configuration: {e}")
        raise


def create_safety_settings() -> Dict[HarmCategory, HarmBlockThreshold]:
    """
    Create a safety settings dictionary.

    Returns:
        Dict[HarmCategory, HarmBlockThreshold]: A dictionary mapping harm categories to block thresholds.
    """
    try:
        logger.info("Creating safety settings")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
        logger.info("Successfully created safety settings")
        return safety_settings
    except Exception as e:
        logger.error(f"Error creating safety settings: {e}")
        raise


def generate_response(model: genai.GenerativeModel, contents: List[Part], response_schema: Dict[str, Any]) -> Any:
    """
    Generate content using the generative model.

    Args:
        model (GenerativeModel): The generative model to use.
        contents (List[Part]): The contents to be processed by the model.
        response_schema (Dict[str, Any]): The schema for the response.

    Returns:
        Any: The generated response.
    """
    try:
        logger.info("Generating response using the generative model")
        response = model.generate_content(
            contents,
            generation_config=create_generation_config(response_schema),
            safety_settings=create_safety_settings()
        )
        output_json = json.loads(response.text.strip())
        logger.info(f"Response generated: {output_json}")
        logger.info(f"Finish reason: {response.candidates[0].finish_reason}")
        logger.info(f"Safety ratings: {response.candidates[0].safety_ratings}")
        return output_json
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")
        raise  # Re-raise the exception after logging
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise  # Re-raise the exception after logging


def llm_extract(model: genai.GenerativeModel, pdf_parts: Part, output_path: str) -> None:
    """
    Extract information from a PDF using a generative model and save the output.

    Args:
        model (GenerativeModel): The generative model to use for extraction.
        pdf_parts (Part): The PDF parts to be processed.
        output_path (str): The path to save the extracted information.

    Raises:
        Exception: If any error occurs during the extraction process, it is logged and re-raised.
    """
    try:
        logger.info("Starting LLM extraction")
        system_instruction = load_system_instruction(workflow='single_step', step=None)
        user_instruction = load_user_instruction(workflow='single_step', step=None)
        response_schema = load_response_schema(workflow='single_step', step=None)
        # model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
        genai.configure(api_key="Your API Key")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-exp-0827", system_instruction=system_instruction )
        contents = [pdf_parts, user_instruction]
        
        response = generate_response(model, contents, response_schema)
        save_json(response, output_path)
        logger.info("LLM extraction completed successfully")
    except Exception as e:
        logger.error(f"Error in LLM extraction: {e}")
        raise  # Re-raise the exception after logging


def run(file_name: str) -> None:
    """
    Run the extraction process on a specified PDF file.

    Args:
        file_name (str): The name of the PDF file to process.

    Raises:
        Exception: If any error occurs during the process, it is logged and re-raised.
    """
    try:
        logger.info(f"Running extraction for file: {file_name}")
        file_path = os.path.join(config.DATA_DIR, f'docs/{file_name}.pdf')
        pdf_bytes = load_binary_file(file_path)
        # pdf_parts = Part(data=pdf_bytes, mime_type='application/pdf')
        pdf_parts = Part(inline_data={'mime_type': 'application/pdf', 'data': pdf_bytes})
        output_path = os.path.join(OUTPUT_DIR, f'single_step/{file_name}/out.txt')
        start_time = time.time()
        print("I am here")
        # sample_file = genai.upload_file(path=file_path,display_name=file_name)
        # Run the LLM extraction
        llm_extract(genai.GenerativeModel, pdf_parts, output_path)
        
        # Convert the output to JSONL format
        convert_json_to_jsonl(output_path, os.path.join(VALIDATION_DIR, f'generated/single_step/{file_name}.jsonl'), workflow='single_step')
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Extraction process completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in run process: {e}")
        raise  # Re-raise the exception after logging



if __name__ == '__main__':
    file_name = 'United Overseas Bank Limited_report'
    run(file_name)