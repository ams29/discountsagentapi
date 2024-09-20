from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import json
import asyncio
from io import BytesIO
import threading
import logging
from openai import AssistantEventHandler
from typing_extensions import override

# ==================== Configuration ====================

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Initialize OpenAI API with your API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

# Initialize FastAPI
app = FastAPI(
    title="Carrier Discount Analyzer API",
    description="API to process contract discount files and return structured discount information for carriers.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

# Predefined discount data for carriers not uploaded
PREDEFINED_CARRIERS = {
    "UPS": {
        "name": "UPS",
        "totalSpend": "$500 M",
        "serviceTypes": [
            {
                "name": "Ground",
                "discounts": [
                    {"type": "DAS", "value": "35%"},
                    {"type": "EDAS", "value": "55%"},
                    {"type": "Fuel", "value": "85%"},
                    {"type": "Address Correction", "value": "80%"},
                    {"type": "Additional Handling", "value": "75%"},
                    {"type": "Oversize Package", "value": "70%"}
                ]
            },
            # Add other service types as needed
        ]
    },
    "FedEx": {
        "name": "FedEx",
        "totalSpend": "$423 M",
        "serviceTypes": [
            {
                "name": "Ground",
                "discounts": [
                    {"type": "DAS", "value": "40%"},
                    {"type": "EDAS", "value": "60%"},
                    {"type": "Fuel", "value": "88%"},
                    {"type": "Address Correction", "value": "85%"},
                    {"type": "Additional Handling", "value": "82%"},
                    {"type": "Oversize Package", "value": "79%"}
                ]
            },
            {
                "name": "Express",
                "discounts": [
                    {"type": "DAS", "value": "42%"},
                    {"type": "EDAS", "value": "65%"},
                    {"type": "Fuel", "value": "83%"},
                    {"type": "Address Correction", "value": "87%"},
                    {"type": "Additional Handling", "value": "80%"},
                    {"type": "Oversize Package", "value": "78%"}
                ]
            },
            # Add other service types as needed
        ]
    },
    "USPS": {
        "name": "USPS",
        "totalSpend": "$350 M",
        "serviceTypes": [
            {
                "name": "Ground",
                "discounts": [
                    {"type": "DAS", "value": "38%"},
                    {"type": "EDAS", "value": "64%"},
                    {"type": "Fuel", "value": "90%"},
                    {"type": "Address Correction", "value": "80%"},
                    {"type": "Additional Handling", "value": "78%"},
                    {"type": "Oversize Package", "value": "76%"}
                ]
            },
            {
                "name": "Express",
                "discounts": [
                    {"type": "DAS", "value": "43%"},
                    {"type": "EDAS", "value": "62%"},
                    {"type": "Fuel", "value": "81%"},
                    {"type": "Address Correction", "value": "85%"},
                    {"type": "Additional Handling", "value": "80%"},
                    {"type": "Oversize Package", "value": "78%"}
                ]
            },
            # Add other service types as needed
        ]
    },
    "Orchestro": {
        "name": "Orchestro",
        "totalSpend": "$289 M",
        "serviceTypes": [
            {
                "name": "Ground",
                "discounts": [
                    {"type": "DAS", "value": "37%"},
                    {"type": "EDAS", "value": "65%"},
                    {"type": "Fuel", "value": "89%"},
                    {"type": "Address Correction", "value": "81%"},
                    {"type": "Additional Handling", "value": "80%"},
                    {"type": "Oversize Package", "value": "77%"}
                ]
            },
            {
                "name": "Express",
                "discounts": [
                    {"type": "DAS", "value": "41%"},
                    {"type": "EDAS", "value": "63%"},
                    {"type": "Fuel", "value": "82%"},
                    {"type": "Address Correction", "value": "84%"},
                    {"type": "Additional Handling", "value": "79%"},
                    {"type": "Oversize Package", "value": "76%"}
                ]
            },
            # Add other service types as needed
        ]
    }
}

# Pydantic models for response
class Discount(BaseModel):
    type: str
    value: str

class ServiceType(BaseModel):
    name: str
    discounts: list[Discount]

class Carrier(BaseModel):
    name: str
    totalSpend: str
    serviceTypes: list[ServiceType]

class DiscountResponse(BaseModel):
    carriers: list[Carrier]

# ==================== Helper Functions ====================

# Helper function to create and configure the Assistant
async def create_assistant(client, assistant_name, instructions, model="gpt-4o-mini"):
    """
    Creates an Assistant using the OpenAI Assistants API.
    """
    logger.info("Creating assistant...")
    try:
        assistant = await asyncio.to_thread(
            client.beta.assistants.create,
            name=assistant_name,
            instructions=instructions,
            model=model,
            tools=[{"type": "file_search"}],
        )
        logger.info(f"Assistant created with ID: {assistant.id}")
        return assistant
    except Exception as e:
        logger.error(f"Error creating assistant: {e}")
        raise e

# Helper function to upload files to vector store
async def upload_to_vector_store(client, vector_store_name, file_stream):
    """
    Uploads files to a Vector Store using the OpenAI Assistants API.
    """
    logger.info("Creating vector store...")
    try:
        vector_store = await asyncio.to_thread(
            client.beta.vector_stores.create,
            name=vector_store_name
        )
        logger.info(f"Vector store created with ID: {vector_store.id}")
        
        logger.info("Uploading files to vector store...")
        file_batch = await asyncio.to_thread(
            client.beta.vector_stores.file_batches.upload_and_poll,
            vector_store_id=vector_store.id,
            files=[file_stream],
        )
        
        if file_batch.status != "completed":
            logger.error("File upload to vector store failed.")
            raise HTTPException(status_code=500, detail="File upload to vector store failed.")
        logger.info("File uploaded to vector store successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error uploading to vector store: {e}")
        raise e

# Helper function to update assistant with vector store
async def update_assistant_with_vector_store(client, assistant_id, vector_store_id):
    """
    Updates the Assistant to use the specified Vector Store.
    """
    logger.info("Updating assistant with vector store...")
    try:
        assistant = await asyncio.to_thread(
            client.beta.assistants.update,
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        )
        logger.info("Assistant updated with vector store successfully.")
        return assistant
    except Exception as e:
        logger.error(f"Error updating assistant with vector store: {e}")
        raise e

# Helper function to create a thread and attach the file
async def create_thread_with_file(client, thread_content, file_id):
    """
    Creates a Thread and attaches a file for the Assistant to use.
    """
    logger.info("Creating thread with attached file...")
    try:
        thread = await asyncio.to_thread(
            client.beta.threads.create,
            messages=[
                {
                    "role": "user",
                    "content": thread_content,
                    "attachments": [
                        {"file_id": file_id, "tools": [{"type": "file_search"}]},
                    ],
                }
            ]
        )
        logger.info(f"Thread created with ID: {thread.id}")
        return thread
    except Exception as e:
        logger.error(f"Error creating thread: {e}")
        raise e

# Define the EventHandler class to capture the response
class EventHandler(AssistantEventHandler):
    def __init__(self):
        self.response = {}
        self.done_event = threading.Event()

    @override
    def on_text_created(self, text) -> None:
        logger.debug(f"Assistant Text Created: {text}")

    @override
    def on_message_completed(self, message) -> None:
        """
        Called when a message is completed.
        """
        try:
            response_content = message.get('content', '')
            # Basic validation to check if the response starts and ends with curly braces
            if response_content.strip().startswith('{') and response_content.strip().endswith('}'):
                self.response = json.loads(response_content)
                logger.info("Received valid JSON response.")
            else:
                self.response = {}
                logger.error("Response is not valid JSON.")
        except json.JSONDecodeError:
            self.response = {}
            logger.error("Failed to decode JSON from message content.")
        finally:
            self.done_event.set()

    @override
    def on_event(self, event, data) -> None:
        """
        Handles generic events.
        """
        logger.debug(f"Received event: {event}, data: {data}")

    @override
    def on_error(self, error) -> None:
        logger.error(f"Stream encountered an error: {error}")
        self.done_event.set()

# Helper function to run the assistant and extract data
async def run_assistant(client, thread_id, assistant_id, instructions, timeout=180):
    """
    Runs the Assistant on the specified Thread and captures the response.
    """
    event_handler = EventHandler()
    logger.info("Running assistant...")
    
    # Define the streaming function
    def stream_run():
        try:
            client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=assistant_id,
                instructions=instructions,
                event_handler=event_handler,
            )
            logger.info("Stream run completed.")
        except Exception as e:
            # Log the exception
            logger.error(f"Stream run exception: {e}")
            event_handler.done_event.set()
    
    # Run the stream in a separate daemon thread
    threading.Thread(target=stream_run, daemon=True).start()
    
    # Wait until the event is set or timeout
    try:
        logger.info("Waiting for assistant to finish processing...")
        await asyncio.wait_for(asyncio.to_thread(event_handler.done_event.wait), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Assistant run timed out.")
        raise HTTPException(status_code=500, detail="Assistant run timed out.")
    
    logger.info("Assistant run completed.")
    return event_handler.response

# ==================== API Endpoint ====================

@app.post("/process-contract/", response_model=DiscountResponse)
async def process_contract(file: UploadFile = File(...)):
    """
    Processes an uploaded contract discount file and returns structured discount information for carriers.
    """
    logger.info("Received request to process contract.")
    
    # Validate file type
    allowed_content_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]
    if file.content_type not in allowed_content_types:
        logger.warning(f"Unsupported file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    # Read file content
    try:
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes from uploaded file.")
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Error reading uploaded file.")
    
    # Initialize OpenAI client
    client = openai
    
    try:
        # Step 1: Create a new Assistant with File Search Enabled
        assistant = await create_assistant(
            client=client,
            assistant_name="Discount Analyzer Assistant",
            instructions=(
                "You are an expert financial analyst. "
                "Use the file_search tool to extract discount information from the uploaded contract file and return the data in the following JSON format:\n"
                "{\n"
                '  "carrierName": "Carrier Name",\n'
                '  "totalSpend": "Total Spend",\n'
                '  "serviceTypes": [\n'
                '    {\n'
                '      "name": "Service Type Name",\n'
                '      "discounts": [\n'
                '        {"type": "Discount Type", "value": "Discount Value"},\n'
                '        ...\n'
                '      ]\n'
                '    },\n'
                '    ...\n'
                '  ]\n'
                "}"
            ),
            model="gpt-4"  # Ensure this is a correct model name
        )
        
        # Step 2: Upload files and add them to a Vector Store
        file_stream = BytesIO(file_content)
        file_stream.name = file.filename  # Set the filename to preserve the extension
        vector_store = await upload_to_vector_store(client, "Contract_Discounts_Vector_Store", file_stream)
        
        # Step 3: Update the assistant to use the new Vector Store
        assistant = await update_assistant_with_vector_store(client, assistant.id, vector_store.id)
        
        # Step 4: Upload the file to OpenAI and create a thread
        file_upload_stream = BytesIO(file_content)
        file_upload_stream.name = file.filename  # Set the filename to preserve the extension
        message_file = await asyncio.to_thread(
            client.files.create,
            file=file_upload_stream,
            purpose="assistants"
        )
        logger.info(f"File uploaded to OpenAI with ID: {message_file.id}")
        
        thread = await create_thread_with_file(
            client=client,
            thread_content="Extract the discount details from the uploaded contract.",
            file_id=message_file.id,
        )
        
        # Step 5: Create a run and check the output
        extracted_data = await run_assistant(
            client=client,
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please extract the discount details and return them in the specified JSON format."
        )
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    
    # Validate extracted data
    if not extracted_data:
        logger.error("Failed to extract data from the contract.")
        raise HTTPException(status_code=500, detail="Failed to extract data from the contract.")
    
    # Determine which carrier was uploaded
    uploaded_carrier = extracted_data.get("carrierName")
    if not uploaded_carrier:
        logger.warning("Carrier name not found in the contract.")
        raise HTTPException(status_code=400, detail="Carrier name not found in the contract.")
    
    # Normalize carrier name
    uploaded_carrier = uploaded_carrier.strip().title()
    if uploaded_carrier not in ["UPS", "FedEx", "USPS", "Orchestro"]:
        logger.warning(f"Unsupported carrier in the contract: {uploaded_carrier}")
        raise HTTPException(status_code=400, detail="Unsupported carrier in the contract.")
    
    # Predefined carriers list
    all_carriers = ["UPS", "FedEx", "USPS", "Orchestro"]
    
    # Prepare the final JSON
    carriers_json = []
    
    for carrier in all_carriers:
        if carrier.lower() == uploaded_carrier.lower():
            # Use extracted data
            carriers_json.append({
                "name": extracted_data.get("carrierName", carrier),
                "totalSpend": extracted_data.get("totalSpend", "N/A"),
                "serviceTypes": extracted_data.get("serviceTypes", [])
            })
        else:
            # Use predefined data
            predefined = PREDEFINED_CARRIERS.get(carrier)
            if predefined:
                carriers_json.append({
                    "name": predefined["name"],
                    "totalSpend": predefined["totalSpend"],
                    "serviceTypes": predefined["serviceTypes"]
                })
            else:
                # If predefined data is missing
                carriers_json.append({
                    "name": carrier,
                    "totalSpend": "N/A",
                    "serviceTypes": []
                })
    
    logger.info("Returning response with carrier discounts.")
    return {"carriers": carriers_json}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Carrier Discount Analyzer API!"}
