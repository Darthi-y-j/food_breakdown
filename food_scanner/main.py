import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Form
from openai import OpenAI
import base64
from typing import Dict, List
import uvicorn
from pydantic import BaseModel, Field
import re

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize FastAPI app
app = FastAPI(title="Food Identification API")

# Get API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Define system prompt
SYSTEM_PROMPT_IMAGE = """
NEVER FORGET: You should ALWAYS quantify the food ONLY in "gms" and NEVER "g". Also, use "pcs", or "ml". NEVER use any other unit.
1.) MOST IMPORTANT: Identify the EXACT food present in the image.
2.) You have to identify the food and the quantity in which it is present.
3.) If there are multiple foods present, identify all the foods and their EXACT quantities too.
- NEVER EVER miss a SINGLE FOOD item present in the image, NOT EVEN the side dishes and the garnishes.
4.) NOTE: The quantity should ALWAYS be only in (gms), (pcs), or (ml).
STRICTLY quantify each item based on its type: solid countable items like paratha or banana in pcs [basically anything that can be counted], grains/noodles/rice in gms, and liquids like tea, milk, or soup in ml —NO EXCEPTIONS.
VERY VERY IMPORTANT: If there are multiple food items present in the image, quantify the nutrients of each food item separately and NEVER EVER combine the nutrients and quantity of multiple items together.
5.) Your output should be ONLY in this format:

[food_name][quantity][description]  
Total Calories: [value]  
Total Carbs: [value]  
Total Fat: [value]  
Total Protein: [value]  
Total Fiber: [value]  
Example:

[Sambar][150 gms][Made out of lentils, Toor Dal, Tamarind]  
Total Calories: 168  
Total Carbs: 22  
Total Fat: 5  
Total Protein: 7  
Total Fiber: 3  
6.) REMEMBER: Always follow this format and be precise with the nutrient values. Never give random values—only precise, constant approximates (like 40, not 35-40) the same goes to the weight quantification, be EXTREMELY precise on how you quantify ANYTHING!
7.) The description should only comprise of 2 to 3 IMPORTANT ingredients used in the food and NOTHING MORE at all!
8.) If there is curry present in the given image, IDENTIFY what curry it is, instead of simply saying "curry."
NOTE: The description should only comprise of 2 to 3 IMPORTANT ingredients used in the food and NOTHING MORE at all and make the description a proper sentence instead of placing SIMPLE WORDS!!
REMEMBER: The description should sound different for every food, don't use the SAME words everywhere!
ABOVE ALL, Identify EVERY FOOD on the plate, even if it's garnish consider it and give the relevant information about EVERYTHING!

Fish Curry Identification:
Fish curry typically includes pieces of fish (cross-sections or whole) in a gravy-like base that can vary in color (commonly red, orange, or yellow) depending on the spices used.
Fish pieces often have a distinct flaky texture or cross-sectional view, showing white layers of fish meat.
9.) Identify all the individual food items on the plate and their specific names, such as 'Japchae,' 'Kimchi,' or 'Bibimbap,' instead of vague descriptions.
10.) MOST IMPORTANT: Only output the identified foods, descriptions, quantities, and nutrient breakdowns without any extra text or comments.
"""

SYSTEM_PROMPT_EDIT = """
MOST IMPORTANT:
- You MUST incorporate the user's corrections AT ALL COSTS.
- You MUST regenerate the ENTIRE food breakdown WITH ALL THE CORRECTIONS.
NOTE — MANDATORY: Every food item must have its quantity strictly in grams (gms), pieces (pcs), or milliliters (ml). Even if the user says "a handful of almonds" or "a glass of milk" or "two bananas", you MUST convert it to the standard quantity of measurement given!!!!

You will be given a food breakdown that was previously generated, along with a user's correction.
Your task is to regenerate the ENTIRE food breakdown, incorporating the user's corrections AT ALL COSTS.

MOST IMPORTANT RULES:
1. Each food item MUST be on a new line
2. Each food item MUST have its OWN separate nutritional breakdown
3. Food items MUST be separated by a blank line
4. NEVER combine nutritional values of multiple foods
5. NEVER skip the nutritional breakdown for any food item

The output format for EACH food item MUST be EXACTLY:

[food_name][quantity][description]
Total Calories: [value]
Total Carbs: [value]
Total Fat: [value]
Total Protein: [value]
Total Fiber: [value]

For example, if there are two food items:

[Dosa][200 gms][Made from fermented rice and lentil batter]
Total Calories: 380
Total Carbs: 65
Total Fat: 14
Total Protein: 9
Total Fiber: 7

[Mayonnaise][50 gms][Creamy condiment made from eggs and oil]
Total Calories: 360
Total Carbs: 0
Total Fat: 40
Total Protein: 1
Total Fiber: 0

NOTE — MANDATORY: Every food item must have its quantity strictly in grams (gms), pieces (pcs), or milliliters (ml).
If you don't follow the given standard quantity of measurement, the person will go through a heart attack.

IMPORTANT FORMAT RULES:
1. The quantity should ALWAYS be only in (gms), (pcs), or (ml)
2. Each food item MUST start with square brackets: [food][quantity][description]
3. Each nutritional value MUST be a whole number (no decimals)
4. Each food item MUST be followed by its own complete nutritional breakdown
5. There MUST be a blank line between different food items
6. The description should only comprise of 2-3 IMPORTANT ingredients
7. NEVER combine multiple foods in one line
8. NEVER share nutritional values between different foods

REMEMBER:
- Each food item needs its own separate breakdown
- Don't skip any nutritional values
- Don't combine foods or their nutritional values
- Keep foods and their breakdowns separated by blank lines
- Always provide ALL nutritional values for EACH food item
"""

# Define Pydantic models for structured responses
class NutritionInfo(BaseModel):
    calories: int = Field(..., description="Total calories")
    carbs: int = Field(..., description="Total carbohydrates in grams")
    fat: int = Field(..., description="Total fat in grams")
    protein: int = Field(..., description="Total protein in grams")
    fiber: int = Field(..., description="Total fiber in grams")

class FoodItem(BaseModel):
    name: str
    quantity: str
    unit: str
    description: str
    caloriesPerQuantity: float = Field(..., description="Calories per unit quantity")
    carbsPerQuantity: float = Field(..., description="Carbs per unit quantity")
    fatPerQuantity: float = Field(..., description="Fat per unit quantity")
    proteinPerQuantity: float = Field(..., description="Protein per unit quantity")
    fiberPerQuantity: float = Field(..., description="Fiber per unit quantity")
    nutrition: NutritionInfo

class IdentifiedFoodResponse(BaseModel):
    foods: List[FoodItem] = Field(..., description="List of identified food items with nutrition and per quantity breakdowns")

class EditFoodRequest(BaseModel):
    previous_breakdown: str = Field(..., description="The original food breakdown text from the identify endpoint.")
    user_correction: str = Field(..., description="The user's correction to the breakdown.")

def encode_image_to_base64(file_content: bytes) -> str:
    """Convert image bytes to base64 string"""
    return base64.b64encode(file_content).decode('utf-8')

def parse_food_items_with_nutrition(text: str) -> List[FoodItem]:
    """Parse the food items with nutrition from the OpenAI response"""
    logger.debug("Starting to parse food items from text:")
    logger.debug(text)
    
    # Split the response into food items (separated by blank lines)
    items_text = [item.strip() for item in text.split('\n\n') if item.strip()]
    food_items = []
    
    for item_text in items_text:
        try:
            lines = [line.strip() for line in item_text.split('\n') if line.strip()]
            if not lines:
                continue

            # Handle both bracket format and regular format
            first_line = lines[0]
            food_info = None
            
            # Try the [food][quantity][description] format
            bracket_match = re.match(r'\[(.*?)\]\s*\[(\d+)\s*(pcs?|gms?|ml)\]\s*\[(.*?)\]', first_line)
            
            # Try alternative format: "food_name [quantity] [description]"
            if not bracket_match:
                alt_match = re.match(r'(.*?)\s*\[(\d+)\s*(pcs?|gms?|ml)\]\s*\[(.*?)\]', first_line)
                if alt_match:
                    bracket_match = alt_match

            if bracket_match:
                name = bracket_match.group(1).strip()
                quantity = bracket_match.group(2).strip()
                unit = bracket_match.group(3).strip()
                description = bracket_match.group(4).strip()
                
                # Standardize units
                unit = 'pcs' if 'pc' in unit else 'gms' if 'gm' in unit else unit
                
                # Initialize nutrition values
                nutrition_values = {
                    'calories': 0,
                    'carbs': 0,
                    'fat': 0,
                    'protein': 0,
                    'fiber': 0
                }
                
                # Parse nutrition information from remaining lines
                for line in lines[1:]:
                    # Handle both "Total X:" and "X:" formats
                    match = re.search(r'(?:Total\s+)?(\w+):\s*(\d+)', line, re.IGNORECASE)
                    if match:
                        key = match.group(1).lower()
                        value = int(match.group(2))
                        
                        if 'calories' in key or 'cal' in key:
                            nutrition_values['calories'] = value
                        elif 'carbs' in key or 'carbohydrates' in key:
                            nutrition_values['carbs'] = value
                        elif 'fat' in key:
                            nutrition_values['fat'] = value
                        elif 'protein' in key:
                            nutrition_values['protein'] = value
                        elif 'fiber' in key or 'fibre' in key:
                            nutrition_values['fiber'] = value
                
                # Calculate calories per quantity
                try:
                    calories_per_quantity = round(nutrition_values['calories'] / int(quantity), 2)
                    carbs_per_quantity = round(nutrition_values['carbs'] / int(quantity), 2)
                    fat_per_quantity = round(nutrition_values['fat'] / int(quantity), 2)
                    protein_per_quantity = round(nutrition_values['protein'] / int(quantity), 2)
                    fiber_per_quantity = round(nutrition_values['fiber'] / int(quantity), 2)
                except (ValueError, ZeroDivisionError):
                    calories_per_quantity = 0.0
                    carbs_per_quantity = 0.0
                    fat_per_quantity = 0.0
                    protein_per_quantity = 0.0
                    fiber_per_quantity = 0.0
                
                # Create nutrition info
                nutrition_info = NutritionInfo(
                    calories=nutrition_values['calories'],
                    carbs=nutrition_values['carbs'],
                    fat=nutrition_values['fat'],
                    protein=nutrition_values['protein'],
                    fiber=nutrition_values['fiber']
                )
                
                # Add to food items list
                food_item = FoodItem(
                    name=name,
                    quantity=quantity,
                    unit=unit,
                    description=description,
                    caloriesPerQuantity=calories_per_quantity,
                    carbsPerQuantity=carbs_per_quantity,
                    fatPerQuantity=fat_per_quantity,
                    proteinPerQuantity=protein_per_quantity,
                    fiberPerQuantity=fiber_per_quantity,
                    nutrition=nutrition_info
                )
                
                food_items.append(food_item)
                logger.debug(f"Successfully parsed food item: {name}")
            else:
                logger.warning(f"Could not parse food info from line: {first_line}")
                
        except Exception as e:
            logger.error(f"Error parsing food item: {str(e)}")
            continue
    
    if not food_items:
        logger.warning("No food items were successfully parsed from the text")
    else:
        logger.info(f"Successfully parsed {len(food_items)} food items")
    
    return food_items

def log_request_data(messages: List[Dict], endpoint: str):
    """Helper function to log request data in a readable format"""
    logger.info(f"\n{'='*50}\nRequest to {endpoint}\n{'='*50}")
    for msg in messages:
        if isinstance(msg.get('content'), list):
            # For messages with image content, show text only
            text_content = next((item['text'] for item in msg['content'] if item['type'] == 'text'), None)
            logger.info(f"Role: {msg['role']}")
            logger.info(f"Content: {text_content}")
            logger.info("(Image data present but not shown)")
        else:
            logger.info(f"Role: {msg['role']}")
            logger.info(f"Content: {msg['content']}\n")
    logger.info(f"{'='*50}")

@app.post("/identify/", response_model=IdentifiedFoodResponse)
async def identify_image(
    file: UploadFile,
    user_prompt: str = Form(...),
) -> IdentifiedFoodResponse:
    """
    Endpoint to identify contents of an uploaded image including nutrition information
    """
    logger.info(f"\n{'='*50}\nNew Image Identification Request\n{'='*50}")
    logger.info(f"File name: {file.filename}")
    logger.info(f"Content-Type: {file.content_type}")
    logger.info(f"User prompt: {user_prompt}\n{'='*50}")

    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.error(f"Invalid file type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the file content
        contents = await file.read()
        file_size_kb = len(contents) / 1024
        logger.info(f"Successfully read image file of size: {file_size_kb:.2f} KB")
        
        # Convert to base64
        base64_image = encode_image_to_base64(contents)
        logger.debug("Image successfully converted to base64")
        
        # Prepare the image URL
        image_url = f"data:{file.content_type};base64,{base64_image}"
        
        # Prepare messages compatible with Responses API
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT_IMAGE}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ]
            }
        ]
        
        # Log the complete request data
        log_request_data(messages, "identify")
        #breakdown with AI
        # Make OpenAI API call
        logger.info("Making request to OpenAI API...")
        response = client.responses.create(
            model="o4-mini",
            input=messages,
            reasoning={"effort": "medium"}
        )
        logger.info("Received response from OpenAI API")
        
        # Extract and log the response
        try:
            result = response.output_text
        except Exception:
            try:
                result = "".join(
                    part.text
                    for item in getattr(response, "output", [])
                    for part in getattr(item, "content", [])
                    if hasattr(part, "text")
                )
            except Exception:
                result = str(response)
        logger.info("\nOpenAI Response:")
        logger.info(f"{'='*50}\n{result}\n{'='*50}")
        
        # Parse the response into structured format
        food_items = parse_food_items_with_nutrition(result)
        logger.info(f"Successfully parsed {len(food_items)} food items")
        
        # Log parsed items in a readable format
        for idx, item in enumerate(food_items, 1):
            logger.info(f"\nFood Item {idx}:")
            logger.info(f"Name: {item.name}")
            logger.info(f"Quantity: {item.quantity} {item.unit}")
            logger.info(f"Description: {item.description}")
            logger.info(f"Nutrition: {json.dumps(item.nutrition.dict(), indent=2)}")
        
        return IdentifiedFoodResponse(foods=food_items)
        
    except Exception as e:
        logger.error(f"Error in identify_image endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/edit/", response_model=IdentifiedFoodResponse)
async def edit_food_breakdown(
    request_data: EditFoodRequest
) -> IdentifiedFoodResponse:
    """
    Endpoint to edit a previously identified food breakdown based on user corrections.
    """
    logger.info(f"\n{'='*50}\nNew Edit Request\n{'='*50}")
    logger.info("Previous breakdown:")
    logger.info(f"{request_data.previous_breakdown}")
    logger.info("\nUser correction:")
    logger.info(f"{request_data.user_correction}\n{'='*50}")

    try:
        # Prepare messages for the edit task (Responses API format)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT_EDIT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Original Breakdown:\n{request_data.previous_breakdown}\n\nUser Correction:\n{request_data.user_correction}"
                    }
                ]
            }
        ]
        
        # Log the complete request data
        log_request_data(messages, "edit")
        #edit with AI
        # Make OpenAI API call via Responses API
        logger.info("Making request to OpenAI API...")
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=messages,
            temperature=0.1
        )
        logger.info("Received response from OpenAI API")
        
        # Extract and log the response
        try:
            result = response.output_text
        except Exception:
            try:
                result = "".join(
                    part.text
                    for item in getattr(response, "output", [])
                    for part in getattr(item, "content", [])
                    if hasattr(part, "text")
                )
            except Exception:
                result = str(response)
        logger.info("\nOpenAI Response:")
        logger.info(f"{'='*50}\n{result}\n{'='*50}")
        
        # Parse the response into structured format
        food_items = parse_food_items_with_nutrition(result)
        logger.info(f"Successfully parsed {len(food_items)} food items")
        
        # Log parsed items in a readable format
        for idx, item in enumerate(food_items, 1):
            logger.info(f"\nFood Item {idx}:")
            logger.info(f"Name: {item.name}")
            logger.info(f"Quantity: {item.quantity} {item.unit}")
            logger.info(f"Description: {item.description}")
            logger.info(f"Nutrition: {json.dumps(item.nutrition.dict(), indent=2)}")
        
        return IdentifiedFoodResponse(foods=food_items)
        
    except Exception as e:
        logger.error(f"Error in edit_food_breakdown endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing edit request: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)