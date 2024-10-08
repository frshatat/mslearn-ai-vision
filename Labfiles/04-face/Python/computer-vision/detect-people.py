from dotenv import load_dotenv
import os
from array import array
from PIL import Image, ImageDraw
import sys
import time
from matplotlib import pyplot as plt
import numpy as np

# Import namespaces
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = 'images/people.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        # Authenticate Azure AI Vision client
        print(ai_endpoint)
        cv_client = ComputerVisionClient(
            ai_endpoint,
            CognitiveServicesCredentials(ai_key)
        )
        
        # Analyze image
        AnalyzeImage(image_file, cv_client)

    except Exception as ex:
        print(ex)


def AnalyzeImage(image_file, cv_client):
    print('\nAnalyzing', image_file)

    # Specify features to be retrieved (OBJECTS)
    analysis_features = [VisualFeatureTypes.objects]

    # Get image analysis
    with open(image_file, "rb") as image_stream:
        analysis = cv_client.analyze_image_in_stream(image_stream, visual_features=analysis_features)

    if analysis.objects:
        print("\nPeople in the image:")

        # Prepare image for drawing
        image = Image.open(image_file)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        for obj in analysis.objects:
            if obj.object_property == "person":
                # Draw object bounding box
                r = obj.rectangle
                bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
                draw.rectangle(bounding_box, outline=color, width=3)

                # Return the confidence of the person detected
                print(" {} (confidence: {:.2f}%)".format(bounding_box, obj.confidence * 100))

        # Save annotated image
        annotated_image_path = "annotated_" + os.path.basename(image_file)
        image.save(annotated_image_path)
        print(f"Annotated image saved as {annotated_image_path}")
    else:
        print("Analysis failed.")
        print("   Error reason: {}".format(analysis.reason))
        print("   Error code: {}".format(analysis.error.code))
        print("   Error message: {}".format(analysis.error.message))

if __name__ == "__main__":
    main()