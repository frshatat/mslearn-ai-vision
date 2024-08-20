from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw, ImageFont
import sys
from matplotlib import pyplot as plt
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

    # Get image analysis
    with open(image_file, "rb") as image_stream:
        analysis = cv_client.analyze_image_in_stream(image_stream, visual_features=[VisualFeatureTypes.faces])

    if analysis.faces:
        print("\nFaces in the image:")

        # Prepare image for drawing
        image = Image.open(image_file)
        fig = plt.figure(figsize=(image.width / 100, image.height / 100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'
        font = ImageFont.load_default()

        for face in analysis.faces:
            # Draw face bounding box
            r = face.face_rectangle
            bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

            # Add label with age
            label = f"Age: {face.age if face.age else 'N/A'}"
            text_bbox = draw.textbbox((r.left, r.top), label, font=font)
            text_location = (r.left, r.top - (text_bbox[3] - text_bbox[1]))
            draw.rectangle([text_location, (text_location[0] + (text_bbox[2] - text_bbox[0]), text_location[1] + (text_bbox[3] - text_bbox[1]))], fill=color)
            draw.text(text_location, label, fill="black", font=font)

            # Return the confidence of the face detected
            print(" {} (age: {})".format(bounding_box, face.age if face.age else 'N/A'))

        # Save annotated image
        annotated_image_path = "annotated_" + os.path.basename(image_file)
        image.save(annotated_image_path)
        print(f"Annotated image saved as {annotated_image_path}")
    else:
        print("No faces detected.")
        
if __name__ == "__main__":
    main()