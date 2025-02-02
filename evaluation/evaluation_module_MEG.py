# Andrew Bui
# National Taipei University of Technology
# Updated in 1/2025
# trongan93@ntut.edu.tw


# File: evaluate_image_quality.py

import os
from skimage import io
from brisque import BRISQUE
from pypiqe import piqe
import pandas as pd


def evaluate_image_quality(image_folder):
    """
    Evaluate the PIQE and BRISQUE scores of each image in a folder.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        pd.DataFrame: Dataframe containing image names and their quality scores.
    """
    results = []
    brisque_model = BRISQUE()  # Initialize BRISQUE
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(image_folder, file_name)
            try:
                # Read the image
                image = io.imread(file_path)

                # Compute PIQE score
                piqe_score, activityMask, noticeableArtifactMask, noiseMask = piqe(image)

                # Compute BRISQUE score
                # brisque_score = brisque_model.score(file_path)

                obj = BRISQUE(url=False)
                brisque_score = obj.score(img=image)

                # Append results
                results.append({
                    "Multi Exposure Generation Levels": file_name,
                    "PIQE Score": piqe_score,
                    "BRISQUE Score": brisque_score
                })
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    # Define the path to the folder containing the images
    image_folder = "/home/trongan93/Desktop/test-meg-1/"  # Replace with your folder path

    # Evaluate image quality
    quality_df = evaluate_image_quality(image_folder)

    # Save the results to a CSV file
    output_file = "/home/trongan93/Desktop/image_quality_scores.csv"
    quality_df.to_csv(output_file, index=False)

    # Display the results
    print("Image quality evaluation complete. Results saved to", output_file)
    print(quality_df)
