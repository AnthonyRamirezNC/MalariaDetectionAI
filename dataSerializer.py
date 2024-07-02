import os
import cv2
import pandas as pd
import numpy as np

class dataSerializer:
    def serialize_training_data():
        # Specify the path to the file you want to remove
        file_path = "data.csv"

        # Check if the file exists before attempting to remove it
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"The file {file_path} has been successfully removed.")
        else:
            print(f"The file {file_path} does not exist.")

        # Base directory where your blood samples are
        base_dir = "train"

        # Create a list to store all image data
        all_image_data = []

        # Loop through each folder in the base directory
        for folder in os.listdir(base_dir):
            print("processing folder: ", folder)
            folder_path = os.path.join(base_dir, folder)
                
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # List all files in the folder
                files = os.listdir(folder_path)
                # Filter out files that are images
                image_files = [f for f in files if f.endswith('.jpg')]
                # Process each image
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    #print(f"Processing {image_path}")

                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is not None:
                        #print("Image found")

                        # Resize image to 224x224 if it's not already that size
                        image = cv2.resize(image, (224, 224))

                        # Flatten the image to a 1D array
                        pixelValues = image.flatten()

                        # Determine if malaria is positive in sample
                        positiveMalaria = 1 if folder == "Parasitized" else 0

                        # Append the positiveMalaria boolean to the end of this array
                        pixelValues_with_conclusion = np.append(pixelValues, positiveMalaria)

                        # Add this image data to our list
                        all_image_data.append(pixelValues_with_conclusion)
                    else:
                        print("Image not found")
            print("Finished folder:" + folder)           

        print("creating dataframe")
        # Create DataFrame from all image data
        column_names = [str(i) for i in range(224*224)] + ['malariaPositive']
        df = pd.DataFrame(all_image_data, columns=column_names)

        print("Data Serialized")
        return df

    def serialize_test_data():
        # Base directory where your blood samples are
        base_dir = "test"

        # Create a list to store all image data
        all_image_data = []

        # Loop through each folder in the base directory
        for folder in os.listdir(base_dir):
            print("processing folder: ", folder)
            folder_path = os.path.join(base_dir, folder)
                
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # List all files in the folder
                files = os.listdir(folder_path)
                # Filter out files that are images
                image_files = [f for f in files if f.endswith('.jpg')]
                # Process each image
                for image_file in image_files:
                    image_path = os.path.join(folder_path, image_file)
                    #print(f"Processing {image_path}")

                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is not None:
                         #print("Image found")

                        # Resize image to 224x224 if it's not already that size
                        image = cv2.resize(image, (224, 224))

                        # Flatten the image to a 1D array
                        pixelValues = image.flatten()

                        # Determine if malaria is positive in sample
                        positiveMalaria = 1 if folder == "Parasitized" else 0

                        # Append the positiveMalaria boolean to the end of this array
                        pixelValues_with_conclusion = np.append(pixelValues, positiveMalaria)

                        # Add this image data to our list
                        all_image_data.append(pixelValues_with_conclusion)
                    else:
                        print("Image not found")
            print("Finished folder:" + folder)           

        print("creating dataframe")
        # Create DataFrame from all image data
        column_names = [str(i) for i in range(224*224)] + ['malariaPositive']
        df = pd.DataFrame(all_image_data, columns=column_names)

        print("Data Serialized")
        return df

    def serialize_single_image():
