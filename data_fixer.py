import os
import shutil
from PIL import Image
from image_resize import resizedImage

# train DATASET PATH
trainingFolderPath = '../data/CATS_DOGS/train/'

classes = sorted(os.listdir(trainingFolderPath))

print(classes)


def dataFixer(imagesFolderPath, size=(128, 128)):
    # Loop through each subfolder in the input folder
    print('Transforming images...')
    for root, folders, files in os.walk(trainingFolderPath):
        for subFolder in folders:
            print('processing folder ' + subFolder)
            # Create a matching subfolder in the output dir
            saveFolder = os.path.join(imagesFolderPath, subFolder)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            # Loop through the files in the subfolder
            fileNames = os.listdir(os.path.join(root, subFolder))
            for fileName in fileNames:
                # Open the file
                filePath = os.path.join(root, subFolder, fileName)
                # print("reading " + filePath)
                image = Image.open(filePath)
                # Create a resized version and save it
                reshapedImage = resizedImage(image, size)
                saveAs = os.path.join(saveFolder, fileName)
                #print("writing " + saveAs)
                reshapedImage.save(saveAs)

    print('Done.')
