[![GitHub issues](https://img.shields.io/github/issues/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch)](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/issues) [![GitHub stars](https://img.shields.io/github/stars/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch)](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/stargazers) [![GitHub license](https://img.shields.io/github/license/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch)](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch)
# Liver Segmentation Using Monai and PyTorch
You'll find all the Python files you need to accomplish liver segmentation with Monai and PyTorch in this repo, and you can use the same code to segment other organs as well.

![Output image](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/blob/main/images/liver_segmentation.PNG)

So do this project, you will find some scripts that I wrote by myself and others that I took from Monai's tutorials. For this reason you need to take a look to their original repo and [website](https://monai.io/) to get more information.

## Cloning the repo
You can start by cloning this repo in your wordspace and then start playing with the function to make your project done.
```
git clone https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch
```
```
cd ./Liver-Segmentation-Using-Monai-and-PyTorch
```
## Packages that need to be installed:
```
pip install monai
```
```
pip install -r requirements.txt
```
## Showing a patient from the dataset
Some of the most common queries I had while utilizing medical imaging were regarding how to present a patient. To address this, I created explicit scripts for how to show a patient from the training and testing datasets, which you can see here.

```Python
def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    This function is to show one patient from your datasets, so that you can si if the it is okay or you need 
    to change/delete something.
    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """

    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    
    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

```

But before calling this function, you need to do the preprocess to your data, in fact this function will help you to visualize your patients after applying the different transforms so that you will know if you need to change some parameters or not.
The function that does the preprocess can be found in the `preprocess.py` file and in that file you will find the function `prepare()` that you can use for the preprocess.

## Training
After understanding how to do the preprocess you can start import the `3D Unet` from monai and defining the parameters of the model (dimensions, input channels, output channels...).

```Python
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
```

And to run the code, you can use the scripts `train.py` that will call the train function that I have created using the same principal used in Monai's tutorials.

## Testing the model
To test the model, there is the jupyter notebook `testing.ipynb` file that contains the different codes that you need. You will find the part to plot the training/testing graphs about the loss and the dice coefficient and of course you will find the the part to show the results of one of the test data to see the output of your model.

![Output image](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/blob/main/images/graphs.PNG)

----------------------------------------------------------------------------------------------------------------------------------
Before using the code, I recommend that you watch my course, in which I explain everything in this repo, or at the very least read my blog entries, in which I explain how to use the various scripts so that you don't get confused.
