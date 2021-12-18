[![GitHub issues](https://img.shields.io/github/issues/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch)](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/issues) [![GitHub stars](https://img.shields.io/github/stars/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch)](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/stargazers) [![GitHub license](https://img.shields.io/github/license/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch)](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch)
# Liver Segmentation Using Monai and PyTorch
In this repo, you will find all the Python files to do liver segmentation using Monai and PyTorch, and you can use the same code for other organs segmentation.

![Output image](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/blob/main/images/liver_segmentation.PNG)

So do this project, you will find some scripts that I wrote by myself and others that I took from Monai's tutorials. For this reason you need to take a look to their original repo and [website](https://monai.io/) to get more information.

## Cloning the repo
You can start by cloning this repo in your wordspace and then start playing with the function to make your project done.
```
git clone https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch
```
```
cd ./Liver-Segmentation-Using-Monai-and-PuTorch
```
## Packages that need to be installed:
```
pip install monai
```
```
pip install -r requirements.txt
```
## Showing a patient from the dataset
Some of the common questions that I received while using the medical imaging is how to show a patient, for that I wrote a clear scripts about how to show a patient from the training and the testing datasets, that you can find here.

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

Before using the code I recommend you to see my course where I explained everything you see in this repo, or at least read my blog posts that I wrote to explain how to use the different scripts so that you won't get confused.
