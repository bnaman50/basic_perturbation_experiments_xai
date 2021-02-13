# basic_perturbation_experiments_xai
This repo contains the code for the preliminary experiments for our project - [SAM](https://anhnguyen.me/project/sam/ "SAM"). 
It allows you to perform following experiments- 
- Visualize the various heatmaps for a given image and a particular model.
- Make a particular row/coloumn or a range of rows/colouns black (zero pixels).
- Flip the image vertically/horizontally.
- Translate the image horizontally and/or vertically.
- Rotate the image.
- Generate heatmap for a label of users choice (options - original, predicted, topN, bottomN or random list of labels).
    - (Here topN/bottomN means top/botton `N` predicyed labels.)

## Takeaways
These experients confirmed our belief that there is definitely some problems with the current heatmap methods. But it was hard to figure out the exact problems due of the complexity of the ImageNet images. Thus, we decided to create a toy dataset and a model to better understand the workings and issues pertaining to the heatmap methods. The code of our toy experiment, **Center Pixel Model** can be found [here](https://github.com/bnaman50/center-pixel-model).

## Setup
Tested for Python 3.6.7
1. Create a virtual environment (preferrably using conda)
2. `pip install -r requirement.txt`
3. Instal `innvestigate==1.0.6` manually since older versions were not available on PyPi. 

    ```git clone https://github.com/albermax/innvestigate; cd innvestigate; git checkout 604017a; python setup.py install; cd ..; rm -rf innvestigate```

## Usage
`python basic_experiments.py -h`

## Result
This is how the resultant plots will look like. 
![alt text](/results/result.png?raw=true "Sample Output")


