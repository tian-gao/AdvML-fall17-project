## Artistic style transfer implementation with Tensorflow

This project is intended to be the course final project for Columbia University GR5242 Advanced Machine Learning.

All copyrights reserved by author: [Tian Gao][email] (CU UNI: tg2585)

### Directory Structure

```
project/
   |---- input/
   |       |---- content/
   |       |        |---- content.jpg
   |       |---- style/
   |       |        |---- style.jpg
   |---- output/
   |       |---- output.jpg
   |
   |---- style_transfer.py
   |---- neural_network.py
   |---- visual_geometry_group.py
   |
   |---- utils.py
   |---- constants.py
   |---- settings.py
   |---- logger.py
   |
   |---- imagenet-vgg-verydeep-19.mat
   |---- requirements.txt
```

This directory structure only shows directories and files that are necessary to run the code and generate certain outputs.

This repository also includes a `report` folder which contains the full project report (source files, illustrations and PDF), which is exclusively for Columbia University GR5242. All copyrights reserved.

### Running

1. Construct your folder exactly as of the structure above; put the content picture (the one to be transfered) in `content` folder and put the style picture (the one that provides with the artistic style) in `style` folder

2. Make sure you have the [pre-trained neural network data from VGG][data]

3. This project runs with Python 3.6. Install dependencies with

   ```
   pip install -r requirements.txt
   ```

   and then run the following command

   ```
   python style_transfer.py --content content.jpg --style style.jpg --output output.jpg
   ```

   Change `*.jpg` in the command input to match your file names

### Related Works

This project is inspired by the [paper][paper] on style transfer.

### Acknowledgement

Thanks to the authors of the [paper][paper], who had such interesting ideas and inspired this work.

[email]: mailto:tian.gao@columbia.edu
[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[data]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

