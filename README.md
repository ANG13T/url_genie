<p align="center">
  <img align="center" alt="URL Genie logo" width="150" src="https://github.com/ANG13T/url_genie/blob/main/webapp/assets/url_genie_logo.png">
</p>

# URL Genie
Malicious URL Detection Model Neural Network Optimized by Genetic Algorithms

## About
URL Genie is a web application implementing a Multilayer Perceptron Neural Network optimized using genetic algorithms.
Detect whether a domain name or URL is malicious by inputting a URL. For instance,
```
https://www.google.com -> SAFE
http://stcdxmt.bigperl.in/klxtv/apps/uk/ -> MALICIOUS
```
 
## Previews
<img alt="Preview 1" width="800" src="https://github.com/ANG13T/url_genie/blob/main/webapp/assets/project_graphic.png">

<img alt="Preview 2" width="800" src="https://github.com/ANG13T/url_genie/blob/main/webapp/assets/preview.png">

## Neural Network Model
<img alt="Model Image" width="800" src="https://github.com/ANG13T/url_genie/blob/main/webapp/assets/model_graphic.png">

The model sequence defined within `genetic_algorithm_implementation.py` is as follows:
1. Integrate CSV Dataset and Remove Unnecessary Columns
2. Use SMOTE to Balance out Class Distribution in Dataset
3. Split Dataset into Training and Testing Sets using 80:20 Ratio
4. Initialize Multilayer Perception
5. Utilize Adam Optimization and Binary Cross Entropy Loss Function
6. Initialize Model Callback to Wait Until 0.1 Validation Loss 
7. Train Model with 10 Epochs and Batch Size of 256
8. Verify Model Results using 10 Examples
9. Run Each Model Iteration through a Genetic Algorithm
10. Evaluate Fitness of Each Model by Referencing Accuracy
11. Determine Best Model within Population
12. Save the Best Model into a .h5 File Output

## Usage
To build from source, you will Python3 and Pip installed.
```
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

Visit `localhost:8501` to see the web application

## Code Structure

### Research Jupyter Notebooks
The `Research_Notebooks` folder contains the Jupyter research notebooks for this project. Each notebook explores a unique aspect of the dataset.

`Feature_Extraction_Notebook.ipynb` extracts pertinent information out of the malicious and benign URLs Kaggle dataset

`Data_Visualization_Notebook.ipynb` provides relevant data visualizations of the features extracted in the feature extraction notebook

`Training_Models_Notebook.ipynb` tests a couple of models to classify which one is best suited for detecting malicious domains

`Genetic_Algorithm_Notebook.ipynb` experiments with genetic algorithms and applies it to a neural network


### Streamlit Web Application
The `webapp` folder contains all the necessary files to setup the web server for the application.

Once you execute `streamlit run app.py` visit `localhost:8501` in a browser to see the application.

The `app.py` file contains all the relevant Streamlit web application code.

The `model_generation.py` file contains the code to generate the classification NN without GA optimization.

The `genetic_algorithm_implementation.py` file contains the code to generate the classification NN with GA optimization. 

## Learning and Resources
To learn more about DNS functionality, malicious URL generation, and the machine learning models used for this project, refer to [this article](https://medium.com/@angelinatsuboi/bio-cybersecurity-using-genetic-algorithms-to-detect-malicious-urls-3811326b7c1).

## Contributing
URL Genie is open to any contributions. Please fork the repository and make a pull request with the features or fixes you want to be implemented.

## Special Thanks
This project is a customized and enhanced derivative of a combination of previous research conducted by [Deepesh Mhatre](https://github.com/deepeshdm/Phishing-Attack-Domain-Detection) and [Suryansh S.](https://towardsdatascience.com/gas-and-nns-6a41f1e8146d) Please feel free to show your support by checking out their projects and profiles!

## Support
If you enjoyed this project, please consider [becoming a sponsor](https://github.com/sponsors/ANG13T) in order to fund my future projects. 

To check out my other works, visit my [GitHub profile](https://github.com/ANG13T).
