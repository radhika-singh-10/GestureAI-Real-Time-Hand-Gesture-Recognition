
# GestureAI


Our project focuses on gesture recognition, a field within computer vision and human-computer interaction that aims to interpret human gestures via Deep Learning. For our study, we have chosen the Jester dataset as it provides a comprehensive collection of hand gesture videos. The Jester dataset is particularly well-suited for our purposes due to the following reasons: 



• Large-scale and Diverse: The Jester dataset contains a large number of videos featuring diverse hand gestures, providing a rich source of data for training and evaluation of gesture recognition algorithms. 

• Real-world Gestures: The gestures captured in the Jester dataset represent common human actions and interactions, making it applicable to a wide range of practical applications. 


• Annotated Data: The dataset comes with pre-defined labels for each gesture, which facilitates supervised learning and evaluation of gesture recognition models. 


• Community Benchmark: The availability of the Jester dataset has led to it being widely used as a benchmark for evaluating gesture recognition algorithms, allowing for comparison and advancement of research in the field. By leveraging the Jester dataset, our project aims to develop and evaluate state-of-the-art gesture recognition models that can accurately interpret and classify human gestures in real-time.


Basic Requirements
```shell
- Python 3.x
- PyTorch 0.4.0
 ```

Command for creating a virtual environment

```
python -m venv gestureRecogEnv
```

Command for activating Virtual Environment

For Windows

```
gestureRecogEnv\Scripts\activate
```

For MacOS\Linux
```
source gestureRecogEnv/bin/activate
```

 
Requirements.txt:  
```shell
pip install -r requirements.txt
```

Command to deactivate the Virtual Environment

```
deactivate
```


Steps to train the model

```
python train.py 
```

Steps to test the model

```
python test.py
```
References are added in comments in the code for you to refer to, in case of any confusion.
For further information, please feel free to reach out to any of the contributors -
  <ul>
    <li>Atharva Musale - amusale@buffalo.edu</li>
    <li>Radhika Singh - rsingh57@buffalo.edu</li>
    <li>Swapnil Khare - skhare@buffalo.edu</li>
  </ul>
