FROM tensorflow/tensorflow:2.2.0

# Copy repo with trained model
RUN apt-get update && apt-get install -y git
WORKDIR /workspace/tf-classifier-repo
RUN git clone https://github.com/AjinkyaChavan9/RGB-Color-Classifier-with-Deep-Learning-using-Keras-and-Tensorflow.git
# WORKDIR /workspace/tf-classifier-repo/RGB-Color-Classifier-with-Deep-Learning-using-Keras-and-Tensorflow
# RUN pip install -r requirements.txt

RUN pip install --upgrade pip

RUN pip install keras==2.4.0
RUN pip install tf2onnx==1.9.3
RUN pip install onnx
RUN pip install jupyter notebook

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888
WORKDIR /workspace
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
