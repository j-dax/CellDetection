# Cell Detection

This was a joint project for our capstone project.

Authors

Kalam, D.; Li, P.; Lin, W.; Piazza, M.; Ryder, W.

## Getting Started

Install the dependencies:

Clone the repo and install the dependencies

```pip install -r requirements.txt```

If pycocotools fails to build wheel, it's a [known issue with Windows](https://github.com/cocodataset/cocoapi/issues/169)

```pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"```

### Running the web server

Now you may run the server from project root directory:

```
python main.py
```

The server defaults to [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Accessing the web application

[Annotated image using XML](http://127.0.0.1:5000/test)

[neural net-generated contours](http://127.0.0.1:5000/net)

[after changes are made](http://127.0.0.1:5000/updated)
