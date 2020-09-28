# IBM AI Enterprise Workflow Capstone
Files for the IBM AI Enterprise Workflow Capstone project. 

Details:

Are there unit tests for the API? Unittest\testAPIs.py

Are there unit tests for the model? Unittest\unittest-model.py

Are there unit tests for the logging? Unittest\unittest-logger.py

Can all of the unit tests be run with a single script and do all of the unit tests pass? Unittest\tests-run-script.py

Is there a mechanism to monitor performance? Unittest\unittest-logger.py

Was there an attempt to isolate the read/write unit tests from production models and logs?

Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined? app.py

Does the data ingestion exists as a function or script to facilitate automation? cslib.py

Were multiple models compared? time-series-notebooks

Did the EDA investigation use visualizations? Capstone Part1.ipynb

Is everything containerized within a working Docker image? Dockerfile 

Did they use a visualization to compare their model to the baseline model? time-series-notebooks


Build the Docker image and run it
Step one: build the image (from the directory that was created with this notebook)
    ~$ cd docker
    ~$ docker build -t predict-app .
Check that the image is there.
    ~$ docker image ls
You may notice images that you no longer use. You may delete them with
    ~$ docker image rm IMAGE_ID_OR_NAME
Run the container
docker run -p 4000:8080 predict-app