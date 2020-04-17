# BUs DETECTION API

Step 1: Download the python installer from here:   https://www.python.org/downloads/release/python-370/

Step 2: Follow these instructions to install:   https://www.itnota.com/install-python-windows-server/

Step 3: Now go to the extracted folder and install the dependencies using pip command:
	
	pip install -r requirements.txt

Step 4: Run the app.py file using: 

	python app.py 80
	
Here 80 is the port number

The application will display the path for the API endpoint which would be like:	"http://127.0.0.1:<PORT NUMBER>/status"


There's a test_call.py in the folder that demonstrates how I use python to make the POST request
the format for the request is: {"base": "C:/Users/azfar/Downloads/samuel/test2"}

The application returns a JSON of format:
{"Status": "departure"}


NOTE: If python and pip commands not work, then use python3 and pip3 instead.
	Also, in powershell "py code.py" is used instead of "python code.py"

Test image folders are in the "test_images" folder
