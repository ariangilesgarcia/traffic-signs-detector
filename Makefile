install:
		python setup.py install

init:
		# Download data
		wget detector.ml:8080/data.zip -O data.zip
		unzip data.zip

		# Configure
		echo "traffic-sign" > labels.txt

test:
		pytest --cov=./
