install:
		python setup.py install

init:
		# Download data
		wget 138.197.90.173:8080/data.zip -O data.zip
		unzip data.zip

		# Configure
		echo "traffic-sign" > labels.txt

config:
		python init_config.py

test:
		pytest --cov=./
