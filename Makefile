install:
		python setup.py install

init:
		# Download data
		wget https://ariangg.s3-sa-east-1.amazonaws.com/data.zip -O data.zip
		unzip data.zip

		# Configure
		echo "traffic-sign" > labels.txt

config:
		python init_config.py

test:
		pytest --cov=./
