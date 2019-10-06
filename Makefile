install:
		python setup.py install

init:
		# Configure
		echo "traffic-sign" > labels.txt

config:
		python init_config.py

test:
		pytest --cov=./
