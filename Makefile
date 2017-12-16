install:
		pip install -r requirements.txt
		pip install git+https://github.com/thtrieu/darkflow

init:
		echo 'traffic-sign' > labels.txt
		wget aberturasdalum.com.ar:8080/trafficsigns.weights -O data/yolo/full/trafficsigns.weights
		wget aberturasdalum.com.ar:8080/trafficsigns.weights -O data/yolo/tiny/trafficsigns_tiny.weights

test:
		mkdir tests/data
		wget http://www.educacionsantacruz.gov.ar/images/Educ_Ambiental/Monte_Leon/PNMonteLeon/imagenes/ruta.jpg -O tests/data/test.jpg
		pytest
		rm -rf tests/data
