# Traffic Signs Detector [![Build Status](https://travis-ci.org/ariaaan/traffic-signs-detector.svg?branch=master)](https://travis-ci.org/ariaaan/traffic-signs-detector) [![codecov](https://codecov.io/gh/ariaaan/traffic-signs-detector/branch/master/graph/badge.svg)](https://codecov.io/gh/ariaaan/traffic-signs-detector)

## Instalación

### 1. Clonar repositorio.
```
git clone https://github.com/ariaaan/traffic-signs-detector/
```

### 2. Instalar pre-requisitos.
#### 2.1. Instalar pre-requisitos desde `pip`.
Ubicados en cualquier lugar.
```
pip install tensorflow-gpu
pip install keras
pip install h5py
pip install pydub # si se quieren notificaciones sonoras
pip install randomcolor
```

#### 2.1. Instalar Darkflow.

Ubicados en cualquier lugar.
```
git clone https://github.com/thtrieu/darkflow
cd darkflow/
python setup.py build_ext --inplace
pip install .
```


### 3. Descargar modelos.
Ubicado en la carpeta principal del proyecto.
```
make init
```

### 4. Configurar sistema.
Ubicado en la carpeta principal del proyecto.
```
make config
```

### 5. Instalar sistema.
Ubicado en la carpeta principal del proyecto.
```
make install
```

### 6. Probar funcionamiento.
Para probar el funcionamiento, crear un script de python con el siguiente código, reemplanzando:
- `config_path`: path al archivo `.json`, ubicado en `./cfg/config.json` en la carpeta principal del proyecto.
- `video_path`: path a un archivo de video en el cual se desea detectar señales de tránsito (para probar se puede utilzar este [video]()).

```python
from detector.detector import create_detector_from_file

detector = create_detector_from_file('<config_path>')
detector.detect_video_feed('<video_path>', output='output.avi')
```
