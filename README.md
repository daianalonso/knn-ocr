# Reconocimiento de dígitos: Métodos Numéricos DC-UBA

## Instrucciones

1. Crear un repo git en donde se bajen esto

```
git init
git remote add origin <nuestra-nueva-url-de-git>
```

2. Bajarse los repositorios de `pybind` y `eigen` como submódulos

```
git submodule init
git submodule add https://github.com/eigenteam/eigen-git-mirror
git submodule add https://github.com/pybind/pybind11
# Elegimos versiones de eigen y pybind
cd pybind11/ && git checkout v2.2.4 && cd ..
cd eigen-git-mirror && git checkout 3.3.7 && cd ..
```

3. Instalar requerimientos (*Previamente activar el entorno virtual. Ver  más abajo*)

```
pip install -r requirements.txt
```

4. Descomprimir datos

```
cd data && gunzip *.gz && cd ..
```

5. Correr Jupyter

```
jupyter lab
```

Listo!

### Datos

En `data/` tenemos los datos de entrenamiento (`data/train.csv`) y los de test (`data/test.csv`).

### Otros directorios

En `src/` está el código de C++, en particular en `src/metnum.cpp` está el entry-point de pybind.

En `notebooks/` hay ejemplos para correr partes del TP usando sklearn y usando la implementación en C++.


## Creación de un entorno virtual de python

### Con pyenv

```
curl https://pyenv.run | bash
```

Luego, se sugiere agregar unas líneas al bashrc. Hacer eso, **REINICIAR LA CONSOLA** y luego...

```
pyenv install 3.6.5
pyenv global 3.6.5
pyenv virtualenv 3.6.5 tp2
```

En el directorio del proyecto

```
pyenv activate tp2
```

### Directamente con python3
```
python3 -m venv tp2
source tp2/bin/activate
```

### Con Conda
```
conda create --name tp2 python=3.6.5
conda activate tp2
```

## Instalación de las depencias
```
pip install -r requirements.txt
```

## Correr notebooks de jupyter

```
cd notebooks
jupyter lab
```
o  notebook
```
jupyter notebook
```


## Compilación
Ejecutar la primera celda del notebook `knn.ipynb` o seguir los siguientes pasos:

- Compilar el código C++ en un módulo de python
```
mkdir build
cd build
rm -rf *
cmake -DPYTHON_EXECUTABLE="$(which python)" -DCMAKE_BUILD_TYPE=Release ..
```
- Al ejecutar el siguiente comando se compila e instala la librería en el directorio `notebooks`
```
make install
```

## Kaggle submission Digit Recognizer

- Compilar el código C++
```
./build.sh
cd build && make
```
- Ejecutar con los datos de test y train de kaggle, setear cantidad de vecinos (k) y cantidad de componentes principales (a) a utilizar
```
./tp2 -m 1 --k <kNN parameter> --a <pca parameter> -i train_kaggle.csv -t test_kaggle.csv -o kaggle_submission.csv
```
- En la carpeta data se guarda el archivo resultado para subir a kaggle
