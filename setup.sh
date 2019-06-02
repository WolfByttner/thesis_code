
DIPHA_PATH=$(pwd)/dipha/dipha
PERSEUS_PATH=$(pwd)/Deps/perseus

git submodule update --init --recursive
cd chofer_nips2017
git reset --hard
sed -i "s#from chofer_torchex.nn import SLayer#from chofer_torchex.nn import SLayerExponential as Slayer#" \
    src/sharedCode/experiments.py


cd tda-toolkit
git reset --hard
cd ..
sed -i "s#dipha=#dipha=$DIPHA_PATH#" tda-toolkit/pershombox/_software_backends/software_backends.cfg
sed -i "s#perseus=#perseus=$PERSEUS_PATH#" tda-toolkit/pershombox/_software_backends/software_backends.cfg

git submodule update --init --recursive
cd chofer_torchex
git pull origin master

cd ../../dipha
cmake CMakeLists.txt && make

cd ..
mkdir Deps
cd Deps
wget http://people.maths.ox.ac.uk/nanda/source/perseus_4_beta.zip
unzip -f perseus_4_beta.zip
g++ Pers.cpp -O3 -fpermissive -o perseus

cd ..
touch chofer_nips2017/__init__.py
