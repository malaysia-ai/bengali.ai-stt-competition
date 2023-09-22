sudo apt install build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev -y
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build
cd kenlm/build
~/.local/bin/cmake ..
make -j2