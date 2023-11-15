cd `dirname $0` # カレントディレクトリに移動
cd build
cmake .. .
make 
cd ..
./build/main