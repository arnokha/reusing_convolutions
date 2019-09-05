cd "BASE_GPU"

./run_performance_tests_2_layers_1.sh
./run_performance_tests_2_layers_2.sh
./run_performance_tests_2_layers_3.sh

cd ..
cd "BASE_CPU"

./run_performance_tests_2_layers_1.sh
./run_performance_tests_2_layers_2.sh
./run_performance_tests_2_layers_3.sh

cd ..
cd "REUSE_CPU"

./run_performance_tests_2_layers_1.sh
./run_performance_tests_2_layers_2.sh
./run_performance_tests_2_layers_3.sh

cd ..
cd "REUSE_GPU"

./run_performance_tests_2_layers_1.sh
./run_performance_tests_2_layers_2.sh
./run_performance_tests_2_layers_3.sh

cd ..
