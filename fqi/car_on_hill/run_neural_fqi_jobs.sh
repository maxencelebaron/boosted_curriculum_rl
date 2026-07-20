export PYTHONPATH=$PYTHONPATH:$PWD/../..

python run.py --n-exp 1 --n-jobs 1 --use-curriculum --use-boosting --use-neural
python run.py --n-exp 1 --n-jobs 1 --use-boosting --use-neural
python run.py --n-exp 1 --n-jobs 1 --use-curriculum --use-neural
python run.py --n-exp 1 --n-jobs 1 --use-neural
python visualize_neural_results.py
