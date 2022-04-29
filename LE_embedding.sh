set -e
# Enter scripts directory
echo "++ Entering Notebooks directory..."
cd ${PRJDIR}/ml_comparison

# Activate miniconda
echo "++ Activating miniconda"
. ${conda_loc}

# Activate vigilance environment
echo "++ Activating rapidtide environment"
conda activate ${conda_env}

# Run the program
python ./LE_embedding.py -sbj ${SBJ} -wl_sec ${wl_sec} -tr ${tr} -k ${k} -n ${n} -met ${metric} -drop ${drop}