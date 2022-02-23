set -e
# Enter scripts directory
echo "++ Entering Notebooks directory..."
cd ${PRJDIR}/ml_comparison

# Activate miniconda
echo "++ Activating miniconda"
. ${conda_loc}

# Activate vigilance environment
echo "++ Activating rapidtide environment"
conda activate ${conds_env}

# Run the program
./SWC_matrix.py -sbj ${SBJ} -wl_sec ${wl_sec} -tr ${tr} -ws_trs ${ws_trs}