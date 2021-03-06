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
python ./Null_embedding.py -sbj ${SBJ} -wl_sec ${wl_sec} -tr ${tr} -ws_trs ${ws_trs} -LE_k ${LE_k} -LE_met ${LE_metric} -p ${p} -TSNE_met ${TSNE_metric} -UMAP_k ${UMAP_k} -UMAP_met ${UMAP_metric} -n ${n} -data ${data} -drop ${drop}