pushd $PSScriptRoot
conda activate cut
python -m visdom.server
conda deactivate
popd