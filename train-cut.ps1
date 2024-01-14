param (
    [string]$name,
    [string]$dataroot,
    [int]$size = 512,
    [int]$batch = 1
)


python train.py --dataroot $dataroot --name $name --CUT_mode CUT --batch_size $batch --load_size $size --crop_size $size