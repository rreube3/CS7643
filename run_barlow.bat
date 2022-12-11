set CONDAPATH=C:\Users\shawn\anaconda3
set ENVNAME=cs7643-final

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

SET PYTHONPATH=C:\Users\shawn\Desktop\Development\CS7643

::python .\barlow\main.py --data .\data\BarlowB\BarlowB\ --workers 16 --projector 262144-1024-1024-1024 --checkpoint-dir .\checkpoint\Barlow\64-128-256 --batch-size 64 --dropout 0.2 --epochs 30 --unet-layers 64-128-256

python .\barlow\main.py --data .\data\BarlowB\BarlowB\ --workers 16 --projector 65536-2048-2048-2048 --checkpoint-dir .\checkpoint\Barlow\16-32-64 --batch-size 32 --dropout 0.2 --epochs 30 --unet-layers 16-32-64

call conda deactivate