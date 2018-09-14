@echo off
cls

set BUCKET_NAME=casadei-cristiano
set REGION=europe-west1
set MODULE_NAME=ocr.ocr_training

set NOW=%time: =0%
set JOB_NR=%date:~6,4%%date:~3,2%%date:~0,2%_%NOW:~0,2%%NOW:~3,2%%NOW:~6,2%
set JOB_NAME=ocr_%JOB_NR%
set JOB_DIR=gs://%BUCKET_NAME%/OCR/jobs_dir/%JOB_NAME%

echo Eseguo il deploy del job su Google ML-Engine
call gcloud ml-engine jobs submit training %JOB_NAME% --config=ocr/cloudml-gpu.yaml --job-dir %JOB_DIR% --runtime-version 1.9 --python-version 3.5 --module-name %MODULE_NAME% --package-path ./ocr --region %REGION%

echo Avvio lo streaming del log
call gcloud ml-engine jobs stream-logs %JOB_NAME% --polling-interval=1
