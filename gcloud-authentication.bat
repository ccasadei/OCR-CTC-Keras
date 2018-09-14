@echo off
cls

set PROJECT_ID=polar-arcana-216310

echo Avvio l'autenticazione dell'applicazione
call gcloud auth application-default login

echo Ora cambio progetto in %PROJECT_ID%
call gcloud config set project %PROJECT_ID%
