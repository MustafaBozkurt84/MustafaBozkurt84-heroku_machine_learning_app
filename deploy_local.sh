rm -fr ./local_deployment/
rm local_deployment.tar
mkdir ./local_deployment/
cp ./Procfile ./local_deployment/
cp ./setup.sh ./local_deployment/
cp ./local.sh  ./local_deployment/
cp ./f-marshal.py ./local_deployment/
cp *.pkl ./local_deployment/
cp requirements.txt ./local_deployment/
tar -cf local_deployment.tar ./local_deployment

