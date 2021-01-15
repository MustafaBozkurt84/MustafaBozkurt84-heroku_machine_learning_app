

rm ./app.py
cp /c/Users/mwolf/PycharmProjects/pythonProject9/app.py ./
git add .
git commit -m "app.py"
mv ./trainchurn.csv ~/Desktop/
git push origin master
mv ~/Desktop/trainchurn.csv .
git add .
git commit -m "app.py"
git push heroku
