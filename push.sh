

rm ./app.py
cp /c/Users/mwolf/PycharmProjects/streamlit-machine-learning/automated-machine-learning.py .
git add .
git commit -m "app.py"

git push origin master
git push heroku
