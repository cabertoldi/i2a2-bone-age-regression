# i2a2-bone-age-regression
This a competition to create a model to predict the Bone Age of children from a Hand X-Ray.  
https://www.kaggle.com/c/i2a2-bone-age-regression/overview

The solution complete published in the competition on Kaggle is on the path `notebooks/complete-solution.ipynb`.

This is a project version to run in a cotainer.
To run this project is necessary some steps:
    1. Run `docker network create i2a2`
    2. Run `docker-compose build`
    3. Create a new folder called `data` and put all datasets that are in page of competition on Kaggle.
    4. Run `docker-compose up`
