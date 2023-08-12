# Recommender-System

Hi! This is a Recommender System for movies using the a Kaggle dataset on movies (according to their content and generes) and ratings (according to user ratings of the movies identified by their movieID).

You will find 2 main files. Here's a description on the distinction between the two files:

- main.ipynb: This Jupyter notebook was aimed to help me understand the math and algorithm behind the 2 different types of filtering in Recommender Systems. Here I implement a Content-based Filtering model and a Collaborative Filtering model to showcase the difference between the two and why they are both important to implement a powerful recommender system. I use Linear Algebra technique of Cosine Similarity for Content Based filtering, and I use SVD (Matrix Factorization), PCA (Dimensionality Reduction), and Cosine Similarity techniques for Collaborative filtering. At the end, I attempt to implement Matrix Factorization from scratch in order to predict the missing ratings of existing users that can further be used to recommend the movies to them. I also utilize Jupyter notebook's markdown feature to comment my observations and learnings throughout the implementation.

- main.py: This python file is aimed to implement a user friendly interface for recommending movies using both Collaborative and Content-based filtering. Here, I am building upon the concepts outlined in main.ipynb and apply those concepts to recommend 5 new movies for the user to watch based on 3 movies that they have liked in the past. I have used several optimization techniques to make the recommender system faster in its predicatbility and I have also implemented a bar chart for user to visualize 50 other movies that were contending to be in the top 5 recommendations. To clarify, the chart presents top 50 movies that are most similar to the user's past choices and highlight the top 5 that are the most similar. Lastly, I used Flask, HTML, and CSS to build the user interface.

I hope you enjoy this project as much as I did!
