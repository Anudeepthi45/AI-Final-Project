# Netflix Content Clustering

## Project Overview
This project focuses on clustering Netflix movies and TV shows using unsupervised machine learning techniques. By analyzing textual metadata such as description, genre, cast, and director, the system groups similar content into clusters, helping to understand content similarity and improve recommendation insights. The project demonstrates the practical application of NLP and machine learning on real-world streaming data.

## Objectives
  - To preprocess and clean textual Netflix content data  
  - To convert text data into numerical features using TF-IDF  
  - To group similar movies and TV shows using clustering techniques  
  - To analyze and display clustered Netflix content  
  - To understand how unsupervised learning can be applied without labeled data  

## Methodology
  1. Load the Netflix dataset  
  2. Handle missing values in text-based features  
  3. Combine important text attributes into a single feature  
  4. Perform text preprocessing (lowercasing, stopword removal, punctuation removal)  
  5. Apply TF-IDF vectorization to convert text into numerical form  
  6. Reduce dimensionality using Truncated SVD  
  7. Apply K-Means clustering to group similar content  
  8. Display and analyze the clustered output  
This step-by-step process ensures meaningful grouping of Netflix content based on textual similarity.

## Tools and Libraries
  - Python  
  - Pandas  
  - NumPy  
  - NLTK  
  - Scikit-learn 

## Technologies Used
  - Machine Learning  
  - Natural Language Processing (NLP)  
  - Unsupervised Learning  
  - Data Preprocessing  

## Model Performance
  The clustering performance is evaluated using the Silhouette Score, which measures how well the data points are grouped within clusters.  
  The obtained score indicates that the clusters formed are reasonably well-separated and internally consistent.

## Result
  The final output of the project displays Netflix content grouped into different clusters.  
  A sample result shows the title, content type, genre, and assigned group ID, confirming that similar content is placed within the same cluster.  
  This validates the effectiveness of the clustering approach used in the project.
  
## Key Outcomes
  - Successfully clustered Netflix movies and TV shows based on textual similarity  
  - Demonstrated the use of unsupervised learning on real-world data  
  - Created a modular and structured AI project suitable for academic submission  
  - Improved understanding of NLP-based content analysis  

## Conclusion
  This project demonstrates how unsupervised machine learning and NLP techniques can be effectively used to analyze and organize streaming platform content.  
  The clustering results provide valuable insights into content similarity and can support recommendation and content discovery systems in the future.

## Author
  Name        : Thotakura Anudeepthi 
  Project     : Artificial Intelligence Final Project 
  Institution : KL UNIVERSITY
## Author
**Anudeepthi**
