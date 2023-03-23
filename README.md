# **Forecast of research trends for Knowledge Representation**

Time series analysis techniques to predict research trends based on their past behaviour around Knowledge Representation. 
Knowledge Representation is a part of artificial intelligence that describes how we can represent knowledge in artificial intelligence 
so that an intelligent machine can learn from knowledge and experiences so that it can behave intelligently like a human 
and solve real-world problems. 

## **Installation**

1. Pandas
2. NumPy
3. Statsmodels
4. Matplotlib
5. Sklearn  
6. Seaborn
7. Math

## **Data**

For this paper, we have drawn a dataset from the DBLP website (https://dblp.uni-trier.de/), particularly from 
the file (dblp-2021-02- 01.xml.gz). This file contains a large number of publications about the science of 
computer science. To extract the data, we created a Python parser, which reads each line looking for the 
tag <title>, checks if it is in a list of words related to the subject area we are studying and then searches in 
the following four lines for the tag <year> and adds the title to the data if it does not exist.





