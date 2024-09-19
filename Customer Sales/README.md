Customer Segmentation with K-Means: Understanding the Customer Closer
Building a K-Means clustering algorithm for customer segmentation in Python
Image by Jopwell from PexelsIntroduction
Have you ever wondered why ads that appear on the internet sometimes feel so relevant? That's because many businesses use Customer Segmentation to understand their customers. Customer segmentation is the process of grouping customers based on similar characteristics, such as age, income, and spending habits.
In this blog post, I will share about my experience of doing customer segmentation using K-Means Clustering. The goal? To develop a more targeted marketing strategy and offer more personalized services to each customer segment.
Project Objective
In this project, I use a dataset that contains customer information, including Gender, Age, Income, Spending, and product category. The main goal is categorizing customers into segments based on their spending patterns. By doing so, businesses can direct appropriate marketing campaigns to each segment and improve the effectiveness of sales strategies.
Methods Used: K-Means Clustering
For customer segmentation, I used K-Means Clustering. This method is popular because it is fast and easy to use to group data based on similarities between data points.
How does K-Means work?
Simply put, K-Means works by dividing data into a number of clusters (k) based on their similarity. K-Means is like sorting something into piles. Imagine you have a bunch of different objects, and you want to organize them into piles where each pile contains similar objects. K-Means helps you figure out which objects go into each pile by adjusting the piles repeatedly until they all fit.
Now let's get started
Import Library and Reading Dataset
The following lines of code are for importing the required libraries and reading the data set:
#LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
#Reading a Dataset
df = pd.read_csv('customer_data.csv')
df.head()
Now, let's take a look at the head of the data frame:
Customer DatasetMy dataset contains customer data with the following columns:
Customer ID: Customer ID
Gender: Customer gender
Age: Customer's age
Income: Annual income
Spending: This shows how much the customer spends compared to the average.
Category: Product category

Why did I choose this dataset? This data gives a pretty complete picture of who our customers are, how they spend, and whether they tend to spend more or less. This is important to understand the needs of each segment.
I'm not going to use the "Customer ID" field so we'll just delete it.
df = df.drop(columns='Customer ID')
Standardize Data
Income and expense data have different scales, so I standardize the data so that the clustering process is more accurate and the data has the same scale.
columns_to_scale = ['Age', 'Income', 'Spending']
data_to_scale = df[columns_to_scale]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
scaled_df.index = df.index
df[columns_to_scale] = scaled_df

Now, let's see the results
Standardize DataDetermining the Number of Clusters (K)
Using the Elbow method, I tried various numbers of clusters to determine the optimal k value.
cluster = []

for i in range (1,11):
    kmeans = KMeans(n_clusters = i, random_state=42)
    kmeans.fit(df)
    cluster.append(kmeans.inertia_)
plt.plot(range(1,11), cluster, marker='o')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.title('ELBOW METHOD')
plt.show()
Visualize model performance:
Elbow MethodBased on the elbow plot above, it can be seen that the optimal number of clusters is 4. So, in this case, I used 4 clusters to give the best results.
Running the K-Means Algorithm
After determining the number of clusters, I ran the K-Means algorithm to group the customers into 4 main segments.
#Choose K=4 based on elbow method 
kmeans = KMeans(n_clusters = 4, random_state=42)
kmeans.fit(df)
cluster_labels = kmeans.labels_

#Add cluster labels to the original data
df['Clusters'] = cluster_labels
Now, we will visualize the clustered segmentation using Principal Component Analysis (PCA). PCA allows us to reduce the dimensionality of the data while retaining as much variation as possible, making it easier to visualize clusters in a lower-dimensional space.
Let's visualize how customers are grouped into clusters based on the principal components obtained from their features.
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(df.drop('Cluster', axis=1, errors='ignore'))

df['PCA1'] = pca_components[:,0]
df['PCA2'] = pca_components[:,1]
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(df.drop(['Cluster', 'PCA1', 'PCA2'], axis=1, errors='ignore'))
df['Clusters'] = labels 
plt.figure(figsize=(10,5))
sns.scatterplot(x='PCA1', y='PCA2', data=df, palette='viridis', hue='Clusters')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.title('Cluster Customer Segmentation (PCA)')
plt.legend(title='Cluster', loc='upper right')
plt.show()
Visualization of results:
Cluster Customer Segmentation (PCA)Cluster Profiling 
Next, we will create a summary of the key characteristics of each cluster. This will help us understand the unique traits of each segment.
#Calculate the mean values of the features for each cluster
cluster_profile = df_new[['Age', 'Income', 'Spending', 'Clusters']].groupby('Clusters').mean()

#Add cluster sizes to the profile
cluster_profile['size'] = df_new['Clusters'].value_counts()
cluster_profile
Summary DatasetResults and Analysis
After running K-Means, I got 4 different customer segments. Here are the results of visualizing the segmentation in a scatter plot.
Segment Interpretation:
Segment 1 :
Average Age : 45 
Gender: Male 
Average Income: 59415 
Average Spending: 7748 
Most Category: Electronics

Segment 2 : 
Average Age: 52 
Gender: Female 
Average Income: 121714 
Average Spending: 3865 
Most Category: Funiture

Segment 3 : 
Average Age: 27 
Gender: Male 
Average Income: 114055 
Average Spending: 5968 
Most Category: Automotive

Segment 4 : 
Average Age: 41 
Gender: Female 
Average Income: 51263 
Average Spending: 2292 
Most Category: Electronics

Insights and Recommendations
Segment 1: Middle-Aged Men with Electronic Preferences
Insight: This segment consists of men with an average age of 45, a modest income (around 59,415), but a sizable spend (7,748) on electronics. They tend to shop for electronics, suggesting that they may be particularly interested in new technology or electronic products.
Strategies: Discount offers on electronic products, loyalty campaigns, or promotions on the latest technology products will attract their attention. Focus on increasing spending through product bundling or exclusive offers.

Segment 2: Older Women with Furniture Preferences
Insight: This segment consists of women with an average age of 52, high income (around 121,714), but relatively low spending (3,865), especially in the furniture category. They may be more selective in their shopping, looking for high quality items at reasonable prices.
Strategy: Campaigns that emphasize the quality, design and functionality of furniture can appeal to this segment. Personalization offers or premium experiential loyalty programs can also increase their loyalty to the brand.

Segment 3: Young Men with Automotive Preferences
Insight: Men with an average age of 27 years, high income (around 114,055), yet moderate spending (5,968), show interest in the automotive category. They may be interested in automotive vehicles or accessories.
Strategy: Campaigns targeting the latest automotive technology, car accessories, or vehicles with cutting-edge features can be very attractive to this segment. Credit offers or financing programs for vehicles can also encourage them to spend more.

Segment 4: Middle-Aged Women with Electronic Preferences
Insight: Women with an average age of 41, lower income (around 51,263), and low spending (2,292), especially on electronic products. They may be more frugal or price-sensitive, but still have an interest in electronics.
Strategy: Promotions with discount offers or affordable electronic products can be very effective for this segment. Value packs or products with essential yet affordable features can also catch their attention.

Conclusion:
The younger segment (Segment 3) tends to have high income and interest in more expensive categories such as automotive.
Older women (Segment 2) with high incomes may be more cautious in spending, while middle-aged women (Segment 4) are more price sensitive.
Men in the middle age segment (Segment 1) show a strong interest in electronics, which could be an opportunity for technology-based marketing campaigns.
