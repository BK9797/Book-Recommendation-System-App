import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load dataset
@st.cache_data
def load_data():
    books = pd.read_csv('Books.csv.gz', compression='gzip', usecols=["ISBN", "Book-Title", "Book-Author"])
    ratings = pd.read_csv("Ratings.csv", usecols=["User-ID", "ISBN", "Book-Rating"])
    return books, ratings

df_books, df_ratings = load_data()

def preprocess(df_books, df_ratings):
    df_books.dropna(inplace=True)
    
    # Calculate the count of ratings given by each user and store it in the 'ratings' Series
    user_ratings_count = df_ratings['User-ID'].value_counts()
    
    # Filter out books with fewer than 100 ratings
    valid_users = user_ratings_count[user_ratings_count >= 100].index
    df_ratings_filtered = df_ratings[df_ratings['User-ID'].isin(valid_users)]
    
    # Create a user-item matrix
    df_matrix = df_ratings_filtered.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0).T
    
    # Add book titles to the index
    df_matrix.index = df_matrix.index.map(df_books.set_index('ISBN')['Book-Title'])
    df_matrix = df_matrix.sort_index()

    return df_matrix

df_matrix = preprocess(df_books, df_ratings)

# Load trained model
@st.cache_resource
def load_model(df_matrix):
    model = NearestNeighbors(metric='cosine')
    model.fit(df_matrix.values)
    return model  # Return the trained model

model = load_model(df_matrix)  # Ensure this is properly assigned

def get_recommends(title=""):
    try:
        # Try to locate the book in the DataFrame
        book = df_matrix.loc[title]
    except KeyError:
        st.error(f"The given book '{title}' does not exist.")
        return None

    # Reshape book values to 2D (1 sample, n_features)
    book_values = book.values.reshape(1, -1)  # reshape to 2D (1, n_features)

    # Get recommendations using KNN (or your model's prediction method)
    distance, indice = model.kneighbors(book_values, n_neighbors=6)

    # Create DataFrame of recommendations
    recommended_books = pd.DataFrame({
        'title': df_matrix.iloc[indice[0]].index.values,
        'distance': distance[0]
    }).sort_values(by='distance', ascending=False).head(5)

    # Remove the input book from the recommendations list
    recommended_books = recommended_books[recommended_books['title'] != title]

    # Only return the titles, no distance column
    recommended_books = recommended_books[['title']].reset_index(drop=True)

    # Add index starting from 1
    recommended_books.index = recommended_books.index + 1

    return title, recommended_books

# Streamlit user interface
def main():
    st.title('📚 Book Recommendation System')

    # Input for book title
    title_input = st.text_input("Enter a book title:")

    # Add a button to trigger the recommendation process
    if st.button("Get Recommendations"):
        if title_input:
            # Generate recommendations
            result = get_recommends(title_input)
            if result:
                title, recommended_books = result
                st.write(f"Recommendations for: {title}")
                st.write(recommended_books)
        else:
            st.error("Please enter a book title.")

if __name__ == "__main__":
    main()
