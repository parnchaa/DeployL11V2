try:
    import streamlit as st
    import pickle
    import pandas as pd
    import numpy as np
    from myfunction import get_movie_recommendations
except ImportError as e:
    print(f"Missing required packages. Please install: pip install streamlit pandas numpy")
    print(f"Error: {e}")
    exit(1)

# Load data
@st.cache_data
def load_data():
    """Load recommendation data from pickle file"""
    try:
        with open('recommendation_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("---")
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Cannot load recommendation data. Please check your data file.")
        return
    
    # Extract data components (handle both dict and list data structures)
    try:
        st.sidebar.header("📊 Data Information")
        
        # Handle different data structures
        if isinstance(data, dict):
            st.sidebar.write("Data type: Dictionary")
            st.sidebar.write("Available data keys:")
            for key in data.keys():
                st.sidebar.write(f"- {key}")
            
            # Try to identify the correct dataframes
            user_similarity_df = None
            user_movie_ratings = None
            
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    st.sidebar.write(f"{key}: {value.shape}")
                    if 'similarity' in key.lower():
                        user_similarity_df = value
                    elif 'rating' in key.lower() or 'movie' in key.lower():
                        user_movie_ratings = value
            
            # If we couldn't auto-detect, use the first two dataframes
            if user_similarity_df is None or user_movie_ratings is None:
                dataframes = [v for v in data.values() if hasattr(v, 'shape') and len(v.shape) == 2]
                if len(dataframes) >= 2:
                    user_similarity_df = dataframes[0]
                    user_movie_ratings = dataframes[1]
        
        elif isinstance(data, list):
            st.sidebar.write("Data type: List")
            st.sidebar.write(f"Number of items: {len(data)}")
            
            # Assume first two items are the dataframes we need
            if len(data) >= 2:
                # Try to identify which is which based on shape or content
                for i, item in enumerate(data[:3]):  # Check first 3 items
                    if hasattr(item, 'shape'):
                        st.sidebar.write(f"Item {i}: shape {item.shape}")
                
                # Based on inspection: 
                # Item 0: User similarity matrix (610x610)
                # Item 1: User movie ratings matrix (610x9719) 
                matrices = [item for item in data if hasattr(item, 'shape') and len(item.shape) == 2]
                if len(matrices) >= 2:
                    # Identify by shape - square matrix is similarity, rectangular is ratings
                    for matrix in matrices:
                        if matrix.shape[0] == matrix.shape[1]:  # Square matrix = similarity
                            user_similarity_df = matrix
                        else:  # Rectangular matrix = ratings (users x movies)
                            user_movie_ratings = matrix
                    
                    # Fallback if we couldn't identify by shape
                    if user_similarity_df is None or user_movie_ratings is None:
                        user_similarity_df = data[0]    # First item
                        user_movie_ratings = data[1]    # Second item
                else:
                    user_movie_ratings = data[0] if len(data) > 0 else None
                    user_similarity_df = data[1] if len(data) > 1 else None
            else:
                st.error("Not enough data items in the list.")
                return
        
        else:
            st.error(f"Unsupported data type: {type(data)}")
            return
        
        if user_similarity_df is None or user_movie_ratings is None:
            st.error("Could not identify user similarity and movie ratings dataframes.")
            st.write("Available data structure:")
            st.write(f"Data type: {type(data)}")
            if isinstance(data, list):
                for i, item in enumerate(data):
                    st.write(f"Item {i}: {type(item)} - {getattr(item, 'shape', 'No shape attribute')}")
            return
            
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return
    
    # User interface
    st.header("🎯 Get Movie Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User selection
        available_users = list(user_movie_ratings.index)
        user_id = st.selectbox(
            "Select User ID:",
            options=available_users,
            index=0
        )
    
    with col2:
        # Number of recommendations
        n_recommendations = st.slider(
            "Number of Recommendations:",
            min_value=1,
            max_value=20,
            value=5
        )
    
    # Display current user's ratings (optional)
    if st.checkbox("Show current user's ratings"):
        st.subheader(f"📈 User {user_id}'s Current Ratings")
        user_ratings = user_movie_ratings.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].sort_values(ascending=False)
        
        if len(rated_movies) > 0:
            st.dataframe(rated_movies.head(10))
        else:
            st.write("This user hasn't rated any movies yet.")
    
    # Get recommendations button
    if st.button("🎬 Get Recommendations", type="primary"):
        try:
            with st.spinner("Generating recommendations..."):
                recommendations = get_movie_recommendations(
                    user_id=user_id,
                    user_similarity_df=user_similarity_df,
                    user_movie_ratings=user_movie_ratings,
                    n_recommendations=n_recommendations
                )
            
            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations!")
                
                st.subheader("🎯 Recommended Movies")
                
                # Display recommendations in a nice format
                for i, movie in enumerate(recommendations, 1):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**#{i}**")
                    with col2:
                        st.write(f"🎬 {movie}")
                
                # Optional: Show recommendation details
                if st.checkbox("Show recommendation details"):
                    st.subheader("📊 Recommendation Details")
                    rec_df = pd.DataFrame({
                        'Rank': range(1, len(recommendations) + 1),
                        'Movie': recommendations
                    })
                    st.dataframe(rec_df, use_container_width=True)
                    
            else:
                st.warning("No recommendations found for this user.")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.write("Please check your data format and function implementation.")
    
    # Additional information
    st.markdown("---")
    st.subheader("ℹ️ How it works")
    st.write("""
    1. **Select a user** from the dropdown menu
    2. **Choose the number of recommendations** you want
    3. **Click 'Get Recommendations'** to see personalized movie suggestions
    
    The system uses collaborative filtering to find users with similar tastes 
    and recommends movies that similar users have rated highly.
    """)
    
    # Data statistics
    with st.expander("📊 Dataset Statistics"):
        st.write(f"**Total Users:** {len(user_movie_ratings.index)}")
        st.write(f"**Total Movies:** {len(user_movie_ratings.columns)}")
        
        # Calculate some basic stats
        total_ratings = (user_movie_ratings > 0).sum().sum()
        avg_ratings_per_user = total_ratings / len(user_movie_ratings.index)
        
        st.write(f"**Total Ratings:** {total_ratings}")
        st.write(f"**Average Ratings per User:** {avg_ratings_per_user:.1f}")

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    main()

