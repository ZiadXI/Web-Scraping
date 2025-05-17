import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import io
from collections import Counter

# Page configuration
st.set_page_config(page_title="IMDB Analysis 🎬", page_icon="🎥", layout="wide")

# Custom CSS (condensed)
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #F5C518; text-align: center;}
    .sub-header {font-size: 1.8rem; font-weight: 600; color: #121212; border-bottom: 2px solid #F5C518;}
    .card {background-color: #f9f9f9; border-radius: 0.5rem; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# Function to convert runtime strings to minutes
def convert_runtime_to_minutes(runtime_str):
    if pd.isna(runtime_str):
        return np.nan
    
    # Convert string format like "2h 30min" or "1 hour 45 minutes" to minutes
    hours = 0
    minutes = 0
    
    # Check for hours
    hour_match = re.search(r'(\d+)\s*(?:h|hour|hr)', str(runtime_str), re.IGNORECASE)
    if hour_match:
        hours = int(hour_match.group(1))
    
    # Check for minutes
    minute_match = re.search(r'(\d+)\s*(?:m|min|minute)', str(runtime_str), re.IGNORECASE)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    total_minutes = hours * 60 + minutes
    return total_minutes if total_minutes > 0 else np.nan

@st.cache_data
def load_data():
    url ='https://raw.githubusercontent.com/Ahmedhossam447/imdbapp/refs/heads/main/imdb_top_250.csv'
    try:
        df = pd.read_csv(url)
        
        # Clean and transform data
        df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in re.split(',|;', str(x))] if isinstance(x, str) else x)
        df['Director'] = df['Director'].apply(lambda x: str(x).strip())
        
        # Add decade column
        df['Decade'] = (df['Year'] // 10) * 10
        
        # Convert runtime to minutes (add this new code)
        if 'Runtime' in df.columns:
            df['Runtime_Minutes'] = df['Runtime'].apply(convert_runtime_to_minutes)
        
        return df
    except Exception as e:
        st.error(f"⚠️ حصلت مشكلة في تحميل البيانات من GitHub: {e}")
        return pd.DataFrame()

# Load the data
df = load_data()


# Sidebar filters
with st.sidebar:
    st.markdown('<div style="text-align:center"><h1 style="color:#F5C518;">🎬 IMDB Analyzer</h1></div>', unsafe_allow_html=True)
    
    option = st.radio(
        "Choose Analysis:",
        ['📊 Rating Distribution', '🎬 Top Directors', '🎭 Genre Analysis', 
         '📅 Decade Trends', '⏱️ Runtime Analysis', '🔍 Custom Search']
    )
    
    # Year range filter
    year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
    year_range = st.slider("Filter by Year Range", year_min, year_max, (year_min, year_max))
    
    # Genre filter
    all_genres = sorted(list(set([genre for sublist in df['Genre'] for genre in sublist if isinstance(genre, str)])))
    selected_genres = st.multiselect("Filter by Genres", all_genres)
    
    # Rating filter
    rating_range = st.slider("Filter by Rating", float(df['Rating'].min()), float(df['Rating'].max()), 
                            (float(df['Rating'].min()), float(df['Rating'].max())))
    
    # Runtime filter (add this if Runtime_Minutes exists)
    if 'Runtime_Minutes' in df.columns:
        runtime_min = int(df['Runtime_Minutes'].min()) if not df['Runtime_Minutes'].isnull().all() else 0
        runtime_max = int(df['Runtime_Minutes'].max()) if not df['Runtime_Minutes'].isnull().all() else 300
        runtime_range = st.slider("Filter by Runtime (minutes)", runtime_min, runtime_max, (runtime_min, runtime_max))
    
    # Apply filters
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) & 
                    (df['Rating'] >= rating_range[0]) & (df['Rating'] <= rating_range[1])]
    
    # Apply runtime filter if available
    if 'Runtime_Minutes' in df.columns:
        filtered_df = filtered_df[(filtered_df['Runtime_Minutes'].isnull()) | 
                                ((filtered_df['Runtime_Minutes'] >= runtime_range[0]) & 
                                 (filtered_df['Runtime_Minutes'] <= runtime_range[1]))]
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['Genre'].apply(lambda x: any(genre in x for genre in selected_genres) if isinstance(x, list) else False)]
    
    # Download filtered data
    if not filtered_df.empty:
        buffer = io.BytesIO()
        filtered_df.to_csv(buffer, index=False)
        st.download_button("📥 Download Filtered Data", buffer, "imdb_filtered.csv", "text/csv")

# Main content
st.markdown('<h1 class="main-header">📽️ IMDB Top Movies Analysis</h1>', unsafe_allow_html=True)

# Display movie count info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Movies", len(filtered_df))
with col2:
    avg_rating = round(filtered_df['Rating'].mean(), 2) if not filtered_df.empty else 0
    st.metric("Average Rating", f"{avg_rating} ⭐")
with col3:
    if 'Runtime_Minutes' in filtered_df.columns and not filtered_df.empty and not filtered_df['Runtime_Minutes'].isnull().all():
        avg_runtime = round(filtered_df['Runtime_Minutes'].mean(), 0)
        st.metric("Average Runtime", f"{int(avg_runtime)} min")
    else:
        st.metric("Average Runtime", "N/A")
with col4:
    if not filtered_df.empty:
        top_genre = Counter([g for sublist in filtered_df['Genre'] for g in sublist if isinstance(g, str)]).most_common(1)
        if top_genre:
            st.metric("Top Genre", top_genre[0][0])
        else:
            st.metric("Top Genre", "N/A")
    else:
        st.metric("Top Genre", "N/A")

# Analyses based on selection
if option == '📊 Rating Distribution' and not filtered_df.empty:
    st.markdown('<h2 class="sub-header">⭐ Rating Distribution Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Rating Distribution")
        
        fig = px.histogram(filtered_df, x='Rating', nbins=20, color_discrete_sequence=['#F5C518'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Ratings by Year")
        
        fig = px.scatter(filtered_df, x='Year', y='Rating', color='Rating', 
                        hover_name='Title', hover_data=['Director', 'Year'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Average Rating by Decade")
    
    ratings_by_decade = filtered_df.groupby('Decade')['Rating'].mean().reset_index()
    movie_counts = filtered_df.groupby('Decade').size().reset_index(name='Count')
    decade_data = pd.merge(ratings_by_decade, movie_counts, on='Decade')
    
    fig = px.bar(decade_data, x='Decade', y='Rating', text='Rating', color='Count')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif option == '🎬 Top Directors' and not filtered_df.empty:
    st.markdown('<h2 class="sub-header">🎬 Director Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Most Featured Directors")
        
        top_directors = filtered_df['Director'].value_counts().reset_index()
        top_directors.columns = ['Director', 'Movie Count']
        top_directors = top_directors.head(10)
        
        fig = px.bar(top_directors, y='Director', x='Movie Count', orientation='h', 
                    color='Movie Count', text='Movie Count')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Directors with Highest Average Ratings")
        
        director_avg_ratings = filtered_df.groupby('Director')['Rating'].agg(['mean', 'count']).reset_index()
        director_avg_ratings = director_avg_ratings[director_avg_ratings['count'] >= 2]
        director_avg_ratings = director_avg_ratings.sort_values(by='mean', ascending=False).head(10)
        director_avg_ratings.columns = ['Director', 'Average Rating', 'Movie Count']
        
        fig = px.bar(director_avg_ratings, y='Director', x='Average Rating', orientation='h',
                    color='Movie Count', text='Average Rating')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif option == '🎭 Genre Analysis' and not filtered_df.empty:
    st.markdown('<h2 class="sub-header">🎭 Genre Analysis</h2>', unsafe_allow_html=True)
    
    # Extract all genres
    all_genres = [genre for sublist in filtered_df['Genre'] for genre in sublist if isinstance(genre, str)]
    genre_counts = pd.Series(all_genres).value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    # Calculate average rating by genre
    genre_ratings = {}
    for genre in set(all_genres):
        genre_movies = filtered_df[filtered_df['Genre'].apply(lambda x: genre in x if isinstance(x, list) else False)]
        genre_ratings[genre] = genre_movies['Rating'].mean()
    
    genre_avg_ratings = pd.DataFrame({'Genre': list(genre_ratings.keys()), 'Average Rating': list(genre_ratings.values())})
    genre_data = pd.merge(genre_counts, genre_avg_ratings, on='Genre')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Genre Distribution")
        
        fig = px.bar(genre_data.head(15), x='Count', y='Genre', orientation='h', 
                    color='Average Rating', text='Count')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Average Rating by Genre")
        
        fig = px.bar(genre_data.head(15).sort_values(by='Average Rating', ascending=False),
                    x='Average Rating', y='Genre', orientation='h', color='Count', text='Average Rating')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif option == '📅 Decade Trends' and not filtered_df.empty:
    st.markdown('<h2 class="sub-header">📅 Decade Trends Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate metrics by decade
    decade_cols = ['Decade', 'Avg Rating', 'Median Rating', 'Movie Count']
    if 'Runtime_Minutes' in filtered_df.columns and not filtered_df['Runtime_Minutes'].isnull().all():
        decade_aggs = {
            'Rating': ['mean', 'median', 'count'],
            'Runtime_Minutes': ['mean', 'median']
        }
        decade_cols.extend(['Avg Runtime', 'Median Runtime'])
    else:
        decade_aggs = {
            'Rating': ['mean', 'median', 'count']
        }
    
    decade_stats = filtered_df.groupby('Decade').agg(decade_aggs).reset_index()
    decade_stats.columns = decade_cols
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Movies per Decade")
        
        fig = px.bar(decade_stats, x='Decade', y='Movie Count', color='Avg Rating', text='Movie Count')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Rating Trends by Decade")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decade_stats['Decade'], y=decade_stats['Avg Rating'],
                                mode='lines+markers', name='Average Rating', line=dict(color='#F5C518', width=3)))
        fig.add_trace(go.Scatter(x=decade_stats['Decade'], y=decade_stats['Median Rating'],
                                mode='lines+markers', name='Median Rating', line=dict(color='#121212', width=3)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add runtime trend by decade if available
    if 'Avg Runtime' in decade_stats.columns:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Runtime Trends by Decade")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=decade_stats['Decade'], y=decade_stats['Avg Runtime'],
                                mode='lines+markers', name='Average Runtime', line=dict(color='#F5C518', width=3)))
        fig.add_trace(go.Scatter(x=decade_stats['Decade'], y=decade_stats['Median Runtime'],
                                mode='lines+markers', name='Median Runtime', line=dict(color='#121212', width=3)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif option == '⏱️ Runtime Analysis' and 'Runtime_Minutes' in filtered_df.columns and not filtered_df['Runtime_Minutes'].isnull().all():
    st.markdown('<h2 class="sub-header">⏱️ Runtime Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Runtime Distribution")
        
        fig = px.histogram(filtered_df.dropna(subset=['Runtime_Minutes']), x='Runtime_Minutes', 
                           nbins=20, color_discrete_sequence=['#F5C518'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Runtime vs. Rating")
        
        fig = px.scatter(filtered_df.dropna(subset=['Runtime_Minutes']), x='Runtime_Minutes', y='Rating', color='Rating',
                        hover_name='Title', hover_data=['Director', 'Year'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Runtime Records")
    
    records_col1, records_col2 = st.columns(2)
    
    with records_col1:
        st.subheader("Longest Movies")
        longest_movies = filtered_df.dropna(subset=['Runtime_Minutes']).sort_values(by='Runtime_Minutes', ascending=False).head(5)
        st.dataframe(longest_movies[['Title', 'Director', 'Year', 'Runtime_Minutes', 'Rating']], use_container_width=True)
    
    with records_col2:
        st.subheader("Shortest Movies")
        shortest_movies = filtered_df.dropna(subset=['Runtime_Minutes']).sort_values(by='Runtime_Minutes').head(5)
        st.dataframe(shortest_movies[['Title', 'Director', 'Year', 'Runtime_Minutes', 'Rating']], use_container_width=True)
    
    # Add a runtime by genre visualization
    st.subheader("Average Runtime by Genre")
    
    # Calculate average runtime by genre
    genre_runtimes = {}
    for genre in set([g for sublist in filtered_df['Genre'] for g in sublist if isinstance(g, str)]):
        genre_movies = filtered_df[filtered_df['Genre'].apply(lambda x: genre in x if isinstance(x, list) else False)]
        if not genre_movies['Runtime_Minutes'].isnull().all():
            genre_runtimes[genre] = genre_movies['Runtime_Minutes'].mean()
    
    genre_runtime_df = pd.DataFrame({'Genre': list(genre_runtimes.keys()), 'Average Runtime': list(genre_runtimes.values())})
    genre_runtime_df = genre_runtime_df.sort_values(by='Average Runtime', ascending=False).head(15)
    
    fig = px.bar(genre_runtime_df, y='Genre', x='Average Runtime', orientation='h', 
                color='Average Runtime', text='Average Runtime')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif option == '🔍 Custom Search' and not filtered_df.empty:
    st.markdown('<h2 class="sub-header">🔍 Custom Search</h2>', unsafe_allow_html=True)
    
    search_term = st.text_input("Search for movies by title, director, or year")
    
    if search_term:
        results = filtered_df[
            filtered_df['Title'].str.contains(search_term, case=False, na=False) |
            filtered_df['Director'].str.contains(search_term, case=False, na=False) |
            filtered_df['Year'].astype(str).str.contains(search_term, na=False)
        ]
        
        if not results.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"Found {len(results)} movies matching '{search_term}'")
            
            # Include Runtime_Minutes in display if available
            display_columns = ['Title', 'Director', 'Year', 'Rating']
            if 'Runtime_Minutes' in results.columns:
                display_columns.append('Runtime_Minutes')
            display_columns.append('Genre')
            
            st.dataframe(results[display_columns], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if len(results) == 1:
                movie = results.iloc[0]
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f"Details for {movie['Title']} ({movie['Year']})")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(f"**Director:** {movie['Director']}")
                    st.markdown(f"**Rating:** {movie['Rating']} ⭐")
                    st.markdown(f"**Year:** {movie['Year']}")
                
                with detail_col2:
                    if 'Runtime_Minutes' in movie and not pd.isna(movie['Runtime_Minutes']):
                        st.markdown(f"**Runtime:** {int(movie['Runtime_Minutes'])} minutes")
                    st.markdown(f"**Genres:** {', '.join(movie['Genre']) if isinstance(movie['Genre'], list) else movie['Genre']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"No movies found matching '{search_term}'")
else:
    if filtered_df.empty:
        st.warning("No data available with current filters. Please adjust your filters or load a valid dataset.")
    elif option == '⏱️ Runtime Analysis' and ('Runtime_Minutes' not in filtered_df.columns or filtered_df['Runtime_Minutes'].isnull().all()):
        st.warning("Runtime data is not available or all values are missing. Please check the dataset.")
