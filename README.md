# Movie Recommender System

A content-based movie recommendation system built using the TMDB 5000 movies dataset. This system recommends movies based on content similarity including genres, keywords, cast, and crew information.

## 📁 Repository Structure

```
MovieRecommender/
├── README.md
├── MovieRecommenderSystem.ipynb    # Main implementation notebook
├── tmdb_5000_movies.csv           # Movies dataset (download required)
└── tmdb_5000_credits.csv          # Credits dataset (download required)
```

## 🎯 Overview

This Jupyter Notebook implements a **content-based movie recommender system** that analyzes movie features like genres, keywords, cast, and crew to suggest similar movies. The system uses cosine similarity to find movies with similar content characteristics.

## 📊 Dataset

The project uses the TMDB 5000 Movie Dataset which includes:
- **tmdb_5000_movies.csv**: Movie information including genres, keywords, overview, etc.
- **tmdb_5000_credits.csv**: Cast and crew information

### Download Dataset
You can download the dataset from:
- [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Place both CSV files in the root directory of the project.

## 🔧 Requirements

This notebook requires Python 3.8+ and the following packages:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

### Core Dependencies
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `seaborn` - Statistical data visualization
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Machine learning utilities (likely for cosine similarity)

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bruteforce1127/MovieRecommender.git
   cd MovieRecommender
   ```

2. **Download and place the dataset files:**
   - Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
   - Place them in the project root directory

3. **Install dependencies:**
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook MovieRecommenderSystem.ipynb
   ```
   or
   ```bash
   jupyter lab MovieRecommenderSystem.ipynb
   ```

5. **Run all cells sequentially** or restart kernel and run all cells.

## 📋 Notebook Structure

The notebook follows this workflow:

1. **Library Imports** - Import essential libraries (numpy, pandas, seaborn, matplotlib)
2. **Data Loading** - Load TMDB movies and credits datasets
3. **Data Merging** - Combine datasets on movie title
4. **Feature Selection** - Extract relevant columns (movie_id, title, overview, genres, keywords, cast, crew)
5. **Data Cleaning** - Handle missing values and null entries
6. **Preprocessing** - Process genres and other categorical features
7. **Feature Engineering** - Create content-based features for similarity calculation
8. **Model Building** - Implement content-based recommendation algorithm
9. **Recommendation Generation** - Generate movie recommendations based on content similarity

## 🎬 Expected Outputs

- **Data Analysis**: Overview of dataset structure and missing values
- **Preprocessed Features**: Cleaned and processed movie features
- **Recommendation Function**: A working movie recommendation system
- **Sample Recommendations**: Example recommendations for selected movies

## 🔄 Algorithm Approach

This is a **content-based filtering** approach that:
- Uses movie metadata (genres, keywords, cast, crew, overview)
- Calculates similarity between movies based on content features
- Recommends movies with highest content similarity scores

## 🛠️ Customization

You can extend this project by:
- Adding more content features (production companies, release year)
- Implementing hybrid recommendation (content + collaborative filtering)
- Adding user rating predictions
- Building a web interface using Streamlit or Flask
- Implementing deep learning-based embeddings

## 📈 Performance Considerations

- The system works well for movies with rich metadata
- Performance depends on the quality of preprocessing and feature engineering
- Content-based systems avoid the cold start problem for new movies

## 🐛 Troubleshooting

- **Missing dataset files**: Ensure both CSV files are in the project root
- **Import errors**: Install missing packages using pip
- **Memory issues**: The dataset is relatively small (~5000 movies) but ensure sufficient RAM
- **File path errors**: Verify CSV files are named exactly as expected

## 📜 License

This project is open source. Please cite the TMDB dataset if using in academic work.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📞 Contact

**Developer**: bruteforce1127  
**GitHub**: [bruteforce1127](https://github.com/bruteforce1127)

---
