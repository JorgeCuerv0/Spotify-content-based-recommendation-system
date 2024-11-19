# **Content-Based Song Recommendation System**

This Python program is a content-based recommendation system for songs, built using Spotify's **Million Playlist Dataset (MPD)**. It recommends similar tracks based on track names, artist names, album names, and other playlist features.

## **Features**
- **Data Ingestion**: Reads playlists from a JSON file and converts them into a structured DataFrame.
- **Feature Engineering**: Processes textual and numerical features to create a "combined feature" for each track.
- **TF-IDF Vectorization**: Converts text data into numerical vectors for cosine similarity calculation.
- **Cosine Similarity**: Measures the similarity between tracks to generate recommendations.
- **Non-Repeating Recommendations**: Ensures the recommended track names are unique.

---

## **Setup Instructions**

### **Prerequisites**
- Python 3.10 or higher
- Required libraries:
  - `pandas`
  - `scikit-learn`
  - `datetime`
  - `re`

### **Installation**

Clone the repository:
```bash
git clone https://github.com/yourusername/song-recommender.git
cd song-recommender
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## **Usage**

### **Run the Program**
1. Update the dataset path in `main.py` to point to your MPD JSON file.
2. Execute the program:
   ```bash
   python main.py
   ```
## **Input and Output**

### **Input**
- The name of a song (e.g., `"Toxic"`).

### **Output**
A list of similar songs with:
- **Track Name**
- **Artist Name**
- **Album Name**

### **Example Output**
```bash
Recommendations for: **Toxic**
--------------------------------------------------
1. Track: 3
   Artist: britney spears
   Album: The Singles Collection
--------------------------------------------------
2. Track: me against the music  lp version  video mix
   Artist: britney spears
   Album: In The Zone
--------------------------------------------------
3. Track: lucky
   Artist: britney spears
   Album: Oops!... I Did It Again
--------------------------------------------------
4. Track: stronger
   Artist: britney spears
   Album: Oops!... I Did It Again
--------------------------------------------------
```
## **How It Works**

### **Data Ingestion**
- **Loads playlist data** from the provided MPD JSON file.
- **Extracts information** about tracks, artists, albums, and playlists.

### **Feature Engineering**
- **Combines features** like track names, artist names, and album names into a single feature string.
- **Adds normalized playlist diversity scores and ages** for additional context.

### **TF-IDF and Similarity Calculation**
- Uses **TF-IDF vectorization** to numerically represent textual data.
- Computes **cosine similarity** between tracks for comparison.

### **Recommendation**
- **Filters similar tracks** based on cosine similarity.
- Ensures **no duplicate track names** appear in the final recommendation list.

---

## **Customizing the Program**

### **Dataset**
- Replace the file path in `main.py` to use your own MPD slice.

### **Number of Recommendations**
- Modify the `top_n` parameter in the `recommend_tracks` function.

## **How It Works**

### **Data Ingestion**
- **Loads playlist data** from the provided MPD JSON file.
- **Extracts information** about tracks, artists, albums, and playlists.

### **Feature Engineering**
- **Combines features** like track names, artist names, and album names into a single feature string.
- **Adds normalized playlist diversity scores and ages** for additional context.

### **TF-IDF and Similarity Calculation**
- Uses **TF-IDF vectorization** to numerically represent textual data.
- Computes **cosine similarity** between tracks for comparison.

### **Recommendation**
- **Filters similar tracks** based on cosine similarity.
- Ensures **no duplicate track names** appear in the final recommendation list.

---

## **Customizing the Program**

### **Dataset**
- Replace the file path in `main.py` to use your own MPD slice.

### **Number of Recommendations**
- Modify the `top_n` parameter in the `recommend_tracks` function.

---

## **Contributing**
- Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.