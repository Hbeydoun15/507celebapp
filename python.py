# app.py

import os, re, unicodedata
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ——— Page config ———————————————
st.set_page_config(
    page_title="🎤  Social Media Fame and accolades ", page_icon="🚀", layout="centered"
)

st.markdown(
    """
<style>
  body {background-color: #f5f7fa;}
  h1 {font-family:'Comic Sans MS', cursive; color:#4A90E2;}
  .stDataFrame {border:1px solid #ddd; border-radius:8px;}
</style>
""",
    unsafe_allow_html=True,
)

# ——— Load & prepare data ———————————————

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    insta = pd.read_csv(
        os.path.join(BASE_DIR, "top_insta_influencers_data.csv"), low_memory=False
    )
    spotify = pd.read_csv(
        os.path.join(BASE_DIR, "Final database.csv"), low_memory=False
    )
    grammy = pd.read_csv(os.path.join(BASE_DIR, "grammy_winners.csv"), low_memory=False)
    return insta, spotify, grammy


insta, spotify, grammy = load_data()

#Cleans and normalizes artist names so they can be merged.
# Parse counts
def parse_count(x):
    if isinstance(x, str):
        s = x.strip().lower().replace(",", "")
        if s.endswith("m"):
            return int(float(s[:-1]) * 1e6)
        if s.endswith("k"):
            return int(float(s[:-1]) * 1e3)
        return int(float(s))
    return int(x)


insta["followers"] = insta["followers"].apply(parse_count)
insta.rename(columns={"channel_info": "artist_name"}, inplace=True)
spotify.rename(
    columns={"Artist": "artist_name", "Artist_followers": "monthly_listeners"},
    inplace=True,
)
grammy.rename(columns={"artist": "artist_name"}, inplace=True)

# Extract primary
spotify["artist_name"] = (
    spotify["artist_name"]
    .astype(str)
    .apply(
        lambda s: re.split(r"\s*(?:&|feat\.?|featuring)\s*", s, flags=re.IGNORECASE)[
            0
        ].strip()
    )
)
spotify["monthly_listeners"] = pd.to_numeric(
    spotify["monthly_listeners"], errors="coerce"
).fillna(0)


# Normalize names
def normalize_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"^the\s+", "", s.lower().strip())
    return re.sub(r"[^\w]", "", s)


for df_ in (insta, spotify, grammy):
    df_["artist_name"] = df_["artist_name"].astype(str).apply(normalize_name)

# Aggregations:Aggregates each artist’s follower count, listener count, and award count.
insta["centrality"] = insta["followers"]
spotify_agg = spotify.groupby("artist_name")["monthly_listeners"].mean().reset_index()
grammy_counts = grammy.groupby("artist_name").size().reset_index(name="num_awards")

df = (
    insta[["artist_name", "centrality"]]
    .merge(spotify_agg, on="artist_name", how="left")
    .merge(grammy_counts, on="artist_name", how="left")
    .fillna(0)
)

# ——— UI header ———————————————
st.markdown(
    "<h1 style='text-align:center;'>🚀 UpNext Dashboard</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Discover emerging talent with social & career insights</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ——— Select artists ———————————————
choices = sorted(df.artist_name.unique())
sel = st.multiselect("Pick artists to compare:", choices)
if not sel:
    st.warning("Select at least one artist!")
    st.stop()

sub = df[df.artist_name.isin(sel)]
display = sub.copy()
display["artist"] = display["artist_name"].str.title()
display = display[["artist", "centrality", "monthly_listeners", "num_awards"]]

st.subheader("Selected Artist Data")
st.dataframe(display.set_index("artist"))

# Formatter
fmt = FuncFormatter(
    lambda x, pos: (
        f"{x * 1e-6:.1f}M"
        if x >= 1e6
        else f"{x * 1e-3:.1f}k"
        if x >= 1e3
        else f"{int(x)}"
    )
)

# ——— Comparison Charts: appropriate 6×4 size ——————————————————

st.subheader("Followers")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(display["artist"], display["centrality"], color="#89CFF0")
ax.yaxis.set_major_formatter(fmt)
ax.tick_params(labelrotation=45)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Monthly Listeners")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(display["artist"], display["monthly_listeners"], color="#FFC0CB")
ax.yaxis.set_major_formatter(fmt)
ax.tick_params(labelrotation=45)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Grammy Awards")
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(display["artist"], display["num_awards"], color="#89CFF0")
ax.tick_params(labelrotation=45)
plt.tight_layout()
st.pyplot(fig)
# ——— Similar Artists Finder —————————————————————————————
st.subheader("🔍 Find Top-5 Similar Artists")

# 1) Text input
artist_input = st.text_input("Enter an artist name to find similar acts:")

if artist_input:
    # normalize exactly the same way you did before
    norm_input = normalize_name(artist_input)

    if norm_input not in df.artist_name.values:
        st.error("Artist not found—try typing the exact spelling (e.g. 'beyonce').")
    else:
        # 2) Build the feature matrix & NN model
        features = df[["centrality", "monthly_listeners", "num_awards"]]
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
        nn.fit(X)

        # 3) Query the model
        idx = df.index[df.artist_name == norm_input][0]
        dists, ids = nn.kneighbors([X[idx]])

        # ids[0] is an array of 6 indices: the first will be itself, so skip it
        similar = df.iloc[ids[0][1:]][
            ["artist_name", "centrality", "monthly_listeners", "num_awards"]
        ].copy()

        # title-case for display
        similar["artist"] = similar["artist_name"].str.title()
        similar = similar[["artist", "centrality", "monthly_listeners", "num_awards"]]

        st.write(f"Top-5 artists similar to **{artist_input.title()}**:")
        st.table(similar.set_index("artist"))
# ——— 11. Top-40 Leaderboards & Intersection —————————————————————————————
st.subheader("🏆 Top 40 by Category (Global)")

# A) Top-40 Followers (raw Instagram)
top_followers = (
    insta[["artist_name", "centrality"]]
    .sort_values("centrality", ascending=False)
    .head(40)
)

# B) Top-40 Grammy Awards
valid = set(spotify_agg.artist_name)
top_grammys = (
    grammy_counts[grammy_counts.artist_name.isin(valid)]
    .sort_values("num_awards", ascending=False)
    .head(40)
)


# Utility to show a leaderboard
def show_leaderboard(df_lead, name_col, val_col, title, fmt):
    disp = df_lead.copy()
    disp[name_col] = disp[name_col].str.title()
    disp[val_col] = disp[val_col].map(fmt)
    st.markdown(f"**Top 40 {title}**")
    st.dataframe(
        disp.rename(
            columns={name_col: "Artist", val_col: title.replace(" ", "_")}
        ).reset_index(drop=True)
    )


comma_fmt = lambda x: f"{int(x):,}"

show_leaderboard(
    top_followers, "artist_name", "centrality", "Instagram Followers", comma_fmt
)
show_leaderboard(top_grammys, "artist_name", "num_awards", "Grammy Awards", comma_fmt)

# Intersection of the two top-40s
common = set(top_followers.artist_name) & set(top_grammys.artist_name)

st.subheader("🌟 In Both Top-40 Followers & Top-40 Grammy Awards")
if common:
    for name in sorted(common):
        st.write(f"- {name.title()}")
else:
    st.write("None")
