
import pandas as pd
from pathlib import Path
import config

def load_stores():
    stores_path = Path("c:/Users/jdh03/OneDrive/바탕 화면/내미래..너가책임지거라/멋사NLP/멋사 AI NLP 3기/FinalProject/Agent/data/raw/stores.csv")
    
    try:
        df = pd.read_csv(stores_path, encoding='cp949')
    except:
        df = pd.read_csv(stores_path, encoding='utf-8')

    jangter_x = config.JANGTER_LOCATION["x"]
    jangter_y = config.JANGTER_LOCATION["y"]
    
    df['distance'] = ((df['x'] - jangter_x) ** 2 + (df['y'] - jangter_y) ** 2) ** 0.5
    
    df_filtered = df[
        (df['카테고리'].str.contains('국밥|한식', na=False)) &
        (df['distance'] < 0.01)
    ].copy()
    
    df_jangter = df_filtered[df_filtered['장소명'] == '장터국밥']
    df_others = df_filtered[df_filtered['장소명'] != '장터국밥'].head(5)
    
    with open("debug_stores.txt", "w", encoding="utf-8") as f:
        f.write("Selected Stores:\n")
        f.write(f"- {df_jangter['장소명'].iloc[0]}\n")
        
        for idx, row in df_others.iterrows():
            f.write(f"- {row['장소명']} ({row['카테고리']})\n")
    
    print("Done writing to debug_stores.txt")

if __name__ == "__main__":
    load_stores()
