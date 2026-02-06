"""
Create heatmap visualization of census data by age group and gender.
"""

import sys
from pathlib import Path
import pandas as pd
import folium
from folium import plugins

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_settings


def create_heatmap_by_category(df, coords, category_name, color_scheme='YlOrRd'):
    """Create a heatmap for a specific demographic category."""
    settings = get_settings()
    center_lat = (settings.area.lat_min + settings.area.lat_max) / 2
    center_lng = (settings.area.lng_min + settings.area.lng_max) / 2

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=14,
        tiles='OpenStreetMap'
    )

    # Prepare heatmap data
    heat_data = []
    for idx, row in df.iterrows():
        lat, lng = coords[idx]
        value = row[category_name]
        if pd.notna(value) and value > 0:
            # Weight by value (repeat coordinates proportional to count)
            heat_data.append([lat, lng, float(value)])

    # Add heatmap layer
    plugins.HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=18,
        radius=25,
        blur=30,
        gradient={
            0.0: 'blue',
            0.3: 'cyan',
            0.5: 'lime',
            0.7: 'yellow',
            0.9: 'orange',
            1.0: 'red'
        }
    ).add_to(m)

    # Add title
    title_html = f'''
    <div style="position: fixed;
                top: 10px; left: 50%; transform: translateX(-50%);
                width: auto; background-color: white; z-index:9999;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                font-family: 맑은 고딕;">
        <h3 style="margin: 0;">{category_name} 분포 히트맵 (망원동 72개 집계구역)</h3>
        <p style="margin: 5px 0 0 0; font-size: 13px;">총합: {int(df[category_name].sum())}명</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def main():
    settings = get_settings()

    # Load census data
    csv_path = settings.paths.data_dir / "area_summary.csv"
    df = pd.read_csv(csv_path, encoding='cp949')

    print(f"Loaded {len(df)} census areas")

    # Calculate coordinates (same as before - grid layout)
    import numpy as np
    n_areas = len(df)
    grid_size = int(np.ceil(np.sqrt(n_areas)))

    lat_min, lat_max = settings.area.lat_min, settings.area.lat_max
    lng_min, lng_max = settings.area.lng_min, settings.area.lng_max

    lat_step = (lat_max - lat_min) / grid_size
    lng_step = (lng_max - lng_min) / grid_size

    coords = []
    for i in range(n_areas):
        row = i // grid_size
        col = i % grid_size
        lat = lat_min + (row + 0.5) * lat_step
        lng = lng_min + (col + 0.5) * lng_step
        coords.append((lat, lng))

    # Calculate demographic categories
    df['총인구'] = df['선택범위_합계']
    df['남성인구'] = df[[col for col in df.columns if '_남자' in col and '100세이상' not in col]].sum(axis=1)
    df['여성인구'] = df[[col for col in df.columns if '_여자' in col]].sum(axis=1)

    # Age groups
    df['10대이하'] = (
        df['4세이하'] + df['5세이상~9세이하'] + df['10세이상~14세이하'] + df['15세이상~19세이하']
    )
    df['20대'] = df['20세이상~24세이하'] + df['25세이상~29세이하']
    df['30대'] = df['30세이상~34세이하'] + df['35세이상~39세이하']
    df['40대'] = df['40세이상~44세이하'] + df['45세이상~49세이하']
    df['50대'] = df['50세이상~54세이하'] + df['55세이상~59세이하']
    df['60대이상'] = df[[col for col in df.columns if '60세이상' in col or '65세이상' in col or
                          '70세이상' in col or '75세이상' in col or '80세이상' in col or
                          '85세이상' in col or '90세이상' in col or '95세이상' in col or
                          '100세이상' in col]].sum(axis=1)

    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create heatmaps for different categories
    categories = [
        ('총인구', 'census_heatmap_total.html'),
        ('남성인구', 'census_heatmap_male.html'),
        ('여성인구', 'census_heatmap_female.html'),
        ('10대이하', 'census_heatmap_age_teens.html'),
        ('20대', 'census_heatmap_age_20s.html'),
        ('30대', 'census_heatmap_age_30s.html'),
        ('40대', 'census_heatmap_age_40s.html'),
        ('50대', 'census_heatmap_age_50s.html'),
        ('60대이상', 'census_heatmap_age_60plus.html'),
        ('1인가구', 'census_heatmap_single_household.html'),
    ]

    print("\nGenerating heatmaps...")
    for category, filename in categories:
        m = create_heatmap_by_category(df, coords, category)
        output_path = output_dir / filename
        m.save(str(output_path))
        print(f"  ✓ {category}: {filename}")

    # Create combined view with layer control
    print("\nCreating combined layer view...")
    center_lat = (lat_min + lat_max) / 2
    center_lng = (lng_min + lng_max) / 2

    m_combined = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=14,
        tiles='OpenStreetMap'
    )

    # Add all heatmaps as layers
    for category, _ in categories:
        heat_data = []
        for idx, row in df.iterrows():
            lat, lng = coords[idx]
            value = row[category]
            if pd.notna(value) and value > 0:
                heat_data.append([lat, lng, float(value)])

        heat_layer = plugins.HeatMap(
            heat_data,
            name=category,
            min_opacity=0.3,
            max_zoom=18,
            radius=25,
            blur=30,
            show=False if category != '총인구' else True
        )
        m_combined.add_child(heat_layer)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m_combined)

    # Add title
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50%; transform: translateX(-50%);
                width: auto; background-color: white; z-index:9999;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                font-family: 맑은 고딕;">
        <h3 style="margin: 0;">망원동 인구 히트맵 (레이어 선택 가능)</h3>
        <p style="margin: 5px 0 0 0; font-size: 13px;">오른쪽 상단에서 보고 싶은 통계를 선택하세요</p>
    </div>
    '''
    m_combined.get_root().html.add_child(folium.Element(title_html))

    output_path = output_dir / "census_heatmap_combined.html"
    m_combined.save(str(output_path))
    print(f"  ✓ Combined view: census_heatmap_combined.html")

    print(f"\n{'='*60}")
    print("All heatmaps saved to: {0}".format(output_dir))
    print("="*60)
    print("\nRecommended file to open:")
    print(f"  👉 census_heatmap_combined.html (all layers in one map)")
    print("\nIndividual heatmaps:")
    for category, filename in categories:
        total = int(df[category].sum())
        print(f"  - {filename}: {total:,}명")


if __name__ == "__main__":
    main()
