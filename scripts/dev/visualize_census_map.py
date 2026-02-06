"""
Visualize census data (population by area, age, gender) on an interactive map.
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


def create_census_visualization():
    """Create interactive map with census statistics."""
    settings = get_settings()

    # Load census data
    csv_path = settings.paths.data_dir / "area_summary.csv"
    df = pd.read_csv(csv_path, encoding='cp949')

    print(f"Loaded {len(df)} census areas")

    # Calculate statistics per area
    df['총인구'] = df['선택범위_합계']
    df['남성인구'] = df[[col for col in df.columns if '_남자' in col and '100세이상' not in col]].sum(axis=1)
    df['여성인구'] = df[[col for col in df.columns if '_여자' in col]].sum(axis=1)
    df['남성비율'] = (df['남성인구'] / df['총인구'] * 100).round(1)
    df['여성비율'] = (df['여성인구'] / df['총인구'] * 100).round(1)

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

    # Household types
    df['1인가구비율'] = (df['1인가구'] / df['총인구'] * 100).round(1)

    # Calculate area center points (망원동 범위 내에 균등 분포 가정)
    # 실제로는 집계구역 경계 GeoJSON이 필요하지만, 여기서는 간단히 격자 배치
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

    df['위도'] = [c[0] for c in coords]
    df['경도'] = [c[1] for c in coords]

    # Create base map
    center_lat = (lat_min + lat_max) / 2
    center_lng = (lng_min + lng_max) / 2
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=14,
        tiles='OpenStreetMap'
    )

    # Add markers for each census area
    for idx, row in df.iterrows():
        # Determine color by population size
        population = row['총인구']
        if population > 600:
            color = 'red'
        elif population > 500:
            color = 'orange'
        elif population > 400:
            color = 'lightblue'
        else:
            color = 'lightgray'

        # Create detailed popup
        popup_html = f"""
        <div style="font-family: 맑은 고딕; width: 300px;">
            <h4 style="margin-bottom: 10px;">집계구역: {row['TOT_OA_CD']}</h4>

            <div style="margin-bottom: 15px;">
                <b style="font-size: 16px;">총 인구: {int(row['총인구'])}명</b>
            </div>

            <div style="margin-bottom: 10px;">
                <b>성별 분포:</b><br>
                남성: {int(row['남성인구'])}명 ({row['남성비율']}%)<br>
                여성: {int(row['여성인구'])}명 ({row['여성비율']}%)<br>
            </div>

            <div style="margin-bottom: 10px;">
                <b>연령대 분포:</b><br>
                10대 이하: {int(row['10대이하'])}명<br>
                20대: {int(row['20대'])}명<br>
                30대: {int(row['30대'])}명<br>
                40대: {int(row['40대'])}명<br>
                50대: {int(row['50대'])}명<br>
                60대 이상: {int(row['60대이상'])}명<br>
            </div>

            <div style="margin-bottom: 10px;">
                <b>주거 유형:</b><br>
                다세대: {int(row['다세대'])}가구<br>
                단독주택: {int(row['단독주택'])}가구<br>
                아파트: {int(row['아파트'])}가구<br>
                연립주택: {int(row['연립주택'])}가구<br>
            </div>

            <div>
                <b>가구 유형:</b><br>
                1인가구: {int(row['1인가구'])}명 ({row['1인가구비율']}%)<br>
                2세대가구: {int(row['2세대가구'])}가구<br>
                3세대가구: {int(row['3세대가구'])}가구<br>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row['위도'], row['경도']],
            radius=population / 30,  # Scale by population
            popup=folium.Popup(popup_html, max_width=350),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            tooltip=f"집계구역 {row['TOT_OA_CD']}: {int(population)}명"
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 200px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                font-family: 맑은 고딕;">
        <p style="margin: 0 0 5px 0;"><b>인구 규모</b></p>
        <p style="margin: 5px 0;"><span style="color: red;">●</span> 600명 초과</p>
        <p style="margin: 5px 0;"><span style="color: orange;">●</span> 500-600명</p>
        <p style="margin: 5px 0;"><span style="color: lightblue;">●</span> 400-500명</p>
        <p style="margin: 5px 0;"><span style="color: lightgray;">●</span> 400명 미만</p>
        <p style="margin: 10px 0 5px 0; font-size: 12px;">
            원의 크기는 인구 수에 비례
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add title
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50%; transform: translateX(-50%);
                width: auto; background-color: white; z-index:9999;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                font-family: 맑은 고딕;">
        <h3 style="margin: 0;">망원동 집계구역별 인구 통계 (72개 구역, 총 35,589명)</h3>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    output_path = project_root / "data" / "output" / "census_map.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    print(f"\nMap saved to: {output_path}")
    print(f"\nOpen the file in a web browser to view the interactive map.")

    # Print summary statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total areas: {len(df)}")
    print(f"Total population: {int(df['총인구'].sum())}")
    print(f"\nPopulation range:")
    print(f"  Min: {int(df['총인구'].min())} people")
    print(f"  Max: {int(df['총인구'].max())} people")
    print(f"  Mean: {int(df['총인구'].mean())} people")
    print(f"  Median: {int(df['총인구'].median())} people")

    print(f"\nGender distribution:")
    print(f"  Male: {int(df['남성인구'].sum())} ({df['남성인구'].sum() / df['총인구'].sum() * 100:.1f}%)")
    print(f"  Female: {int(df['여성인구'].sum())} ({df['여성인구'].sum() / df['총인구'].sum() * 100:.1f}%)")

    print(f"\nAge distribution:")
    total_pop = df['총인구'].sum()
    print(f"  10대 이하: {int(df['10대이하'].sum())} ({df['10대이하'].sum() / total_pop * 100:.1f}%)")
    print(f"  20대: {int(df['20대'].sum())} ({df['20대'].sum() / total_pop * 100:.1f}%)")
    print(f"  30대: {int(df['30대'].sum())} ({df['30대'].sum() / total_pop * 100:.1f}%)")
    print(f"  40대: {int(df['40대'].sum())} ({df['40대'].sum() / total_pop * 100:.1f}%)")
    print(f"  50대: {int(df['50대'].sum())} ({df['50대'].sum() / total_pop * 100:.1f}%)")
    print(f"  60대 이상: {int(df['60대이상'].sum())} ({df['60대이상'].sum() / total_pop * 100:.1f}%)")

    print(f"\n1-person households: {int(df['1인가구'].sum())} ({df['1인가구'].sum() / total_pop * 100:.1f}%)")


if __name__ == "__main__":
    create_census_visualization()
