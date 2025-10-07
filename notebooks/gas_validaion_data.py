import pandas as pd

# EU-27 plus UK gas demand in residential and commercial sector in 2024 (bcm)
# Figure 22 in https://www.oxfordenergy.org/wpcms/wp-content/uploads/2025/07/OIES-Quarterly-Gas-Review-Issue-29.pdf
ds_2024_residential = pd.Series(
    [26.0, 18.0, 17.0, 13.0, 7.0, 5.5, 4.0, 4.0, 6.0, 11.0, 17.5, 23.0],
    index=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    name="residential commercial"
)

# EU-27 plus UK gas demand in industrial sector in 2024 (bcm)
# Figure 21 in https://www.oxfordenergy.org/wpcms/wp-content/uploads/2025/07/OIES-Quarterly-Gas-Review-Issue-29.pdf
ds_2024_industrial = pd.Series(
    [8.3, 7.7, 7.7, 6.8, 6.5, 6., 6., 5.2, 6.2, 7., 7.9, 7.9],
    index=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    name="industry"
)


# EU-27 + UK gas production (bcm)
# Figure 11 in https://www.oxfordenergy.org/wpcms/wp-content/uploads/2025/07/OIES-Quarterly-Gas-Review-Issue-29.pdf
data_2024_production = {
    "Quarter": [
        "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2024", 
        "Q2-2024", "Q3-2024", "Q4-2024", "Q1-2025", "Q2-2025"
    ],
    "UK": [8.0, 7.3, 7.7, 7.4, 6.5, 6.1, 7.1, 6.9, 6.6],
    "Netherlands": [2.6, 2.4, 2.6, 2.7, 2.2, 2.3, 2.4, 2.4, 2.7],
    "Romania": [2.1, 2.1, 2.2, 2.2, 2.1, 2.1, 2.2, 2.2, 2.1],
    "Rest_of_EU": [3.2, 3.1, 3.3, 3.2, 3.2, 3.3, 3.5, 3.4, 3.5],
    "Total": [16.6, 15.6, 16.4, 16.2, 14.7, 14.4, 15.9, 15.7, 15.7],
}

df_2024_production = pd.DataFrame(data_2024_production).set_index("Quarter")
df_2024_production["Italy"] = (
    df_2024_production["Total"]
    - df_2024_production["UK"]
    - df_2024_production["Netherlands"]
    - df_2024_production["Romania"]
    - df_2024_production["Rest_of_EU"]
)

# European imports of gas (bcm)
# Figure 12 in https://www.oxfordenergy.org/wpcms/wp-content/uploads/2025/07/OIES-Quarterly-Gas-Review-Issue-29.pdf
data_pipeline_imports = {
    "Quarter": [
        "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2024",
        "Q2-2024", "Q3-2024", "Q4-2024", "Q1-2025", "Q2-2025"
    ],
    "Russia": [5.7, 7.4, 8.0, 7.6, 7.7, 8.3, 8.4, 4.4, 3.7],
    "Norway": [26.0, 24.7, 31.9, 32.3, 29.7, 28.1, 31.7, 31.0, 28.5],
    "North_Africa": [8.9, 8.9, 8.6, 7.6, 8.8, 7.0, 8.8, 8.3, 8.0],
    "Azerbaijan": [3.0, 3.0, 3.2, 3.1, 3.1, 2.8, 3.3, 2.7, 2.9],
    "OtherPipe": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.4],
    "Total_Pipeline": [43.6, 44.3, 51.9, 50.9, 49.7, 46.8, 52.6, 46.9, 43.5]
}

df_pipeline_imports = pd.DataFrame(data_pipeline_imports).set_index("Quarter")

# EU-27 + UK gas supply by source (bcm)
# Figure 12 in https://www.oxfordenergy.org/wpcms/wp-content/uploads/2025/07/OIES-Quarterly-Gas-Review-Issue-29.pdf
# ( sendout refers to LNG-regasified gas sent into Europe's gas network)
data_total_supply = {
    "Quarter": [
        "Q2-2023", "Q3-2023", "Q4-2023", "Q1-2024",
        "Q2-2024", "Q3-2024", "Q4-2024", "Q1-2025", "Q2-2025"
    ],
    "Production": [16.6, 15.6, 16.4, 16.2, 14.7, 14.4, 15.9, 15.7, 15.7],
    "Pipeline_Import": [43.6, 44.3, 51.9, 50.9, 49.7, 46.8, 52.6, 46.9, 43.5],
    "LNG_Sendout": [40.0, 29.9, 37.0, 34.0, 28.1, 23.2, 31.0, 39.1, 37.5],
    "Net_Storage_Withdrawal": [-22.8, -20.4, 9.8, 29.3, -19.4, -18.9, 23.7, 41.3, -26.4],
    "Re_exports": [-0.4, -2.8, -1.0, -0.3, -0.3, -0.3, -0.9, -0.6, -1.5],
    "Total_Supply": [76.9, 66.8, 114.5, 130.1, 72.4, 64.6, 122.2, 141.9, 68.8]
}

df_total_supply = pd.DataFrame(data_total_supply).set_index("Quarter")

# EU-27 gas stocks
# Figure 14 in https://www.oxfordenergy.org/wpcms/wp-content/uploads/2025/07/OIES-Quarterly-Gas-Review-Issue-29.pdf
data_eu_storage_2024 = {
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    "Storage_Bcm": [
        85, 80, 75, 60, 63, 70, 80, 90, 98, 100, 100, 90
    ]
}

df_storage_2024 = pd.DataFrame(data_eu_storage_2024).set_index("Month")
