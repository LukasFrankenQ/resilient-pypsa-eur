import pandas as pd

data_2024_residential = {
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ],
    "Gas_Demand_Bcm": [
        26.0, 18.0, 17.0, 13.0, 7.0, 5.5,
        4.0, 4.0, 6.0, 11.0, 17.5, 23.0
    ]
}

df_2024_residential = pd.DataFrame(data_2024_residential)
print(df_2024_residential)
