import networkx as nx
def get_walmart_definitions():
    # This dictionary defines a hierarchical grouping of observed variables into
    # semantically meaningful latent constructs for the Walmart dataset.
    causal_groups = {
        "Holidays": ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "holiday_proximity", "holiday_weight"],
        "Date_Features": [
            "Year", "Quarter",
            "Year_Progress", "Month_Progress",
            "Is_Month_Start", "Is_Month_End",
            "Month_sin", "Month_cos",
            "DayOfWeek_sin", "DayOfWeek_cos",
            "WeekOfYear_sin", "WeekOfYear_cos", ],
        "Season": ["Season_sin", "Season_cos","Is_Summer"],
        "City": ["city"],
        "Economy": ["Fuel_Price", "CPI", "Unemployment"],
        "Weather": [
            "weather_condition", "Temperature",
            "precipitation", "wind_speed", "humidity", "temperature_min", "temperature_max"
        ],
        "Store_Features": ["Store", "Type", "Size", "Dept"],
        "MarkDowns": ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"],
        "Sales": ["Weekly_Sales"]
    }

    # Specification of causal structure between macro-level variables

    # These directed edges encode domain-informed assumptions about the data-
    # generating process. They define a Directed Acyclic Graph (DAG) over latent groups.
    causal_edges_groups = [
        ("Date_Features", "Season"),
        ("Date_Features", "Holidays"),
        ("Date_Features", "Sales"),
        ("Date_Features", "Economy"),
        ("Season", "Weather"),
        ("Season", "Sales"),
        ("Holidays", "MarkDowns"),
        ("Holidays", "Sales"),
        ("MarkDowns", "Sales"),
        ("City", "Weather"),
        ("City", "Store_Features"),
        ("City", "Economy"),
        ("City", "Sales"),
        ("Store_Features", "Sales"),
        ("Store_Features", "MarkDowns"),
        ("Weather", "Sales"),
        ("Economy", "Sales")
    ]

    macro_graph = nx.DiGraph()
    macro_graph.add_nodes_from(causal_groups.keys())
    macro_graph.add_edges_from(causal_edges_groups)

    return causal_groups, causal_edges_groups, macro_graph


def build_feature_graph(df_columns):
    causal_groups, causal_edges_groups, macro_graph = get_walmart_definitions()

    feature_graph = nx.DiGraph()

    for group_name, columns in causal_groups.items():
        for col in columns:
            if col in df_columns:
                feature_graph.add_node(col)

    for source_group, target_group in causal_edges_groups:
        for src_col in causal_groups[source_group]:
            for tgt_col in causal_groups[target_group]:
                if src_col in df_columns and tgt_col in df_columns:
                    feature_graph.add_edge(src_col, tgt_col)

    return feature_graph, causal_groups, macro_graph