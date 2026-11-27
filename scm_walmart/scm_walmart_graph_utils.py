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
            "WeekOfYear_sin", "WeekOfYear_cos",
        ],
        "Season": ["Season_sin", "Season_cos", "Is_Summer"],
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

    invalid_edges = []

    non_seasonal_dates = ["Year", "Month_Progress", "Is_Month_Start", "Is_Month_End", "DayOfWeek_sin", "DayOfWeek_cos"]
    for src in non_seasonal_dates:
        for tgt in causal_groups["Season"]:
            invalid_edges.append((src, tgt))

    non_holiday_dates = ["Year", "Month_Progress", "Is_Month_Start", "Is_Month_End"]
    for src in non_holiday_dates:
        for tgt in causal_groups["Holidays"]:
            invalid_edges.append((src, tgt))
    invalid_edges.extend([
        ("DayOfWeek_sin", "Is_Christmas_Season"),
        ("DayOfWeek_cos", "Is_Christmas_Season"),
    ])

    micro_dates = ["DayOfWeek_sin", "DayOfWeek_cos", "Is_Month_Start", "Is_Month_End", "Month_Progress"]
    for src in micro_dates:
        for tgt in causal_groups["Economy"]:
            invalid_edges.append((src, tgt))

    for md in causal_groups["MarkDowns"]:
        invalid_edges.append(("Dept", md))
    invalid_edges.append(("city", "Dept"))

    for src, tgt in invalid_edges:
        if feature_graph.has_edge(src, tgt):
            feature_graph.remove_edge(src, tgt)

    hierarchical_edges = [
        ("Store", "Type"),
        ("Store", "Size"),
        ("Type", "Size"),
        ("Store", "Dept"),
        ("Type", "Dept"),
        ("Size", "Dept"),
        ("Temperature", "temperature_min"),
        ("Temperature", "temperature_max"),
    ]
    for src, tgt in hierarchical_edges:
        if src in df_columns and tgt in df_columns:
            feature_graph.add_edge(src, tgt)

    assert nx.is_directed_acyclic_graph(feature_graph), "Graph contains a cycle!"

    print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")
    print("\n--- FINAL FEATURE-LEVEL GRAPH NODES ---")
    for node in feature_graph.nodes():
        print(node)

    print("\n--- FINAL FEATURE-LEVEL GRAPH EDGES ---")
    for source, target in feature_graph.edges():
        print(f"{source} -> {target}")

    return feature_graph, causal_groups, macro_graph