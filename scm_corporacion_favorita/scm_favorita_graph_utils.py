import networkx as nx


def get_favorita_definitions():
    # This dictionary defines a hierarchical grouping of observed variables into
    # semantically meaningful latent constructs for the Favorita dataset.
    causal_groups = {
        "Holidays": ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "holiday_proximity", "holiday_weight"],
        "Date_Features": [
            "Year", "Quarter", "Year_Progress", "Month_Progress",
            "Is_Month_Start", "Is_Month_End",
            "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
            "WeekOfYear_sin", "WeekOfYear_cos"
        ],
        "Season": ["Season_sin", "Season_cos", "Is_Summer"],
        "Location": ["city", "state"],
        "Economy": ["dcoilwtico", "cpi", "unemployment"],
        "Weather": ["weather_condition", "temperature", "precipitation", "wind_speed", "humidity"],
        "Store_Metadata": ["store_nbr", "store_type", "cluster"],
        "Store_Activity": ["transactions"],
        "Product_Features": ["family", "class", "perishable", "onpromotion"],
        "External_Events": ["earthquake_impact"],
        "Sales": ["unit_sales"]
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
        ("Holidays", "Product_Features"),
        ("Holidays", "Sales"),
        ("Location", "Weather"),
        ("Location", "Store_Metadata"),
        ("Location", "Store_Activity"),
        ("Location", "Economy"),
        ("Location", "Sales"),
        ("Store_Metadata", "Product_Features"),
        ("Store_Metadata", "Store_Activity"),
        ("Product_Features", "Store_Activity"),
        ("Store_Activity", "Sales"),
        ("Product_Features", "Sales"),
        ("Economy", "Sales"),
        ("Weather", "Sales"),
        ("External_Events", "Economy"),
        ("External_Events", "Store_Activity"),
        ("External_Events", "Sales")
    ]

    macro_graph = nx.DiGraph()
    macro_graph.add_nodes_from(causal_groups.keys())
    macro_graph.add_edges_from(causal_edges_groups)

    return causal_groups, causal_edges_groups, macro_graph


def build_feature_graph(df_columns):
    causal_groups, causal_edges_groups, macro_graph = get_favorita_definitions()

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

    for holiday_var in causal_groups["Holidays"]:
        if feature_graph.has_edge(holiday_var, "family"):
            feature_graph.remove_edge(holiday_var, "family")
        if feature_graph.has_edge(holiday_var, "class"):
            feature_graph.remove_edge(holiday_var, "class")
        if feature_graph.has_edge(holiday_var, "perishable"):
            feature_graph.remove_edge(holiday_var, "perishable")

    if feature_graph.has_edge("family", "transactions"):
        feature_graph.remove_edge("family", "transactions")
    if feature_graph.has_edge("class", "transactions"):
        feature_graph.remove_edge("class", "transactions")
    if feature_graph.has_edge("perishable", "transactions"):
        feature_graph.remove_edge("perishable", "transactions")

    print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")
    print("\n--- FINAL FEATURE-LEVEL GRAPH NODES ---")
    for node in feature_graph.nodes():
        print(node)

    print("\n--- FINAL FEATURE-LEVEL GRAPH EDGES ---")
    for source, target in feature_graph.edges():
        print(f"{source} -> {target}")

    return feature_graph, causal_groups, macro_graph