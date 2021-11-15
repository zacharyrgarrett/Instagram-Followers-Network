# Zachary Garrett
# Social Media: Instagram
# Network Type: Friendship Network
# User Source: zach_garrett3

import pandas as pd
import http.client
import json
import time
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab


# Dumps the object to a json string and into a file
def save_object_to_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)
        print(f"Saved to: '{filename}'")


# Saves the following information for each of the followers of the username provided
def save_followings_from_followers(username):
    f = open(f"{username}_followers.json")
    data = json.load(f)

    followings = dict()
    for row in data:
        if "collector" in row:
            users = row["collector"]
            for user in users:
                if "username" in user:
                    uname = user["username"]
                    print(f"\nGetting {uname} followings...")
                    followings[uname] = save_followX_information_from_api("followings", uname, 1250, False)
    save_object_to_json(followings, f"{username}_followings.json")


# Saves the follow(ings/ers) information from the Instagram API for the given username
def save_followX_information_from_api(req_type, username, count_limit=math.inf, save_file=True):
    data = []
    req_response = json.loads(request(req_type, username))
    print("Retrieved: " + str(req_response))
    data.append(req_response)

    count = 50
    while "has_more" in req_response and req_response[
        "has_more"] and "end_cursor" in req_response and count < count_limit:
        req_response = json.loads(request(req_type, username, req_response["end_cursor"]))
        print("Retrieved: " + str(req_response))
        data.append(req_response)
        count += 50

    if save_file:
        fname = f"{username}_{req_type}.json"
        save_object_to_json(data, fname)
        return
    else:
        return data


# Send API request
# Returns JSON string
def request(req_type, username, end_cursor='', retry_attempt_no=0):
    conn = http.client.HTTPSConnection("instagram-data1.p.rapidapi.com", timeout=10)
    headers = {
        'x-rapidapi-host': "instagram-data1.p.rapidapi.com",
        'x-rapidapi-key': "API-KEY"
    }
    end_cursor = "&end_cursor=" + str(end_cursor) if end_cursor != '' else ''

    try:
        if retry_attempt_no > 3:
            return '{}'
        conn.request("GET", f"/{req_type}?username={username}" + end_cursor, headers=headers)
        res = conn.getresponse()
        data = res.read()
        decoded_information = data.decode("utf-8")
    except Exception:
        print("\nConnection timed out. Waiting for 5 seconds then trying again...")
        time.sleep(5)
        decoded_information = request(req_type, username, end_cursor, retry_attempt_no + 1)

    return decoded_information


# Returns list of public followers' usernames
def get_public_followers(uname):
    # Open data file
    f = open(f"{uname}_followers.json")
    data = json.load(f)

    # Make list of usernames that are public
    public_followers = []
    for query_segment in data:
        if "collector" in query_segment:
            for follower in query_segment["collector"]:
                if "is_private" in follower and not follower["is_private"]:
                    public_followers.append(follower["username"])

    return public_followers


# Returns list of edges for a single follower
def get_single_follower_edges(follower_uname, all_followers, followings):
    edges = []

    for following_segment in followings:
        valid_indices = "collector" in following_segment and len(following_segment["collector"]) > 0
        if valid_indices:
            for user in following_segment["collector"]:
                if "username" in user and user["username"] in all_followers:
                    edges.append((follower_uname, user["username"]))

    return edges


# Constructs a list of edges
def get_all_edges(uname):
    followers = get_public_followers(uname)

    # Open data file
    f = open(f"{uname}_followings.json")
    data = json.load(f)

    edge_list = []
    for follower_uname in followers:
        edge_list += get_single_follower_edges(follower_uname, followers, data[follower_uname])
    return edge_list


def save_graph(graph, file_name):
    # Initialize Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    plt.title("Instagram Friendship Network - zach_garrett3", fontdict=dict(size=30))
    fig = plt.figure(1)
    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    del fig

    print(f"\nSaved the Friendship Network to '{file_name}'")


# Returns fully built friendship network
def build_friendship_network(uname):
    # Get edges
    edge_list = get_all_edges(uname)

    # Make network
    network = nx.DiGraph()
    network.add_edges_from(edge_list)

    return network


# Plots Degree Distribution
def plot_degree_distribution(network, file_name):
    # Get degree frequencies
    in_deg_freq = degree_histogram_directed(network, "in")
    out_deg_freq = degree_histogram_directed(network, "out")

    # Build figure
    plt.figure(figsize=(12, 8))
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.loglog(range(len(in_deg_freq)), in_deg_freq, 'ro-', label='in-degree')
    plt.loglog(range(len(out_deg_freq)), out_deg_freq, 'bo-', label='out-degree')
    plt.legend(loc="upper right")
    plt.title('Instagram Friendship Network\nDegree Distribution - zach_garrett3')
    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()

    print(f"\nSaved the Degree Distribution to '{file_name}'")


def degree_histogram_directed(graph, direction):
    # Get nodes
    nodes = graph.nodes()

    # Get degree sequence values
    if direction == "in":
        in_degree = dict(graph.in_degree())
        sequence = [in_degree.get(k, 0) for k in nodes]
    elif direction == "out":
        out_degree = dict(graph.out_degree())
        sequence = [out_degree.get(k, 0) for k in nodes]
    else:
        sequence = [v for k, v in graph.degree()]

    degree_max = max(sequence) + 1
    frequency = [0 for deg in range(degree_max)]
    for deg in sequence:
        frequency[deg] += 1
    return frequency


# Compute Clustering Coefficient
def compute_clustering_coefficient(graph, file_name):
    # Convert to undirected
    undirected_graph = graph.to_undirected()
    all_cluster = nx.clustering(undirected_graph)

    fig, ax = plt.subplots()

    # Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    # Generate table
    data = {"Node": all_cluster.keys(), "Clustering Coefficient": all_cluster.values()}
    df = pd.DataFrame(data)
    df['Rank'] = df['Clustering Coefficient'].rank(ascending=False)
    print(f"\nAverage Clustering Coefficient: {df['Clustering Coefficient'].mean()}")

    # Save Table
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    print(f"Saved the Clustering Coefficient to '{file_name}'")


# Compute Diameter
def compute_betweenness(graph, file_name):

    # Compute betweenness for each node
    betweenness = nx.betweenness_centrality(graph)

    fig, ax = plt.subplots()

    # Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    # Generate table
    data = {"Node": betweenness.keys(), "Betweenness Centrality": betweenness.values()}
    df = pd.DataFrame(data)
    df['Rank'] = df['Betweenness Centrality'].rank(ascending=False)
    print(f"\nAverage Betweenness Centrality: {df['Betweenness Centrality'].mean()}")

    # Save Table
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.savefig(file_name, bbox_inches="tight")
    pylab.close()
    print(f"Saved the Betweenness Centrality to '{file_name}'")


if __name__ == "__main__":
    # Build Network
    friendship_network = build_friendship_network("zach_garrett3")

    # Save to file
    save_graph(friendship_network, "zach_garrett3-friendship_network.png")

    # Compute
    print("\n*******************************")
    print("            RESULTS            ")
    print("*******************************")
    plot_degree_distribution(friendship_network, "zach_garrett3-degree_distribution.png")
    compute_clustering_coefficient(friendship_network, "zach_garrett3-clustering_coefficient.png")
    compute_betweenness(friendship_network, "zach_garrett3-betweenness_centrality.png")

    # Saves to Gephi
    # nx.write_gexf(friendship_network, "zach_garrett3-friendship_network.gexf")

    exit()
