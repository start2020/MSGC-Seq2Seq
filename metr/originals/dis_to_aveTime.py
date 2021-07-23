import numpy as np
import pandas as pd
import pickle


def create_distance_matrix():
    distance_path = 'distance.csv'
    # ids_path = os.path.join(data_path, dataset_name, 'graph_sensor_ids.txt')
    # nodes and indexs
    # with open(ids_path) as f:
    #     ids = f.read().strip().split(',')
    #     # print(ids)
    num_ids = 307
    id_to_index = {}
    index_to_id = {}
    ids = []
    for i in range(num_ids):
        ID = str(i)
        ids.append(ID)
        id_to_index[ID] = i
        index_to_id[i] = ID

    # create matrix
    dist_matrix = np.zeros((num_ids, num_ids), dtype=np.float32)
    # dist_matrix[:] = np.inf
    distance = pd.read_csv(distance_path, dtype={'from': 'str', 'to': 'str', 'cost': float})
    for row in distance.values:
        if row[0] not in ids or row[1] not in ids:
            continue
        dist_matrix[id_to_index[row[0]], id_to_index[row[1]]] = row[2] / 70000.0 * 60.0
    adj = dist_matrix


    print(adj.shape)
    # # save
    # path1 = os.path.join(data_path, dataset_name, 'id_to_index.json')
    # path2 = os.path.join(data_path, dataset_name, 'index_to_id.json')
    path3 = 'connection.xlsx'
    # print(path3)
    # with open(path1, 'w') as f:
    #     json.dump(id_to_index, f)
    # with open(path2, 'w') as f:
    #     json.dump(index_to_id, f)
    df = pd.DataFrame(adj, index=ids, columns=ids)
    df.to_excel(path3)

    adjacent = df.replace(0, np.inf).values
    print(adjacent)
    def dijkstra(adjacent, j, MAX):
        N = adjacent.shape[0]
        visit = [False for _ in range(N)]  # 标记已确定好最短花销的点
        cost = adjacent[j]  # 最短花销记录
        cost[j] = 0
        done = []  # 已经确定好最短花销的点列表
        while len(done) < N:
            # 从cost里面找最短花销(找还没确定的点的路径)，标记这个最短边的顶点，把顶点加入t中
            minCost = MAX
            for i in range(N):
                if visit[i] == False and cost[i] < minCost:
                    minCost = cost[i]
                    minNode = i
            done.append(minNode)
            visit[minNode] = True
            # 从这个顶点出发，遍历与它相邻的顶点的边，计算最短路径，更新cost
            for w in range(N):
                if visit[w] == False and adjacent[minNode][w] != np.inf:
                    if (adjacent[minNode][w] + cost[minNode]) < cost[w]:
                        cost[w] = adjacent[minNode][w] + cost[minNode]  # 更新权值
        return cost

    N = adjacent.shape[0]
    MAX = np.inf
    short = []
    for j in range(N):
        cost = dijkstra(adjacent, j, MAX)
        short.append(cost)
    short = np.array(short)

    df1 = pd.DataFrame(short, columns=df.columns, index=df.index)
    file_name = "ave_time.xlsx"
    path = file_name
    df1.to_excel(path)

    n = adj.shape[0]
    with open("Adj.txt", 'w') as f:
        for i in range(n):
            for j in range(n):
                if i == j: adj[i, j] = 1 #强制赋值为1,不保存为邻接矩阵
                line = ""
                line += str(i)
                line += " "
                line += str(j)
                line += " "
                # line += str(np.exp(-ave_time[i,j]))
                line += str(adj[i, j])
                f.write(line + "\n")


def excel_to_pkl():
    df = pd.read_excel("connection.xlsx",index_col=0)
    connection_s = df.values
    connection = connection_s
    connection = connection.astype(np.float32)
    print(connection)
    f = open('adj_mx_pems04.pkl', 'wb+')
    sensor_ids, sensor_id_to_ind = 1, 2
    pickle.dump([sensor_ids, sensor_id_to_ind, connection], f)
    f.close()

create_distance_matrix()
excel_to_pkl()
