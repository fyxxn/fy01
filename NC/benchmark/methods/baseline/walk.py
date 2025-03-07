import numpy as np
import scipy.sparse as sp
import random
import sys
sys.path.append('../../')
sys.path.append("/home/zlj/code/HGB_autobenchmark/NC/benchmark/scripts")
from utils.data_emb import load_data

def row_normalize(mx):
    # rowsum是一个list，list[0]是mx第一行的和
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    # mx就是对每一行进行一个正则化
    return mx


class DataGeneratorDBLP(object):
    def __init__(self, walk_n, walk_L, args):
        # adj = sp.load_npz(path + '/adjM.npz').toarray()
        # type_mask = np.load(path + '/node_types.npy')


        features_list, adj, labels, train_val_test_idx, dl = load_data(args.dataset)
        adj=adj.toarray()
        adj=np.triu(adj)
        adj += adj.T - np.diag(adj.diagonal())
        
        # print(adj.shape)
        print(features_list[0].shape)
        print(features_list[1].shape)
        print(features_list[2].shape)
        print(features_list[3].shape)
        type_mask=np.zeros(adj.shape[0])
        type_mask[features_list[0].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]]=1
        type_mask[features_list[0].shape[0]+features_list[1].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]]=2
        type_mask[features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]:]=3


        mask_a = np.where(type_mask == 0)[0]
        mask_p = np.where(type_mask == 1)[0]
        mask_t = np.where(type_mask == 2)[0]
        mask_v = np.where(type_mask == 3)[0]
        self.A_n = mask_a.shape[0]
        self.P_n = mask_p.shape[0]
        self.T_n = mask_t.shape[0]
        self.V_n = mask_v.shape[0]
        self.walk_n = walk_n
        self.walk_L = walk_L

        paper_author_list = [[] for _ in range(self.P_n)]
        author_paper_list = [[] for _ in range(self.A_n)]
        paper_term_list = [[] for _ in range(self.P_n)]
        term_paper_list = [[] for _ in range(self.T_n)]
        paper_venue_list = [[] for _ in range(self.P_n)]
        venue_paper_list = [[] for _ in range(self.V_n)]

        relation_list = [paper_author_list, author_paper_list, paper_term_list, term_paper_list,
                         paper_venue_list, venue_paper_list]
        header = ['aAuthor#', 'pPaper#', 'tTerm#', 'pPaper#', 'vVenue#', 'pPaper#']

        relations = [adj[mask_p, :][:, mask_a], adj[mask_a, :][:, mask_p], adj[mask_p, :][:, mask_t],
                     adj[mask_t, :][:, mask_p], adj[mask_p, :][:, mask_v], adj[mask_v, :][:, mask_p]]
        for k, relation in enumerate(relations):
            for i in range(relation.shape[0]):
                for j in range(relation.shape[1]):
                    if relation[i, j] != 0:
                        relation_list[k][i].append(header[k] + str(j))

        # paper neighbor: author + citation + venue + term
        paper_neigh_list = [[] for _ in range(self.P_n)]
        for i in range(self.P_n):
            paper_neigh_list[i] += paper_author_list[i]
            paper_neigh_list[i] += paper_venue_list[i]
            paper_neigh_list[i] += paper_term_list[i]

        self.author_paper_list = author_paper_list
        self.paper_author_list = paper_author_list
        self.paper_term_list = paper_term_list
        self.paper_venue_list = paper_venue_list
        self.paper_neigh_list = paper_neigh_list
        self.venue_paper_list = venue_paper_list
        self.term_paper_list = term_paper_list

        # features = sp.load_npz(path + '/features_1.npz').toarray().astype(np.float)
        # self.paper_word = row_normalize(features)
        # self.word_paper = row_normalize(features.T)

    def het_walk_APVPA(self):
        het_walk_f = open("walks_DBLP.txt", "w")
        for i in range(self.walk_n):
            print('walk_APVPA: {}...'.format(i))
            for j in range(self.A_n):
                if len(self.author_paper_list[j]):
                    line = ''
                    # define header for this walk
                    cur_node = "aAuthor#" + str(j)
                    for l in range(self.walk_L - 1):
                        # save A
                        line += cur_node + " "
                        # walk and save P
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.author_paper_list[cur_node])
                        line += cur_node + " "
                        # walk and save V
                        cur_node = int(cur_node.split('#')[1])
                        if len(self.paper_venue_list[cur_node]) == 0:
                            cur_node = "aAuthor#" + str(j)
                            continue
                        cur_node = random.choice(
                            self.paper_venue_list[cur_node])
                        line += cur_node + " "
                        # walk and save P
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.venue_paper_list[cur_node])
                        line += cur_node + " "
                        # walk A
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.paper_author_list[cur_node])
                    line = line[:len(line)-1] + '\n'
                    het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_APTPA(self):
        het_walk_f = open("walks_DBLP.txt", "a")
        for i in range(self.walk_n):
            print('walk_APTPA: {}...'.format(i))
            for j in range(self.A_n):
                if len(self.author_paper_list[j]):
                    line = ''
                    # define header for this walk
                    cur_node = "aAuthor#" + str(j)
                    for l in range(self.walk_L - 1):
                        # save A
                        line += cur_node + " "
                        # walk and save P
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.author_paper_list[cur_node])
                        line += cur_node + " "
                        # walk and save T
                        cur_node = int(cur_node.split('#')[1])
                        if len(self.paper_term_list[cur_node]) == 0:
                            cur_node = "aAuthor#" + str(j)
                            continue
                        cur_node = random.choice(
                            self.paper_term_list[cur_node])
                        line += cur_node + " "
                        # walk and save P
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.term_paper_list[cur_node])
                        line += cur_node + " "
                        # walk A
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.paper_author_list[cur_node])
                    line = line[:len(line)-1] + '\n'
                    het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_APA(self):
        het_walk_f = open("walks_DBLP.txt", "a")
        for i in range(self.walk_n):
            print('walk_APA: {}...'.format(i))
            for j in range(self.A_n):
                if len(self.author_paper_list[j]):
                    line = ''
                    # define header for this walk
                    cur_node = "aAuthor#" + str(j)
                    for l in range(self.walk_L - 1):
                        # save A
                        line += cur_node + " "
                        # walk and save P
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.author_paper_list[cur_node])
                        line += cur_node + " "
                        # walk A
                        cur_node = int(cur_node.split('#')[1])
                        cur_node = random.choice(
                            self.paper_author_list[cur_node])
                    line = line[:len(line)-1] + '\n'
                    het_walk_f.write(line)
        het_walk_f.close()

    
        

class DataGeneratorACM(object):
    def __init__(self, walk_n, walk_L, args):
        # 0 for papers, 1 for authors, 2 for subjects(4019、7167、60)
        features_list, adj, labels, train_val_test_idx, dl = load_data(args.dataset)
        adj=adj.toarray()
        adj=np.triu(adj)
        adj += adj.T - np.diag(adj.diagonal())
        
        # print(adj.shape)
        print(features_list[0].shape)
        print(features_list[1].shape)
        print(features_list[2].shape)
        print(features_list[3].shape)
        type_mask=np.zeros(adj.shape[0])
        type_mask[features_list[0].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]]=1
        type_mask[features_list[0].shape[0]+features_list[1].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]]=2
        type_mask[features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]:]=3
        # adj = sp.load_npz(path + '/adjM.npz').toarray()
        # type_mask = np.load(path + '/node_types.npy')
        # mask_p是类型为0的下标list
        mask_p = np.where(type_mask == 0)[0]
        mask_a = np.where(type_mask == 1)[0]
        mask_s = np.where(type_mask == 2)[0]
        mask_t = np.where(type_mask == 3)[0]

        
        self.save_path = 'walks_ACM.txt'

        
        

        self.P_n = mask_p.shape[0]
        self.A_n = mask_a.shape[0]
        self.S_n = mask_s.shape[0]
        self.T_n = mask_t.shape[0]
        print(self.P_n)
        print(self.A_n)
        print(self.S_n)
        print(self.T_n)
        self.walk_n = walk_n
        self.walk_L = walk_L

        # paper_author_list是一个二维数组，长度为len(paper),第一个值是一个list,长度为p-a路径中a的数量
        paper_author_list = [[] for _ in range(self.P_n)]
        author_paper_list = [[] for _ in range(self.A_n)]
        paper_subject_list = [[] for _ in range(self.P_n)]
        subject_paper_list = [[] for _ in range(self.S_n)]
        paper_term_list = [[] for _ in range(self.P_n)]
        term_paper_list = [[] for _ in range(self.T_n)]

        relation_list = [paper_author_list, author_paper_list, paper_subject_list, subject_paper_list,paper_term_list,term_paper_list]
        header = ['aAuthor#', 'pPaper#', 'sSubject#', 'pPaper#','tTerm#','pPaper#']

        relations = [adj[mask_p, :][:, mask_a], adj[mask_a, :][:, mask_p], adj[mask_p, :][:, mask_s],
                     adj[mask_s, :][:, mask_p], adj[mask_p, :][:, mask_t], adj[mask_t, :][:, mask_p]]
        for k, relation in enumerate(relations):
            for i in range(relation.shape[0]):
                for j in range(relation.shape[1]):
                    if relation[i, j] != 0:
                        relation_list[k][i].append(header[k] + str(j))

        self.author_paper_list = author_paper_list
        self.paper_author_list = paper_author_list
        self.paper_subject_list = paper_subject_list
        self.subject_paper_list = subject_paper_list
        self.paper_term_list = paper_term_list
        self.term_paper_list = term_paper_list

        print(paper_author_list[0])

    def het_walk_PAP(self):
        het_walk_f = open(self.save_path, "w")
        for i in range(self.walk_n):
            for j in range(self.P_n):
                line = ''
                # define header for this walk
                cur_node = "pPaper#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveP
                    line += cur_node + " "
                    # walk and save A
                    cur_node = int(cur_node.split('#')[1])
                    if(len(self.paper_author_list[cur_node])==0):
                        break
                    else:
                        cur_node = random.choice(
                            self.paper_author_list[cur_node])
                    line += cur_node + " "
                    # walk P
                    cur_node = int(cur_node.split('#')[1])

                    cur_node = random.choice(
                        self.author_paper_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_PSP(self):
        het_walk_f = open(self.save_path, "a")
        for i in range(self.walk_n):
            for j in range(self.P_n):
                line = ''
                # define header for this walk
                cur_node = "pPaper#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveP
                    line += cur_node + " "
                    # walk and save S
                    cur_node = int(cur_node.split('#')[1])
                    if(len(self.paper_subject_list[cur_node])==0):
                        break
                    else:
                        cur_node = random.choice(
                            self.paper_subject_list[cur_node])
                    line += cur_node + " "
                    # walk P
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.subject_paper_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()
    def het_walk_PTP(self):
        het_walk_f = open(self.save_path, "a")
        for i in range(self.walk_n):
            for j in range(self.P_n):
                line = ''
                # define header for this walk
                cur_node = "pPaper#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveP
                    line += cur_node + " "
                    # walk and save A
                    
                    cur_node = int(cur_node.split('#')[1])
                    if(len(self.paper_term_list[cur_node])==0):
                        break
                    else:
                        cur_node = random.choice(
                            self.paper_term_list[cur_node])
                    line += cur_node + " "
                    # walk P
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.term_paper_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    


class DataGeneratorIMDB(object):
    def __init__(self, walk_n, walk_L, args):
        # 0 for papers, 1 for authors, 2 for subjects(4019、7167、60)
        # adj = sp.load_npz(path + '/adjM.npz').toarray()
        # type_mask = np.load(path + '/node_types.npy')

        features_list, adj, labels, train_val_test_idx, dl = load_data(args.dataset)
        adj=adj.toarray()
        adj=np.triu(adj)
        adj += adj.T - np.diag(adj.diagonal())
        
        # print(adj.shape)
        print(features_list[0].shape)
        print(features_list[1].shape)
        print(features_list[2].shape)
        print(features_list[3].shape)
        type_mask=np.zeros(adj.shape[0])
        type_mask[features_list[0].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]]=1
        type_mask[features_list[0].shape[0]+features_list[1].shape[0]:features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]]=2
        type_mask[features_list[0].shape[0]+features_list[1].shape[0]+features_list[2].shape[0]:]=3
        # mask_p是类型为0的下标list
        mask_m = np.where(type_mask == 0)[0]
        mask_d = np.where(type_mask == 1)[0]
        mask_a = np.where(type_mask == 2)[0]

        # features = sp.load_npz(path + '/features_0.npz').toarray().astype(np.float)
        self.save_path = 'walks_IMDB.txt'

        # paper_word是features每一行的正则化
        # word_paper是features转置后的正则化
        # self.paper_word = row_normalize(features)
        # self.word_paper = row_normalize(features.T)

        self.M_n = mask_m.shape[0]
        self.D_n = mask_d.shape[0]
        self.A_n = mask_a.shape[0]
        self.walk_n = walk_n
        self.walk_L = walk_L

        # paper_author_list是一个二维数组，长度为len(paper),第一个值是一个list,长度为p-a路径中a的数量
        movie_director_list = [[] for _ in range(self.M_n)]
        director_movie_list = [[] for _ in range(self.D_n)]
        movie_actor_list = [[] for _ in range(self.M_n)]
        actor_movie_list = [[] for _ in range(self.A_n)]

        relation_list = [movie_director_list, director_movie_list, movie_actor_list, actor_movie_list]
        header = ['dDirector#', 'mMovie#',  'aActor#', 'mMovie#']

        relations = [adj[mask_m, :][:, mask_d], adj[mask_d, :][:, mask_m], adj[mask_m, :][:, mask_a], adj[mask_a, :][:, mask_m]]
        for k, relation in enumerate(relations):
            for i in range(relation.shape[0]):
                for j in range(relation.shape[1]):
                    if relation[i, j] != 0:
                        relation_list[k][i].append(header[k] + str(j))

        self.movie_director_list = movie_director_list
        self.director_movie_list = director_movie_list
        self.movie_actor_list = movie_actor_list
        self.actor_movie_list = actor_movie_list

    def het_walk_MDM(self):
        het_walk_f = open(self.save_path, "w")
        for i in range(self.walk_n):
            for j in range(self.M_n):
                line = ''
                # define header for this walk
                cur_node = "mMovie#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveP
                    line += cur_node + " "
                    # walk and save A
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.movie_director_list[cur_node])
                    line += cur_node + " "
                    # walk P
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.director_movie_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_MAM(self):
        het_walk_f = open(self.save_path, "a")
        for i in range(self.walk_n):
            for j in range(self.M_n):
                line = ''
                # define header for this walk
                cur_node = "mMovie#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveP
                    line += cur_node + " "
                    # walk and save S
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.movie_actor_list[cur_node])
                    line += cur_node + " "
                    # walk P
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.actor_movie_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    


class DataGeneratorYelp(object):
    def __init__(self, walk_n, walk_L, path):
        adj = sp.load_npz(path + '/adjM.npz').toarray()
        type_mask = np.load(path + '/node_types.npy')
        mask_b = np.where(type_mask == 0)[0]
        mask_u = np.where(type_mask == 1)[0]
        mask_s = np.where(type_mask == 2)[0]
        mask_l = np.where(type_mask == 3)[0]

        self.save_path = 'walks_Yelp.txt'

        features = sp.load_npz(path + '/features_0.npz').toarray()
        self.businesses_word = row_normalize(features)
        self.word_businesses = row_normalize(features.T)

        self.B_n = mask_b.shape[0]
        self.U_n = mask_u.shape[0]
        self.S_n = mask_s.shape[0]
        self.L_n = mask_l.shape[0]
        self.walk_n = walk_n
        self.walk_L = walk_L

        businesses_user_list = [[] for _ in range(self.B_n)]
        user_businesses_list = [[] for _ in range(self.U_n)]
        businesses_service_list = [[] for _ in range(self.B_n)]
        service_businesses_list = [[] for _ in range(self.S_n)]
        businesses_level_list = [[] for _ in range(self.B_n)]
        level_businesses_list = [[] for _ in range(self.L_n)]

        relation_list = [businesses_user_list, user_businesses_list, businesses_service_list, service_businesses_list,
                         businesses_level_list, level_businesses_list]
        header = ['uUser#', 'bBusinesses#', 'sService#', 'bBusinesses#', 'lLevel#', 'bBusinesses#']

        relations = [adj[mask_b, :][:, mask_u], adj[mask_u, :][:, mask_b], adj[mask_b, :][:, mask_s],
                     adj[mask_s, :][:, mask_b], adj[mask_b, :][:, mask_l], adj[mask_l, :][:, mask_b]]
        for k, relation in enumerate(relations):
            for i in range(relation.shape[0]):
                for j in range(relation.shape[1]):
                    if relation[i, j] != 0:
                        relation_list[k][i].append(header[k] + str(j))

        self.businesses_user_list = businesses_user_list
        self.user_businesses_list = user_businesses_list
        self.businesses_service_list = businesses_service_list
        self.service_businesses_list = service_businesses_list
        self.businesses_level_list = businesses_level_list
        self.level_businesses_list = level_businesses_list

    def het_walk_BUB(self):
        het_walk_f = open(self.save_path, "w")
        for i in range(self.walk_n):
            for j in range(self.B_n):
                line = ''
                # define header for this walk
                cur_node = "bBusinesses#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveB
                    line += cur_node + " "
                    # walk and save U
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.businesses_user_list[cur_node])
                    line += cur_node + " "
                    # walk U
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.user_businesses_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_BSB(self):
        het_walk_f = open(self.save_path, "a")
        for i in range(self.walk_n):
            for j in range(self.B_n):
                line = ''
                # define header for this walk
                cur_node = "bBusinesses#" + str(j)
                for l in range(self.walk_L - 1):
                    # saveB
                    line += cur_node + " "
                    # walk and save U
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.businesses_service_list[cur_node])
                    line += cur_node + " "
                    # walk U
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.service_businesses_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_BLB(self):
        het_walk_f = open(self.save_path, "a")
        for i in range(self.walk_n):
            for j in range(self.B_n):
                line = ''
                # define header for this walk
                cur_node = "bBusinesses#" + str(j)
                for l in range(self.walk_L - 1):
                    # save B
                    line += cur_node + " "
                    # walk and save U
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.businesses_level_list[cur_node])
                    line += cur_node + " "
                    # walk U
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = random.choice(
                        self.level_businesses_list[cur_node])
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()

    def het_walk_BWB(self):
        het_walk_f = open(self.save_path, "a")
        for i in range(self.walk_n):
            for j in range(self.B_n):
                line = ''
                # define header for this walk
                cur_node = "bBusinesses#" + str(j)
                for l in range(self.walk_L - 1):
                    # save B
                    line += cur_node + " "
                    # walk and save W
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = 'wWord#' + str(np.random.choice(
                        list(range(self.businesses_word.shape[1])), p=self.businesses_word[cur_node].ravel()))
                    line += cur_node + " "
                    # walk B
                    cur_node = int(cur_node.split('#')[1])
                    cur_node = 'bBusinesses#' + str(np.random.choice(
                        list(range(self.word_businesses.shape[1])), p=self.word_businesses[cur_node].ravel()))
                line = line[:len(line) - 1] + '\n'
                het_walk_f.write(line)
        het_walk_f.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description='args')
    ap.add_argument('--dataset', default='DBLP', help='Dataset.')
    args = ap.parse_args()

    if args.dataset == 'DBLP':
        obj1 = DataGeneratorDBLP(walk_n=40, walk_L=80, args=args)
        obj1.het_walk_APVPA()
        obj1.het_walk_APTPA()
        obj1.het_walk_APA()
        print('DBLP Done')

    if args.dataset == 'ACM':
        obj4 = DataGeneratorACM(walk_n=40, walk_L=80, args=args)
        obj4.het_walk_PAP()
        obj4.het_walk_PSP()
        obj4.het_walk_PTP()
        print('ACM Done')

    if args.dataset == 'IMDB':
        obj4 = DataGeneratorIMDB(walk_n=40, walk_L=80, args=args)
        # obj4.het_walk_MDM()
        # obj4.het_walk_MAM()
        
        print('IMDB Done')

    if args.dataset == 'Yelp':
        obj1 = DataGeneratorYelp(walk_n=40, walk_L=80, path='../data/Yelp')
        obj1.het_walk_BUB()
        obj1.het_walk_BSB()
        obj1.het_walk_BLB()
        obj1.het_walk_BWB()
        print('Yelp Done')