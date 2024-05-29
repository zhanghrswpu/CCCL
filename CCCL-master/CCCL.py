
r"""
CCCL
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import faiss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class CCCL(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CCCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.device = torch.device("cuda")
        # load parameters info
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.gender_temp = config['gender_temp']
        self.gender_weight = config['gender_weight']

        self.occupation_temp = config['occupation_temp']
        self.occupation_weight = config['occupation_weight']

        self.age_temp = config['age_temp']
        self.age_weight = config['age_weight']

        self.alpha = config['alpha']

        self.com_reg = config['com_reg']
        self.k = config['num_clusters']

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()


        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)


        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.user_age = None
        self.user_gender = None
        self.user_occupation = None
        self.indices1 = None
        self.indices2 = None
        self.indices3 = None
        self.indices4 = None
        self.indices5 = None
        self.indices6 = None
        self.indices7 = None
        self.indices8 = None
        self.indices9 = None
        self.indices10 = None
        self.indices11 = None
        self.indices12 = None
        self.indices13 = None
        self.indices14 = None
        self.indices15 = None
        self.indices16 = None
        self.indices17 = None
        self.indices18 = None
        self.indices19 = None
        self.indices20 = None
        self.indices21 = None
        self.indices_age1 = None
        self.indices_age2 = None
        self.indices_age3 = None
        self.indices_age4 = None
        self.indices_age5 = None
        self.indices_age6 = None
        self.indices_age7 = None

    def user_attribute(self,data):
        user = data.dataset.user_feat.interaction
        age = user['age']
        age = age[torch.arange(age.size(0)) != 0]
        gender = user['gender']
        gender = gender[torch.arange(gender.size(0)) != 0]
        occupation = user['occupation']
        occupation = occupation[torch.arange(occupation.size(0)) != 0]

        self.occupation_indice(occupation)
        self.age_indice(age)
        self.user_age = age
        self.user_gender = gender
        self.user_occupation = occupation

    def gender(self,user_embeddings):
        user_gender = self.user_gender.numpy()
        user_embeddings, current_item_embeddings = torch.split(user_embeddings, [self.n_users, self.n_items])
        temp, user_embeddings = torch.split(user_embeddings, [1, self.n_users - 1])
        two_indices = np.where(user_gender == 2)
        one_indices = np.where(user_gender == 1)

        user_embeddings = F.normalize(user_embeddings).detach().cpu().numpy()
        two_values = torch.from_numpy(user_embeddings[two_indices])
        one_values = torch.from_numpy(user_embeddings[one_indices])
        user_embeddings = torch.from_numpy(user_embeddings).to(self.device)

        two_center_values = two_values.mean(dim=0).unsqueeze(0).to(self.device)
        one_center_values = one_values.mean(dim=0).unsqueeze(0).to(self.device)
        centroids = torch.cat((two_center_values,one_center_values),dim = 0).to(self.device)

        num = 0
        for value in user_gender:
            if num == 0:
                if value == 2:
                    result = two_center_values.to(self.device)
                else:
                    result = one_center_values.to(self.device)
            else:
                if value == 2:
                    result = torch.cat((result, two_center_values)).to(self.device)
                else:
                    result = torch.cat((result, one_center_values)).to(self.device)
            num = num + 1

        pos_score_user = torch.mul(user_embeddings, result).sum(dim=1).to(self.device)
        pos_score_user = torch.exp(pos_score_user / self.gender_temp)
        ttl_score_user = torch.matmul(user_embeddings, centroids.transpose(0, 1)).to(self.device)
        ttl_score_user = torch.exp(ttl_score_user / self.gender_temp).sum(dim=1)

        gender_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        return gender_loss_user

    def age_indice(self,user_age):
        user_age = user_age.numpy()

        self.indices_age1 = np.where(user_age == 1)
        self.indices_age2 = np.where(user_age == 2)
        self.indices_age3 = np.where(user_age == 3)
        self.indices_age4 = np.where(user_age == 4)
        self.indices_age5 = np.where(user_age == 5)
        self.indices_age6 = np.where(user_age == 6)
        self.indices_age7 = np.where(user_age == 7)


    def occupation_indice(self,user_occupation):
        user_occupation = user_occupation.numpy()

        self.indices1 = np.where(user_occupation == 1)
        self.indices2 = np.where(user_occupation == 2)
        self.indices3 = np.where(user_occupation == 3)
        self.indices4 = np.where(user_occupation == 4)
        self.indices5 = np.where(user_occupation == 5)
        self.indices6 = np.where(user_occupation == 6)
        self.indices7 = np.where(user_occupation == 7)
        self.indices8 = np.where(user_occupation == 8)
        self.indices9 = np.where(user_occupation == 9)
        self.indices10 = np.where(user_occupation == 10)
        self.indices11 = np.where(user_occupation == 11)
        self.indices12 = np.where(user_occupation == 12)
        self.indices13 = np.where(user_occupation == 13)
        self.indices14 = np.where(user_occupation == 14)
        self.indices15 = np.where(user_occupation == 15)
        self.indices16 = np.where(user_occupation == 16)
        self.indices17 = np.where(user_occupation == 17)
        self.indices18 = np.where(user_occupation == 18)
        self.indices19 = np.where(user_occupation == 19)
        self.indices20 = np.where(user_occupation == 20)
        self.indices21 = np.where(user_occupation == 21)

    def age(self,user_embeddings):
        user_age = self.user_age.numpy()
        user_embeddings, current_item_embeddings = torch.split(user_embeddings, [self.n_users, self.n_items])
        temp, user_embeddings = torch.split(user_embeddings, [1, self.n_users - 1])

        user_embeddings = F.normalize(user_embeddings).detach().cpu().numpy()
        values1 = torch.from_numpy(user_embeddings[self.indices_age1]).mean(dim=0).unsqueeze(0)
        values2 = torch.from_numpy(user_embeddings[self.indices_age2]).mean(dim=0).unsqueeze(0)
        values3 = torch.from_numpy(user_embeddings[self.indices_age3]).mean(dim=0).unsqueeze(0)
        values4 = torch.from_numpy(user_embeddings[self.indices_age4]).mean(dim=0).unsqueeze(0)
        values5 = torch.from_numpy(user_embeddings[self.indices_age5]).mean(dim=0).unsqueeze(0)
        values6 = torch.from_numpy(user_embeddings[self.indices_age6]).mean(dim=0).unsqueeze(0)
        values7 = torch.from_numpy(user_embeddings[self.indices_age7]).mean(dim=0).unsqueeze(0)

        user_embeddings = torch.from_numpy(user_embeddings).to(self.device)

        centroids = np.concatenate((values1, values2, values3, values4,values5, values6, values7), axis=0)
        centroids = torch.from_numpy(centroids).to(self.device)

        num = 0
        result = values1.to(self.device)
        for value in user_age:
            if value == 1:
                result = torch.cat((result, values1.to(self.device)))
            elif value == 2:
                result = torch.cat((result, values2.to(self.device)))
            elif value == 3:
                result = torch.cat((result, values3.to(self.device)))
            elif value == 4:
                result = torch.cat((result, values4.to(self.device)))
            elif value == 5:
                result = torch.cat((result, values5.to(self.device)))
            elif value == 6:
                result = torch.cat((result, values6.to(self.device)))
            elif value == 7:
                result = torch.cat((result, values7.to(self.device)))
            num = num + 1

        temp, result = torch.split(result, [1, self.n_users - 1])

        pos_score_user = torch.mul(user_embeddings, result).sum(dim=1).to(self.device)
        pos_score_user = torch.exp(pos_score_user / self.age_temp)
        ttl_score_user = torch.matmul(user_embeddings, centroids.transpose(0, 1)).to(self.device)
        ttl_score_user = torch.exp(ttl_score_user / self.age_temp).sum(dim=1)

        age_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        return age_loss_user

    def occupation(self,user_embeddings):
        user_occupation = self.user_occupation.numpy()
        user_embeddings, current_item_embeddings = torch.split(user_embeddings, [self.n_users, self.n_items])
        temp, user_embeddings = torch.split(user_embeddings, [1, self.n_users - 1])

        user_embeddings = F.normalize(user_embeddings).detach().cpu().numpy()
        values1 = torch.from_numpy(user_embeddings[self.indices1]).mean(dim=0).unsqueeze(0)
        values2 = torch.from_numpy(user_embeddings[self.indices2]).mean(dim=0).unsqueeze(0)
        values3 = torch.from_numpy(user_embeddings[self.indices3]).mean(dim=0).unsqueeze(0)
        values4 = torch.from_numpy(user_embeddings[self.indices4]).mean(dim=0).unsqueeze(0)
        values5 = torch.from_numpy(user_embeddings[self.indices5]).mean(dim=0).unsqueeze(0)
        values6 = torch.from_numpy(user_embeddings[self.indices6]).mean(dim=0).unsqueeze(0)
        values7 = torch.from_numpy(user_embeddings[self.indices7]).mean(dim=0).unsqueeze(0)
        values8 = torch.from_numpy(user_embeddings[self.indices8]).mean(dim=0).unsqueeze(0)
        values9 = torch.from_numpy(user_embeddings[self.indices9]).mean(dim=0).unsqueeze(0)
        values10 = torch.from_numpy(user_embeddings[self.indices10]).mean(dim=0).unsqueeze(0)
        values11 = torch.from_numpy(user_embeddings[self.indices11]).mean(dim=0).unsqueeze(0)
        values12 = torch.from_numpy(user_embeddings[self.indices12]).mean(dim=0).unsqueeze(0)
        values13 = torch.from_numpy(user_embeddings[self.indices13]).mean(dim=0).unsqueeze(0)
        values14 = torch.from_numpy(user_embeddings[self.indices14]).mean(dim=0).unsqueeze(0)
        values15 = torch.from_numpy(user_embeddings[self.indices15]).mean(dim=0).unsqueeze(0)
        values16 = torch.from_numpy(user_embeddings[self.indices16]).mean(dim=0).unsqueeze(0)
        values17 = torch.from_numpy(user_embeddings[self.indices17]).mean(dim=0).unsqueeze(0)
        values18 = torch.from_numpy(user_embeddings[self.indices18]).mean(dim=0).unsqueeze(0)
        values19 = torch.from_numpy(user_embeddings[self.indices19]).mean(dim=0).unsqueeze(0)
        values20 = torch.from_numpy(user_embeddings[self.indices20]).mean(dim=0).unsqueeze(0)
        values21 = torch.from_numpy(user_embeddings[self.indices21]).mean(dim=0).unsqueeze(0)

        user_embeddings = torch.from_numpy(user_embeddings).to(self.device)

        centroids = np.concatenate((values1, values2, values3, values4,values5, values6, values7, values8,values9, values10, values11, values12,values13, values14, values15, values16,values17, values18, values19, values20, values21), axis=0)
        centroids = torch.from_numpy(centroids).to(self.device)

        num = 0
        result = values1.to(self.device)
        for value in user_occupation:
            if value == 1:
                result = torch.cat((result, values1.to(self.device)))
            elif value == 2:
                result = torch.cat((result, values2.to(self.device)))
            elif value == 3:
                result = torch.cat((result, values3.to(self.device)))
            elif value == 4:
                result = torch.cat((result, values4.to(self.device)))
            elif value == 5:
                result = torch.cat((result, values5.to(self.device)))
            elif value == 6:
                result = torch.cat((result, values6.to(self.device)))
            elif value == 7:
                result = torch.cat((result, values7.to(self.device)))
            elif value == 8:
                result = torch.cat((result, values8.to(self.device)))
            elif value == 9:
                result = torch.cat((result, values9.to(self.device)))
            elif value == 10:
                result = torch.cat((result, values10.to(self.device)))
            elif value == 11:
                result = torch.cat((result, values11.to(self.device)))
            elif value == 12:
                result = torch.cat((result, values12.to(self.device)))
            elif value == 13:
                result = torch.cat((result, values13.to(self.device)))
            elif value == 14:
                result = torch.cat((result, values14.to(self.device)))
            elif value == 15:
                result = torch.cat((result, values15.to(self.device)))
            elif value == 16:
                result = torch.cat((result, values16.to(self.device)))
            elif value == 17:
                result = torch.cat((result, values17.to(self.device)))
            elif value == 18:
                result = torch.cat((result, values18.to(self.device)))
            elif value == 19:
                result = torch.cat((result, values19.to(self.device)))
            elif value == 20:
                result = torch.cat((result, values20.to(self.device)))
            elif value == 21:
                result = torch.cat((result, values21.to(self.device)))
            num = num + 1

        temp, result = torch.split(result, [1, self.n_users - 1])

        pos_score_user = torch.mul(user_embeddings, result).sum(dim=1).to(self.device)
        pos_score_user = torch.exp(pos_score_user / self.occupation_temp)
        ttl_score_user = torch.matmul(user_embeddings, centroids.transpose(0, 1)).to(self.device)
        ttl_score_user = torch.exp(ttl_score_user / self.occupation_temp).sum(dim=1)

        occupation_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        return occupation_loss_user

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):

        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self):

        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)

        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D

        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).to(self.device)

        return SparseL

    def get_ego_embeddings(self):

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings().to(self.device)
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings).to(self.device)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def COMMON_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]
        user2centroids = self.user_centroids[user2cluster]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        com_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]
        item2centroids = self.item_centroids[item2cluster]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        com_item = -torch.log(pos_score_item / ttl_score_item).sum()

        common_loss = self.com_reg * (com_user + com_item)
        return common_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        gender_loss_user = self.gender(context_embedding)*self.gender_weight
        occupation_loss_user = self.occupation(context_embedding)*self.occupation_weight
        age_loss_user = self.age(context_embedding)*self.age_weight

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        common_loss = self.COMMON_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]


        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user).to(self.device)
        pos_ego_embeddings = self.item_embedding(pos_item).to(self.device)
        neg_ego_embeddings = self.item_embedding(neg_item).to(self.device)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss ,ssl_loss, gender_loss_user+occupation_loss_user+age_loss_user ,common_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()

        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
