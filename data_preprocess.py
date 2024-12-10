import pandas as pd
import numpy as np
import os
import torch


def tsv_to_csv(tsv_fnames=['MicroLens-100k_pairs.tsv'], csv_folders=["ks"], user_history_length=10):
    user_history_length = user_history_length
    tsv_fnames = tsv_fnames
    csv_folders = csv_folders

    for idx in range(len(tsv_fnames)):
        dat_seq = pd.read_csv(tsv_fnames[idx], sep='\t',header=None)
        dat_arr = np.array(dat_seq)
        inter = []
        for seq in dat_arr:
            uid = seq[0]
            iseq = seq[1].split()
            for i, item in enumerate(iseq):
                inter.append([item, uid, i])

        inter_df = np.array(inter)
        dat = pd.DataFrame(inter_df)
        dat.columns = ['item_id', 'user_id', 'timestamp']
        dat['timestamp'] = dat['timestamp'].astype(int)
        dat.sort_values(by='timestamp', inplace=True, ascending=True)
        user_list = dat['user_id'].values
        item_list = dat['item_id'].values

        index = {}
        for i, key in enumerate(user_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)

                indices = []

        for index in index.values():
            indices.extend(list(index)[-(user_history_length+3):])

        csv_data = dict()
        for k in dat:
            csv_data[k] = dat[k].values[indices]

        csv_data = pd.DataFrame(csv_data)
        print(csv_data.head(3))
        print(csv_data['user_id'].nunique(),csv_data['item_id'].nunique(),csv_data.shape[0] )
        os.makedirs(f'./{csv_folders[idx]}/', exist_ok=True)
        csv_data.to_csv(f'./{csv_folders[idx]}/{csv_folders[idx]}.inter', index=False)


class Data:
    def __init__(self, df, user_history_length=10):
        self.inter_feat = df
        self.user_history_length = user_history_length
        self._data_processing()


    def _data_processing(self):

        self.id2token = {}
        self.token2id = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            feats = self.inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats)
            mp = np.array(['[PAD]'] + list(mp))
            token_id = {t: i for i, t in enumerate(mp)}
            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            self.inter_feat[feature] = new_ids_list+1

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']


    def build(self):

        self.sort(by='timestamp')
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        grouped_index = self._grouped_index(user_list)

        user_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]

        self.user_seq = user_seq
        train_feat = dict()
        test_feat = dict()
        valid_feat = dict()
        indices = []

        for index in grouped_index.values():
            indices.extend(list(index)[:-2])
        for k in self.inter_feat:
            train_feat[k] = self.inter_feat[k].values[indices]

        indices = []
        for index in grouped_index.values():
            indices.extend([index[-2]])
        for k in self.inter_feat:
            valid_feat[k] = self.inter_feat[k].values[indices]

        indices = []
        for index in grouped_index.values():
            indices.extend([index[-1]])
        for k in self.inter_feat:
            test_feat[k] = self.inter_feat[k].values[indices]

        self.train_feat = train_feat
        return train_feat, valid_feat, test_feat


    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index

    def _build_seq(self, train_feat):
        max_item_list_len = self.user_history_length+1
        uid_list, item_list_index= [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid :
                save = True
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0],sub[-1]+1))


                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start,i))


                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        seq_train_item = []
        for index in item_list_index:
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])
            seq_train_item+=list(train_feat['item_id'][index])

        self.seq_train_item = seq_train_item
        return seq_train_feat


    def sort(self, by, ascending=True):
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)


def generate_popnpy(csv_folders_list = ['ks', ]):
    '''
    Generates the popularity count file (i.e. the pop.npy file needed in baseline code) of the dataset.
    '''

    for idx in range(len(csv_folders_list)):
        inter = pd.read_csv(f'./{csv_folders_list[idx]}/{csv_folders_list[idx]}.inter',
                            delimiter=',', dtype={'item_id':str, 'user_id':str, 'timestamp':int}, header=0, names=['item_id', 'user_id', 'timestamp']
                )

        item_num = inter['item_id'].nunique()
        D  = Data(inter)
        train, valid, test = D.build()
        D._build_seq(train)
        train_items = D.seq_train_item
        train_item_counts = [0] * (item_num + 1)
        for i in train_items:
            train_item_counts[i] += 1
        item_counts_powered = np.power(train_item_counts, 1.0)
        pop_prob_list = []

        for i in range(1, item_num + 1):
            pop_prob_list.append(item_counts_powered[i])
        pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
        pop_prob_list = np.append([1], pop_prob_list)
        print(('prob max: {}, prob min: {}, prob mean: {}'.\
                format(max(pop_prob_list), min(pop_prob_list), np.mean(pop_prob_list))))

        np.save(f'./{csv_folders_list[idx]}/pop',pop_prob_list)

if __name__ == "__main__":
    # generate data files needed for the NFM model: ks.inter and pop.npy inside ks folder
    tsv_to_csv()
    generate_popnpy()