import numpy as np
import os


class DataHandler:
    def __init__(self, train, test, region_size=30, pool_size=40):
        self.train = train
        self.test = test
        self.classes = [os.path.splitext(
            e.split('/')[-1].split('_')[0])[0] for e in self.test]
        self.num_classes = len(self.classes)
        self.max_len = 0
        # 5 is the background signal for padding
        self.vocabulary = ['A', 'C', 'G', 'T', 'N', 5]
        self.vocab_size = len(self.vocabulary)
        self.x_train, self.y_train, self.seqs_len_train, self.seqs_ref_train = self.get_data(
            self.train)
        self.x_test, self.y_test, self.seqs_len_test, self.seqs_ref_test = self.get_data(
            self.test)
        self.train_size = len(self.y_train)
        self.test_size = len(self.y_test)
#        while(self.max_len - region_size + 1)%pool_size != 0:
#            self.max_len += 1
        self.x_train = np.array([np.pad(self.x_train[i], (0, self.max_len-len(self.x_train[i])), 'constant', constant_values=(0, 0))
                                # pads with zero the sequences with length different from max_len
                                 for i in range(self.train_size)])
        self.x_test = np.array([np.pad(self.x_test[i], (0, self.max_len-len(self.x_test[i])), 'constant', constant_values=(0, 0))
                               # pads with zero the sequences with length different from max_len
                                for i in range(self.test_size)])

    def get_data(self, files):
        x = []
        y = []
        len_seqs = []
        ref_seqs = []
        counter = 0
        for fl in files:
            with open(fl, 'r') as f:
                print("processing the file: "+fl+" as class "+str(counter))
                counter += 1
                cl = self.classes.index(os.path.splitext(
                    fl.split('/')[-1].split('_')[0])[0])
                for l in f.readlines():
                    if l[0] == '>':
                        ref_seqs.append(l)

                        if x != []:
                            len_seqs.append(len(x[-1]))
                            x[-1] = np.array([1 if c == 'A' else 2 if c == 'C' else 3 if c ==
                                             'G' else 4 if c == 'T' else 5 for c in x[-1]], dtype=np.uint8)
                            if len(x[-1]) > self.max_len:
                                self.max_len = len(x[-1])
                        x.append('')
                        y.append(cl)
                    else:
                        x[-1] += l.upper().strip()

                x[-1] = np.array([1 if c == 'A' else 2 if c == 'C' else 3 if c ==
                                 'G' else 4 if c == 'T' else 5 for c in x[-1]], dtype=np.uint8)
                if len(x[-1]) > self.max_len:
                    self.max_len = len(x[-1])

        len_seqs.append(len(x[-1]))

        return x, np.array([e for e in y]), len_seqs, ref_seqs
