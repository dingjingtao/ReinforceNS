import scipy.sparse as sp
import numpy as np
from time import time
from collections import defaultdict
import random


class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trainDict: {u:[i,i,i], ...}
        testRatings: load leave-one-out rating test for class Evaluate
    '''
    def __init__(self, train_path, test_path, num_users, num_items, candidates = 0, report_path = None, evalRecList = False, LRecList = None, write = False):
        self.num_users, self.num_items = num_users, num_items
        # evaluate all the items
        if not evalRecList:
            self.trainDict, self.trainNegDict = self.load_training_file_as_2dict(train_path, candidates = candidates)
            self.trainDictSet = defaultdict(set)
            for u in self.trainDict:
                self.trainDictSet[u] = set(self.trainDict[u])
            self.trainNegDictArray = {}
            for u in self.trainNegDict:
                self.trainNegDictArray[u] = np.array(list(self.trainNegDict[u]))

            self.items_pop_score = self.calcItemPopScore(train_path)
            # print len(self.trainDict), len(self.trainNegDict)


            self.testRatings, self.pretrainGenLabels, self.testNegDictSet = self.load_rating_file_as_2list(test_path)
            # filter out those displayed items clicked in the test dataset
            neg_conflict = 0
            # remove those displays in the earlier session but clicked later (both in train)
            for u in self.trainDict:
                for i in self.trainDict[u]:
                    if u in self.trainNegDict and i in self.trainNegDict[u]:
                        self.trainNegDict[u].remove(i)
                        neg_conflict += 1
                self.trainNegDictArray[u] = np.array(list(self.trainNegDict[u]))
            print neg_conflict
        # evaluate the reclist, generate the list first
        else:
            self.trainDict, self.trainNegDict = self.load_training_file_as_2dict(train_path, candidates=candidates)
            self.trainDictSet = defaultdict(set)
            for u in self.trainDict:
                self.trainDictSet[u] = set(self.trainDict[u])
            self.trainDictArray = {}
            for u in self.trainDict:
                self.trainDictArray[u] = np.array(self.trainDict[u])
            self.trainNegDictArray = {}
            for u in self.trainNegDict:
                self.trainNegDictArray[u] = np.array(list(self.trainNegDict[u]))

            self.items_pop_score = self.calcItemPopScore(train_path)

            random.seed(1)
            self.testRatings, self.testNegDictSet, self.testRecList, self.testFlagList, self.testDictSet = self.load_rating_file_gen_reclist(
                    test_path, LRecList, write, report=False)
            self.reportRatings, self.reportNegDictSet, self.reportRecList, self.reportFlagList, self.reportDictSet = self.load_rating_file_gen_reclist(
                report_path, LRecList, write, report=True)
            random.seed()

    def load_rating_file_gen_reclist(self, filename, LRecList, write, report):
        ratingList = []
        # labelList = []
        user_rating_dict = defaultdict(list)
        user_label_dict = defaultdict(set)
        user_recommend_list = []
        user_flag_list = []
        # counter
        cnt_smallLrec = 0
        cnt_0display = 0
        cnt_displayTo1 = 0
        cnt_L1bgtL0 = 0
        click_notenough = set()
        click_previous = set()
        # time
        begin_time = time()

        with open(filename, "r") as f:
            for line in f.readlines():
                arr = line.split(",")
                if arr[3] == "event_click":
                    user, item = int(arr[0]), int(arr[1])
                    user_rating_dict[user].append(item)
                elif arr[3] == "list_show":
                    user, items = int(arr[0]), arr[1].split("|")
                    user_label_dict[user] = set([int(d) for d in items])

        user_rating_dictSet = defaultdict(set)
        for user in xrange(self.num_users):
            if user_rating_dict[user]:
                ratingList.append([user, user_rating_dict[user]])
                user_rating_dictSet[user] = set(user_rating_dict[user])

                #
                u_click = set(user_rating_dict[user])
                u_display = user_label_dict[user]
                L = len(u_click) + len(u_display)
                if len(u_display) == 0:
                    cnt_0display += 1

                if L <= LRecList:
                    item_sampled = set()
                    for _ in range(0, LRecList-L): # sample non-clicks to fill the list
                        i = random.randint(0, self.num_items - 1)
                        if not report:
                            while i in item_sampled or i in self.trainDictSet[user] or i in u_click or i in u_display:
                                i = random.randint(0, self.num_items - 1)
                        else:
                            while i in item_sampled or i in self.trainDictSet[user] or i in self.testDictSet[user] or i in u_click or i in u_display:
                                i = random.randint(0, self.num_items - 1)
                        item_sampled.add(i)
                    u_display_list = list(u_display)
                    u_click_list = list(u_click)
                    item_sampled_list = list(item_sampled)
                    L1 = len(u_click_list)
                    L0 = LRecList - L1
                else:
                    cnt_smallLrec += 1
                    if not u_display:
                        i = random.randint(0, self.num_items - 1)
                        if not report:
                            while i in self.trainDictSet[user] or i in u_click:
                                i = random.randint(0, self.num_items - 1)
                        else:
                            while i in self.trainDictSet[user] or i in self.testDictSet[user] or i in u_click:
                                i = random.randint(0, self.num_items - 1)
                        u_display.add(i)
                    L = len(u_click) + len(u_display)
                    L0 = int(np.floor(len(u_display) / (0. + L) * LRecList))
                    if L0 < 1:
                        L0 = 1
                        cnt_displayTo1 += 1

                    L1 = LRecList-L0
                    u_display_list = random.sample(u_display, L0)
                    u_click_list = random.sample(u_click, L1)
                    item_sampled_list = []

                recommend_list = []
                recommend_list.extend(item_sampled_list)
                recommend_list.extend(u_display_list)
                recommend_list.extend(u_click_list)
                user_recommend_list.append(np.array(recommend_list, dtype=np.int32))
                tmp = np.zeros(LRecList, dtype=np.int32)
                tmp[-L1:] = 1
                user_flag_list.append(tmp)

                # counter update
                assert (L1>0 and L0>0)
                if L1 > L0:
                    cnt_L1bgtL0 += 1

        print "RecList Stats: click+display>Lrec (%d), 0 display (%d), set display to 1 (%d), L1>L0 (%d), click not enough in train to construct L0 (%d), click previous displays (%d)"\
              %(cnt_smallLrec, cnt_0display, cnt_displayTo1, cnt_L1bgtL0, len(click_notenough), len(click_previous))

        if write:
            if not report:
                listfile1 = filename[0:-len(filename.split("/")[-1])] + "reclist.validation.Len"+str(LRecList)
                listfile2 = filename[0:-len(filename.split("/")[-1])] + "flaglist.validation.Len" + str(LRecList)

            else:
                listfile1 = filename[0:-len(filename.split("/")[-1])] + "reclist.test.Len" + str(LRecList)
                listfile2 = filename[0:-len(filename.split("/")[-1])] + "flaglist.test.Len" + str(LRecList)

            with open(listfile1, "w") as fw1:
                for l in user_recommend_list:
                    for ll in l:
                        fw1.write(str(ll)+',')
                    fw1.write("\n")
            with open(listfile2, "w") as fw2:
                for l in user_flag_list:
                    for ll in l:
                        fw2.write(str(ll)+',')
                    fw2.write("\n")


        user_recommend_list = np.array(user_recommend_list)
        user_flag_list = np.array(user_flag_list)
        end_time = time()
        if not report:
            print "Finished loading the testRatings as list and generating the RecList %.1f s" % (end_time-begin_time)
        else:
            print "Finished loading the reportRatings as list and generating the RecList %.1f s" % (end_time-begin_time)

        return ratingList, user_label_dict, user_recommend_list, user_flag_list, user_rating_dictSet


    def load_rating_file_as_list(self, filename):
        ratingList = []
        user_rating_dict = defaultdict(list)
        with open(filename, "r") as f:
            for line in f.readlines():
                arr = line.split(",")
                user, item = int(arr[0]), int(arr[1])
                user_rating_dict[user].append(item)
        for u in xrange(len(user_rating_dict)):
            ratingList.append([u, user_rating_dict[u]])
        print "Finished loading the testRatings as list..."
        return ratingList

    def load_rating_file_as_2list(self, filename):
        ratingList = []
        labelList = []
        user_rating_dict = defaultdict(list)
        user_label_dict = defaultdict(set)
        with open(filename, "r") as f:
            for line in f.readlines():
                arr = line.split(",")
                if arr[3] == "event_click":
                    user, item = int(arr[0]), int(arr[1])
                    user_rating_dict[user].append(item)
                elif arr[3] == "list_show":
                    user, items = int(arr[0]), arr[1].split("|")
                    label_num = len(user_rating_dict[user])
                    label1s = [(user,int(d),1) for d in items[0:min(len(items),label_num)]]
                    label0s = [(user,int(d),0) for d in user_rating_dict[user]]
                    labelList.extend(label1s)
                    labelList.extend(label0s)

                    user_label_dict[user] = set([int(d) for d in items])


        for u in xrange(len(user_rating_dict)):
            ratingList.append([u, user_rating_dict[u]])

        print "Finished loading the testRatings as list ..."
        return ratingList, np.array(labelList), user_label_dict

    def load_training_file_as_2dict(self, filename, candidates = 0):
        trainDict = defaultdict(list)
        trainNegDict = defaultdict(list)

        user_neg_comp =  0

        s_pre = ""
        with open(filename, "r") as f:
            for line in f.readlines():
                arr = line.split(",")
                s = arr[4]
                if s != s_pre:
                    click_i = []
                    s_pre = s
                if arr[3] == "event_click":
                    u, i = int(arr[0]), int(arr[1])
                    click_i.append(i)
                    trainDict[u].append(i)
                elif arr[3] == "list_show":
                    u, i_ns = int(arr[0]), [int(d) for d in arr[1].split("|")]
                    trainNegDict[u].extend(i_ns)
        for u in trainNegDict:
            trainNegDict[u] = set(trainNegDict[u])
            if len(trainNegDict[u]) < candidates/2:
                user_neg_comp += 1
            # if returnNegasArray:
            #     trainNegDict[u] = np.array(list(trainNegDict[u]))
            trainDict[u].sort()
        print "Finished loading the trainDict as dict (with Neg)..."
        print "users with less than <candidates> negs: %d" % user_neg_comp
        return trainDict,trainNegDict

    def load_training_file_as_dict(self, filename):
        trainDict = defaultdict(list)
        with open(filename, "r") as f:
            for line in f.readlines():
                arr = line.split(",")
                u, i = int(arr[0]), int(arr[1])
                trainDict[u].append(i)
        print "Finished loading the trainDict as dict..."
        return trainDict

    def calcItemPopScore(self, filename):
        items_pop_score = np.zeros([self.num_items, 1], dtype=np.float32)
        with open(filename, "r") as f:
            for line in f.readlines():
                arr = line.split(",")
                if arr[3] == "event_click":
                    u, i = int(arr[0]), int(arr[1])
                    items_pop_score[i] += 1
        print "Finished loading items_pop_score ..."
        return items_pop_score

