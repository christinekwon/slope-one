import sys
import numpy as np
import math

U = None # num users
I = None # num items
R = None # num ratings

# these maps are used because the id of each user/movie doesn't necessarily 
# lie within the range of 0 ~ U / 0 ~ M; it can be any integer
u_id2index = {} # map the id of each user to its index in the ratings matrix
i_id2index = {} # map the id of each movie to its index in the ratings matrix
u_index2id = {} # reverse map index --> id
i_index2id = {} # reverse map index --> id

ratings = None # will later be initialized to shape (U, M)
rec_user = None # id of user to recommend to
rec_count = None # number of recommendations to make

# open the specified file and assign the correct values to each variable
def init_data(fileName):
    file = open('./input/' + fileName, 'r')

    global U, I, R, ratings, rec_user, rec_count

    line_count = 0
    u_count = 0
    i_count = 0

    for line in file.readlines():

        # initialize the number of users and items, 
        # and with those values, create an empty array of ratings
        if line_count == 0:
            data = line.split()
            U = int(data[0])
            I = int(data[1])
            ratings = np.zeros((U, I))

        # init number of ratings provided
        elif line_count == 1:
            R = int(line)

        # for each rating info, 
        elif line_count >= 2 and line_count <= R + 1:
            data = line.split()
            u_id = int(data[0]) # current user id
            i_id = int(data[1]) # current movie id
            r = int(data[2]) # current rating
            u_index = None # user id mapped to an index between 0 ~ U
            i_index = None # item id mapped to an index between 0 ~ I

            # check if the current user has already rated something, and init u_index accordingly
            if u_id in u_id2index:
                u_index = u_id2index[u_id]
            else:
                u_index = u_count
                u_id2index[u_id] = u_count
                u_index2id[u_count] = u_id
                u_count += 1
            # check if the current move has already been rated, and init i_index accordingly
            if i_id in i_id2index:
                i_index = i_id2index[i_id]
            else:
                i_index = i_count
                i_id2index[i_id] = i_count
                i_index2id[i_count] = i_id
                i_count += 1

            # update ratings matrix
            ratings[u_index][i_index] = r

        # init ID of user to recommend to
        elif line_count == R + 2:
            rec_user = int(line)

        # init number of recommendations to make
        elif line_count == R + 3:
            rec_count = int(line)
        else:
            sys.stderr.write("Exceeded number of ratings.\n")
            exit()

        line_count += 1

def print_data():
    print(U)
    print(I)
    print(R)
    print(ratings)
    print(u_id2index)
    print(u_index2id)
    print(i_id2index)
    print(i_index2id)
    print(rec_user)
    print(rec_count)

# initialize the average deviation matrix
# data: the ratings matrix to use as reference
# dev: an empty deviaton matrix of shape (1,1) to fill out with the deviations of each item combination (skew-symmetric matrix)
# rel: an empty matrix of shape (1,1) that keeps track of how many users have simultaneously rated each item combination (symmetrix matrix)
# ie. if rel[1][0] = 3, then that means 3 users have rated both items 1 and 0
# returns updated dev and rel matrices
def init_dev(data, dev, rel):
    for j in range(I):
        # since the deviation matrix is a skew-symmetric matrix, we only need to visit I^2 / 2 - I of the indices (above the TL --> BR diagonal)
        for i in range(j + 1, I):
            rel_count = 0
            dev_sum = 0 # running sum of deviations for current item pair
            for u in range(U):
                if data[u][j] != 0 and data[u][i] != 0:
                    rel[j][i] += 1
                    rel[i][j] += 1
                    rel_count += 1
                    dev_sum += data[u][j] - data[u][i]
            if rel_count > 0:
                dev[j][i] = dev_sum / rel_count
                dev[i][j] = -1 * dev_sum / rel_count

    return dev, rel

# given the map of predictions, return the top 'rec_count' predictions
# output is a list of item ids and a list of their corresponding ratings (in matching order)
def filter_preds(preds):
    item_ids = []
    filtered_ratings = []

    # throw error is there are fewer predictions than required by stdin
    if len(preds) < rec_count:
        sys.stderr.write('Not enough data to recommend appropriate items.\n')
        exit()
    
    # pop the item with the maximum predicted rating and update id/ratings lists, rec_count times
    for i in range(rec_count):
        max_key = max(preds, key=lambda k:preds[k])
        item_ids.append(max_key)
        filtered_ratings.append(preds[max_key])
        preds.pop(max_key)

    return item_ids, filtered_ratings

# format the rating so it is rounded up to three decimal places
def format_rating(rating, places):
    rating *= 10**places
    rating = math.ceil(rating)
    rating /= 10**places
    return rating

# the basic slope one algorithm
def slope_one():
    dev = np.zeros((I, I)) # store the average deviation of item i with respect to item j
    rel = np.zeros((I, I)) # track whether two items are relevant to eachother i.e. have been rated by a common user
    dev, rel = init_dev(ratings, dev, rel)

    # predict slope one
    rec_user_index = u_id2index[rec_user]
    rec_user_ratings = ratings[rec_user_index]
    predictions = {}

    # go through each item and predict a rating for an item if it hasn't been rated by the user yet
    for j in range(I):
        pred_sum = 0
        count = 0
        # don't bother if the target user has already rated this item
        if rec_user_ratings[j] != 0:
            continue
        for i in range(I):
            # check if item i is relevant:
            # i.e i was rated by the user, and has been rated with j by some user
            if rec_user_ratings[i] != 0 and rel[j][i] > 0:
                pred_sum += dev[j][i] + rec_user_ratings[i]
                count += 1
        if count > 0:
            predictions[i_index2id[j]] = pred_sum / count

    filtered_ids, filtered_ratings = filter_preds(predictions)
    return filtered_ids, filtered_ratings

# the weighted slope one algorithm
def weighted_slope_one():
    dev = np.zeros((I, I)) # store the average deviation of item i with respect to item j
    rel = np.zeros((I, I)) # track whether two items are relevant to eachother i.e. have been rated by a common user
    dev, rel = init_dev(ratings, dev, rel)

    # predict slope one
    rec_user_index = u_id2index[rec_user]
    rec_user_ratings = ratings[rec_user_index]
    predictions = {}

    # go through each item and predict a rating for an item if it hasn't been rated by the user yet
    for j in range(I):
        num_sum = 0
        den_sum = 0
        # don't bother if the target user has already rated this item
        if rec_user_ratings[j] != 0:
            continue
        for i in range(I):
            if i != j and rec_user_ratings[i] != 0:
                num_sum += (dev[j][i] + rec_user_ratings[i]) * rel[j][i]
                den_sum += rel[j][i]
        if den_sum > 0:
            predictions[i_index2id[j]] = num_sum / den_sum

    filtered_ids, filtered_ratings = filter_preds(predictions)
    return filtered_ids, filtered_ratings

# the bipolar slope one algorithm
def bipolar_slope_one():

    # separate ratings matrix into likes and dislikes
    like = np.zeros((U, I))
    dislike = np.zeros((U, I))

    for j in range(U):
        avg = np.average(ratings[j])
        for i in range(I):
            curr = ratings[j][i]
            if curr >= avg:
                like[j][i] = curr
            else:
                dislike[j][i] = curr

    # calculate the deviation and relevance matrices for liked items, as well as disliked items
    dev_like = np.zeros((I, I))
    rel_like = np.zeros((I, I))
    dev_dislike = np.zeros((I, I))
    rel_dislike = np.zeros((I, I))

    dev_like, rel_like = init_dev(like, dev_like, rel_like)
    dev_dislike, rel_dislike = init_dev(dislike, dev_dislike, rel_dislike)

    rec_user_index = u_id2index[rec_user]
    rec_user_likes = like[rec_user_index]
    rec_user_dislikes = dislike[rec_user_index]
    predictions = {}

    # go through each item and predict a rating for an item if it hasn't been rated by the user yet
    for j in range(I):
        like_pred_sum = 0
        dislike_pred_sum = 0
        like_weight_sum = 0
        dislike_weight_sum = 0

        # don't bother if the target user has already rated this item
        if ratings[rec_user_index][j] != 0:
            continue
        for i in range(I):
            if i == j:
                continue
            if rec_user_likes[i] != 0:
                like_pred_sum += (dev_like[j][i] + rec_user_likes[i]) * rel_like[j][i]
                like_weight_sum += rel_like[j][i]
            if rec_user_dislikes[i] != 0:
                dislike_pred_sum += (dev_dislike[j][i] + rec_user_dislikes[i]) * rel_dislike[j][i]
                dislike_weight_sum += rel_dislike[j][i]

        if like_weight_sum + dislike_weight_sum > 0:
            predictions[i_index2id[j]] = (like_pred_sum + dislike_pred_sum) / (like_weight_sum + dislike_weight_sum)

    filtered_ids, filtered_ratings = filter_preds(predictions)
    return filtered_ids, filtered_ratings