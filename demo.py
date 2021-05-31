import sys
from slope_one import init_data, print_data, filter_preds, slope_one, weighted_slope_one, bipolar_slope_one

# with a list of ids and ratings, write them to stdout in the proper format
def write_preds(ids, ratings):
    for x in range(len(ids)):
        sys.stdout.write(str(ids[x]) + ' ' + str('%0.3f'%ratings[x]) + '\n')

if __name__ == '__main__':

    init_data(sys.argv[1])

    sys.stdout.write('\nSLOPE ONE PREDICTION\n\n')

    basic_ids, basic_ratings = slope_one()
    write_preds(basic_ids, basic_ratings)

    sys.stdout.write('\nWEIGHTED SLOPE ONE PREDICTION\n\n')

    weighted_ids, weighted_ratings = weighted_slope_one()
    write_preds(weighted_ids, weighted_ratings)

    sys.stdout.write('\nBIPOLAR SLOPE ONE PREDICTION\n\n')

    bipolar_ids, bipolar_ratings = bipolar_slope_one()
    write_preds(bipolar_ids, bipolar_ratings)

