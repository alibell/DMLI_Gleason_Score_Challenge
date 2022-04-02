from numpy import bincount
import pandas as pd

def dataframe_from_predictions (prediction, main_df, bincount=True):
    prediction_df = pd.DataFrame(prediction).T \
        .reset_index() \
        .assign(index = lambda x: x["index"].apply(lambda y: y.split(".")[0])) \
        .rename(columns = {"index":"image_id"})

    if bincount:
        for column in ["gleason1", "gleason2"]:
            prediction_df[column] = prediction_df[column].apply(lambda x: np.bincount(x, minlength=6))

    output_df = pd.merge(
        prediction_df,
        main_df,
        left_on="image_id",
        right_on="image_id",
        how="inner"
    )

    return output_df