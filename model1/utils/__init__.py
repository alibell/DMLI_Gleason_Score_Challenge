import numpy as np
import pandas as pd

def dataframe_from_predictions (prediction, main_df, bincount=True):
    prediction_df = pd.DataFrame(prediction).T \
        .reset_index() \
        .assign(index=lambda x: x["index"].apply(lambda y: y.split(".")[0])) \
        .rename(columns={"index":"image_id", "gleason1":"gleason1_predicted", "gleason2":"gleason2_predicted"})

    main_df = main_df \
        .assign(gleason_score_clean = lambda x: x["gleason_score"].replace("negative", "0+0")) \
        .assign(gleason1=lambda x: x["gleason_score_clean"].apply(lambda y: y.split("+")[0]).astype("int")) \
        .assign(gleason2=lambda x: x["gleason_score_clean"].apply(lambda y: y.split("+")[1]).astype("int")) \
        .drop(columns=["gleason_score_clean"])

    if bincount:
        for column in ["gleason1_predicted", "gleason2_predicted"]:
            prediction_df[column] = prediction_df[column].apply(lambda x: np.bincount(x, minlength=6))

    output_df = pd.merge(
        prediction_df,
        main_df,
        left_on="image_id",
        right_on="image_id",
        how="inner"
    )

    return output_df