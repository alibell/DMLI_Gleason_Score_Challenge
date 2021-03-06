from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torch

def get_prediction(model, dataset, device):
    """
        Get prediction for each image given a model and a dataset object

        Parameters:
        ----------
        model: model object with a predict function
        dataset: dataset object
        device: device to use for prediction

        Output:
        -------
        Dictionnary containing the prediction : score1, score2, isup_grade
    """

    predictions = {}
    
    model = model.to(device)

    for image_name in tqdm(dataset.get_image_list()):
        image_dataset = dataset.get_subdataset(image_name)
        image_dataset.transform = False
        image_dataloader = DataLoader(image_dataset, batch_size=12, shuffle=False)
        
        predictions[image_name] = {}
        scores_1 = []
        scores_2 = []

        for x, y in image_dataloader:
            x = x.float().to(device)
            with torch.no_grad():
                y_hat, loss_weight = model.predict(x)
            score_1, score_2 = y_hat
            
            scores_1.append(score_1)
            scores_2.append(score_2)

        predictions[image_name]["gleason1"] = torch.cat(scores_1, axis=0).argmax(axis=1).cpu().numpy()
        predictions[image_name]["gleason2"] = torch.cat(scores_2, axis=0).argmax(axis=1).cpu().numpy()

    return predictions    
