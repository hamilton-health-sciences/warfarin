import pandas as pd
import numpy as np
import torch
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F
import math 

from .network_architecture import FC_Q
from .threshold_model import ThresholdModel
from .constants import Constants
# from .replay_buffer_ded import ReplayBufferDed


class Model():

    def __init__(self, model_path=None, model_name=None, num_layers=3, hidden_states=25, state_dim=1, num_actions=3,
                 bcq_threshold=0.3, max_inr=4.5, min_inr=0.5, device="cpu", model=None):
        
        assert (model is not None) or (model_path is not None), "Model path and model cannot both be None"
        
        if model is None:
            model = FC_Q(state_dim=state_dim, num_actions=num_actions, num_layers=num_layers, hidden_states=hidden_states)
            state = torch.load(model_path, map_location=torch.device('cpu'))
            try:
                model.load_state_dict(state["Q_state_dict"])
            except Exception as e:
                model.load_state_dict(state)
        
        self.model = model
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.bcq_threshold = bcq_threshold
        self.max_inr = max_inr
        self.min_inr = min_inr
        self.df = None
        self.device = device
        self.name = model_name if model_name is not None else f"State Dim: {state_dim}"
        
    @staticmethod
    def load_model(model_name=None, suffix=None, iteration=None, lr=None, folder="./warfarin_rl/discrete_BCQ/models/", num_actions=5, state_dim=20, threshold=0.2, hidden_states=25, events_batch_size=None, seed=None):
        
        assert bool(model_name is not None) ^ bool(suffix is not None), "Only ONE of model_name or suffix can be None."    
        if model_name is None:
            model_name = f"actions_{num_actions}_state_{state_dim}_{suffix}"
        model_path = Model.get_model_path(model_name, folder=folder, lr=lr, iteration=iteration, threshold=threshold, hidden_states=hidden_states, events_batch_size=events_batch_size, seed=seed)
        model = Model(model_path, num_actions=num_actions, state_dim=state_dim, bcq_threshold=threshold, hidden_states=hidden_states)
        return model
        
        
    @staticmethod
    def get_model_path(model_name, iteration=None, lr=None, folder="./warfarin_rl/discrete_BCQ/models/", threshold=0.2, hidden_states=25, events_batch_size=None, seed=None):
        if iteration is None:
            if events_batch_size is not None and seed is not None:
                mypath = folder + model_name + f"/lr{lr}_bcq{threshold}_hstates_{hidden_states}_evBatchSize{events_batch_size}_seed_{seed}/"
            else:
                mypath = folder + model_name + f"/lr_{lr}_bcq_{threshold}_hidden_states_{hidden_states}/"
            files = [f for f in listdir(mypath) if isfile(join(mypath,f))] 
            if len(files): 
                file_name = files[-1]
                print(files)
                print(f"Found {len(files)} networks for this model. Selecting the latest version... {file_name}")
            else:
                print(f"ERROR: Could not find files for model name: {model_name}") 
        else:
            if events_batch_size is not None and seed is not None:
                file_name = f"lr_{lr}_bcq_{threshold}_hidden_states_{hidden_states}_events_batchsize_{events_batch_size}_seed_{seed}/{model_name}_QN_{iteration}"
            else:
                file_name = f"lr_{lr}_bcq_{threshold}_hidden_states_{hidden_states}/{model_name}_QN_{iteration}"
        
        return folder + f"{model_name}/" + file_name

         
    def get_model_actions(self, sample_state, return_prob=False, use_threshold=True, inr_val=None):
        """Returns the actions corresponding to each entry in the given sample_state.
        
        Uses the model specified in self.model to generate the actions. 
        
        The supported models: 
            "threshold":
            "naive":
            "random":
            Q-network: 
        
        Returns: 
            A np.array of actions corresponding to each transition in the sample_state, of dimension [num transitions, 1]. 
                The actions are integers.
        """
        
        if self.model == "threshold":
            if inr_val is None:
                inr_val = sample_state[0]
            model = ThresholdModel()
            if inr_val.max() <= 1:
                inr_val_adj = inr_val * 4 + 0.5
            else:
                inr_val_adj = inr_val

            df = pd.DataFrame({})
            df['INR_VALUE'] = inr_val_adj.flatten()

            pred_df = model.predict_dosage_thresholds(df, colname="INR_VALUE", return_dose=False)
            rec_dosages = pred_df["REC_DOSAGE_MULT"]

            # rec_dosages = rec_dosages[rec_dosages > 0]
            if self.num_actions == 5:
                policy = rec_dosages.apply(lambda x: model.threshold_action_map[x])
            elif self.num_actions == 7:
                policy = rec_dosages.apply(lambda x: model.threshold_action_map_7[x])
            actions = policy.values
            policy = policy.fillna(self.num_actions)
            prob = F.one_hot(torch.LongTensor(policy.values)).reshape((sample_state.shape[0], sample_state.shape[1], self.num_actions + 1))[:, :, :-1].numpy()

        elif self.model == "naive":
            middle_action = math.floor(num_actions / 2)
            prob = torch.zeros(sample_state.shape[0], sample_state.shape[1], self.num_actions)
            prob[:, :, middle_action] = 1
            actions = np.ones(sample_state.shape[0]) * middle_action
        elif self.model == "random":
            np.random.seed(123)
            prob = torch.ones(sample_state.shape[0], sample_state.shape[1], self.num_actions) / self.num_actions
            actions = np.random.choice(self.num_actions, len(sample_state))
        else:
            q, imt, i = self.model(torch.FloatTensor(sample_state).to(self.device))
            imt = imt.exp()
            if use_threshold:    
                imt = (imt / imt.max(1, keepdim=True)[0] >= self.bcq_threshold).float() 
            else:
                imt = (imt / imt.max(1, keepdim=True)[0] >= 0).float() 
            prob = np.array((imt * q + (1. - imt) * -1e8).detach().to("cpu"))
            actions = np.array([[x] for x in np.array((imt * q + (1. - imt) * -1e8).argmax(1).to("cpu"))])
            
        if return_prob:
            return prob
        else:
            return actions

    def get_model_results(self, sample_state, num_bins=5, num_actions=5, state_method=3, is_ais=False, obs_state=None):
        """Generates a dataframe containing the actions for each row (transition). 

        Determines the actions for each of the df entries (each entry is a transition), based on the specified action space. Stores the resulting dataframe as self.df. Uses the self.get_model_actions() method to determine the actions.

        Args:
            df: Dataframe of transitions.
            colname: The column which is used to calculate the action. 
                When action_space == 'percent', colname is the Warfarin dose as a multiple of the previous dose.
                When action_space == 'absolute', colname is the absolute Warfarin dose prescribed at that time.
            num_actions: When action_space == 'percent', 
                num_actions specifies the number of actions in the action space.
            action_space: One of ['percent', 'absolute'].

        Returns:
            None. The dataframe is stored under self.df. 
        
        """
        if num_actions not in [3,5,7]:
            raise ValueError(f"Num actions provided should be one of: [3, 5, 7]. You provided: {num_actions}")
        if num_bins not in [3,5]:
            raise ValueError(f"Num INR bins provided should be one of: [3, 5]. You provided: {num_bins}")
        
        actions = self.get_model_actions(sample_state)
        
        if not is_ais:
            state_cols = Constants.state_cols_mapping[state_method]
            model_results = pd.DataFrame(sample_state, columns=state_cols)
        else:
            if obs_state is None:
                raise ValueError(f"This is an AIS state. obs_state is required but is missing.")
            model_results = pd.DataFrame({'INR_VALUE': obs_state[:,0]})
            
        model_results["ACTION"] = actions.transpose()[0]
        model_results["INR_VALUE"] = (model_results["INR_VALUE"] * (self.max_inr - self.min_inr)) + self.min_inr
        model_results = model_results.sort_values(by="ACTION")
        model_results["INR_BIN"] = Model.bin_inr(model_results, num_bins)
        
        if num_actions == 3:
            model_results["ACTION_NAME"] = model_results["ACTION"].apply(lambda x: Constants.action_map[x])
        elif num_actions == 5:
            model_results["ACTION_NAME"] = model_results["ACTION"].apply(lambda x: Constants.action_map_5[x])
        elif num_actions == 7:
            model_results["ACTION_NAME"] = model_results["ACTION"].apply(lambda x: Constants.action_map_7[x])
            
        self.df = model_results
        
        
    @staticmethod
    def bin_inr(df, num_bins=3, colname="INR_VALUE"):

        if num_bins == 3:
            cut_labels = ['<2', '[2,3]', '>2']
            cut_bins = [0, 1.999, 3.00, 10]
        elif num_bins == 5:
            cut_labels = ['<=1.5', '(1.5, 2)', '[2, 3]', '(3, 3.5)', '>=3.5']
            cut_bins = [0, 1.5, 1.999, 3, 3.499, 10]
        else:
            raise ValueError(f"Did not understand number of bins: {num_bins}")
            return None

        return pd.cut(df[colname], bins=cut_bins, labels=cut_labels)   
        
     

    @staticmethod
    def eval_good_actions(state, pred_action):
        device = "cpu"
        inr_state = state

        high_inr = sum(np.where(np.logical_and(inr_state > 0.625, pred_action == 0), 1, 0))
        in_range_inr = sum(
            np.where(np.logical_and(np.logical_and(inr_state >= 0.375, inr_state <= 0.625), pred_action == 1), 1, 0))
        low_inr = sum(np.where(np.logical_and(inr_state < 0.375, pred_action == 2), 1, 0))

        num_good_actions = high_inr + in_range_inr + low_inr

        perc_good_actions = num_good_actions / len(pred_action)
        print(f"Percent good actions: {perc_good_actions:,.2%}")

        return perc_good_actions
    
#     @staticmethod
#     def get_network_values(buffer_data, state_method, model_d=None, model_r=None, delta_r=0.75, delta_d=-0.25, delta_ry=0.85, delta_dy=-0.15):
        
#         temp_state = ReplayBufferDed.get_state(buffer_data, method=state_method)
#         if model_d is not None: 
#             q, imt, i = model_d.model(torch.FloatTensor(temp_state))
#             imt = imt.exp()
#             imt = (imt / imt.max(1, keepdim=True)[0] > model_r.bcq_threshold).float()
#             values_d = np.array([[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
#             medians_d = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])        
#             buffer_data["Q_D_Median"] = Model.clip_values(medians_d, Constants.clip_d)
#             buffer_data["Q_D_Max"] = Model.clip_values(values_d, Constants.clip_d)

#         if model_r is not None: 
#             q, imt, i = model_r.model(torch.FloatTensor(temp_state))
#             imt = imt.exp()
#             imt = (imt / imt.max(1, keepdim=True)[0] > model_r.bcq_threshold).float()
#             values_r = np.array([[x] for x in torch.max(imt * q + (1. - imt) * -1e8, 1)[0].to("cpu").detach().numpy()])
#             medians_r = np.array([[x] for x in torch.median(q, 1)[0].to("cpu").detach().numpy()])
#             buffer_data["Q_R_Median"] = Model.clip_values(medians_r, Constants.clip_r)
#             buffer_data["Q_R_Max"] = Model.clip_values(values_r, Constants.clip_r)

#         if model_d is not None and model_r is not None:
#             flag = np.where(np.logical_or(medians_r <= delta_r, medians_d <= delta_d), 1, 0)
#             yellow_flag = np.where(np.logical_or(medians_r <= delta_ry, medians_d <= delta_dy), 1, 0)
#             buffer_data["Red_Flag"] = flag
#             buffer_data["Yellow_Flag"] = yellow_flag

#         return buffer_data


    @staticmethod
    def get_regret(inr_state, pred_action, sample_state, model=None):

        df = pd.DataFrame({"STATE": inr_state, "ACTION": pred_action})
        df.loc[:, "EXP_ACTION"] = np.where(df["STATE"] > 0.625, 0, np.where(df["STATE"] < 0.375, 2, 1))

        wrong_df = df[df["ACTION"] != df["EXP_ACTION"]]

        if model is None:
            wrong_df.loc[:, "PROB_DIFF"] = 1 / 3
        else:
            # Prob of expected action
            inds = np.array([[x] for x in wrong_df["EXP_ACTION"].values])
            prob = model.get_model_actions(sample_state, return_prob=True)
            prob_for_expected_actions = (prob[np.array(wrong_df.index)])[np.arange(len(wrong_df))[:, None], inds]
            wrong_df.loc[:, "PROB_ACTION"] = prob_for_expected_actions.transpose()[0]

            # Prob of taken action
            inds = np.array([[x] for x in wrong_df["ACTION"].values])
            prob = model.get_model_actions(sample_state, return_prob=True)
            prob_for_expected_actions = (prob[np.array(wrong_df.index)])[np.arange(len(wrong_df))[:, None], inds]
            wrong_df.loc[:, "PROB_ACTION_ACTUAL"] = prob_for_expected_actions.transpose()[0]

            display(wrong_df[wrong_df["PROB_ACTION_ACTUAL"] < wrong_df["PROB_ACTION"]])
            wrong_df = wrong_df[wrong_df["PROB_ACTION"] > -1e6]
            wrong_df.loc[:, "PROB_DIFF"] = (wrong_df["PROB_ACTION_ACTUAL"] - wrong_df["PROB_ACTION"])

        regret = wrong_df["PROB_DIFF"].sum()
        perc_regret = regret / len(inr_state)

        print(f"Regret: {regret:,.2f}  |  Wrong actions: {len(wrong_df):,.0f} \n" +
              f"Percent Regret (Normalized by total entries): {perc_regret:,.2%} \n"
              f"Percent Regret (Normalized by wrong actions): {regret / len(wrong_df):,.2%}")

        return regret, perc_regret
    
    @staticmethod
    def clip_values(arr, clip_dict):
        return np.maximum(clip_dict['min'], np.minimum(clip_dict['max'], arr))
   