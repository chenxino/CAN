import json
from tqdm import tqdm
from lib.eval_methods import *
from utils import *


class Predictor:
    """
    :param model: model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt", batch_size=256, run_mode=None, topk=2):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = batch_size
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        self.run_mode = run_mode
        self.topk = topk

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                y_hat, _, _, _ = self.model(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon, _, _ = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]
            recons = recons[:, self.target_dims]
            preds = preds[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual) # (N-w_s) * k
        df = pd.DataFrame()
        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            df[f"Recon_{i}"] = recons[:, i]
            df[f"True_{i}"] = actual[:, i]
            # a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt((recons[:, i] - actual[:, i]) ** 2)
            if self.run_mode =="all":
                a_score = (1-self.gamma)*np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt((recons[:, i] - actual[:, i]) ** 2)
            else:
                a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) if self.run_mode == 'fore' else np.sqrt((recons[:, i] - actual[:, i]) ** 2)

            # a_score = np.sqrt((recons[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                epsilon=1e-2
                # epsilon=1
                a_score = (a_score - median) / (epsilon+iqr)

            anomaly_scores[:, i] = a_score
            df[f"A_Score_{i}"] = a_score

        # add the topk process from GDN
        total_err_scores = anomaly_scores.T
        total_features = total_err_scores.shape[0]
        topk = self.topk
        topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
        total_topk_err_scores = np.mean(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
        anomaly_scores = total_topk_err_scores
        
        # anomaly_scores = np.mean(anomaly_scores, 1)
        # anomaly_scores = np.max(anomaly_scores, 1)
        
        df['A_Score_Global'] = anomaly_scores

        return df

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True, scale_scores=False, value=None):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            # 根据模型，计算数据的各个属性的预测预测和重构误差
            train_pred_df = self.get_score(train)
            test_pred_df = self.get_score(test)

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values
            # 针对MSL，SMAP数据集，调整再拼接起来的数据边缘的异常分数为0，让由每个段拼起来的数据再每个段里做标准化。
            train_anomaly_scores = adjust_anomaly_scores(train_anomaly_scores, self.dataset, True, self.window_size)
            test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, False, self.window_size)

            # Update df
            train_pred_df['A_Score_Global'] = train_anomaly_scores
            test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            # 指数权重滑窗算法，平滑一下
            # smoothing_window = int(self.batch_size * self.window_size * 0.05)
            # smoothing_window = 2
            # train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            # test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

            smoothed_train_anomaly_scores = np.zeros(train_anomaly_scores.shape)
            before_num = 2
            for i in range(before_num, len(train_anomaly_scores)):
                smoothed_train_anomaly_scores[i] = np.mean(train_anomaly_scores[i-before_num:i+1])

            smoothed_test_anomaly_scores = np.zeros(test_anomaly_scores.shape)
            for i in range(before_num, len(test_anomaly_scores)):
                smoothed_test_anomaly_scores[i] = np.mean(test_anomaly_scores[i-before_num:i+1])
            test_anomaly_scores = smoothed_test_anomaly_scores 
            train_anomaly_scores = smoothed_train_anomaly_scores




        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        all_train_scores = np.zeros((len(train_pred_df), out_dim))
        all_test_scores = np.zeros((len(test_pred_df), out_dim))
        if value is not None:
            value_pred_df = self.get_score(value)
            all_value_scores = np.zeros((len(value_pred_df), out_dim))
            for i in range(out_dim):
                value_feature_anom_scores = value_pred_df[f"A_Score_{i}"].values
                all_value_scores[:, i] = value_feature_anom_scores
        
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            all_train_scores[:, i] = train_feature_anom_scores
            all_test_scores[:, i] = test_feature_anom_scores

            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)
            
            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features
        # These predictions are used to evaluate performance, as true anomalies are labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and peaks-over-treshold
        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)
        # p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
        #                   q=self.q, level=self.level, dynamic=self.dynamic_pot)
        p_eval = None
        if true_anomalies is not None:
            bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=200, step_num=1000, verbose=False)
            bf_eval_nopa = bf_search_nopa(test_anomaly_scores, true_anomalies, start=0.01, end=200, step_num=1000, verbose=False)
            # get_best_f1(test_anomaly_scores, true_anomalies)
        else:
            bf_eval = {}
        # from GDN
        top1_best_info = get_best_performance_data(all_test_scores, true_anomalies, topk=15) 
        print(f"GDN Results using best GDN method:\n {top1_best_info}")

        if value is None:
            all_value_scores = all_train_scores
        top1_val_info = get_val_performance_data(all_test_scores, all_value_scores, true_anomalies, topk=15)
        print(f"GDN Results using val GDN method:\n {top1_val_info}")

        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")
        print(f"Results using best f1 score search without PA(point-adjust):\n {bf_eval_nopa}")


        for k, v in e_eval.items():
            if not type(e_eval[k]) == list:
                e_eval[k] = float(v)
        # for k, v in p_eval.items():
        #     if not type(p_eval[k]) == list:
        #         p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)
        for k, v in bf_eval_nopa.items():
            bf_eval_nopa[k] = float(v)

        # Save
        summary = {"GDN_best": top1_best_info, "GDN_val": top1_val_info, "epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval, "bf_eval_nopa": bf_eval_nopa}
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
        if save_output:
            global_epsilon = e_eval["threshold"]
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df[f"A_Pred_Global"] = (train_anomaly_scores >= global_epsilon).astype(int)
            test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(None, true_anomalies, global_epsilon, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")
