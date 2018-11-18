from numpy.random import seed
seed(6)
import tensorflow as tf
tf.set_random_seed(1)
import numpy as np
import sys

from preprocess import Word2Vec, MSRP, WikiQA, Med
from utils import build_path
# from sklearn import linear_model, svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.externals import joblib
from mix_model import MixModel
import xgboost as xgb

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction=0.7
# sess = tf.Session(config=config)


def train(lr, epoch, batch_size, data_type, word2vec, model_type):
    if data_type == "WikiQA":
        train_data = WikiQA(word2vec=word2vec)
    elif data_type == 'Med':
        train_data = Med(word2vec=word2vec)
    else:
        train_data = MSRP(word2vec=word2vec)

    test_data = Med(word2vec=word2vec)
    train_data.open_file(mode="train")
    test_data.open_file(mode='test')

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

    model = MixModel(train_data.max_len, lr, model_type, train_data.num_features)


    # Due to GTX 970 memory issues
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    # saver = tf.train.Saver(max_to_keep=100)

    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        # train_summary_writer = tf.summary.FileWriter("C:/tf_logs/train", sess.graph)

        sess.run(init)

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")
            train_data.shuffle_data()
            train_data.reset_index()
            i = 0

            # LR = linear_model.LogisticRegression()
            rfr_model = ensemble.RandomForestRegressor(n_estimators=20)
            ada = ensemble.AdaBoostRegressor(n_estimators=50)
            gbrt = ensemble.GradientBoostingRegressor(n_estimators=50)
            xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3, 
            min_child_weight=1.7817, n_estimators=50,
            reg_alpha=0.4640, reg_lambda=0.8571,
            subsample=0.5213, silent=1,
            random_state =7, nthread = -1)
            clf_features = []
            all_features = []
            test_all_features = []
            while train_data.is_available():
                i += 1

                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size)

                # print(batch_x1.shape)
                # print(batch_x2.shape)
                # print(batch_y)
                # print(batch_features)
                _, c, predictions, output_features = sess.run([model.optimizer, model.mse_loss, model.predictions, model.output_features],
                                                  feed_dict={model.embedded_x1: batch_x1,
                                                             model.embedded_x2: batch_x2,
                                                             model.labels: batch_y,
                                                             model.features: batch_features,
                                                             })
                
                # print('predictions:', predictions)
                pearson = np.corrcoef(np.squeeze(predictions), np.squeeze(batch_y))[0][1]

                all_features.append(output_features)
                # clf_features.apppend()

                if i % 1 == 0:
                    print("[batch " + str(i) + "] cost:", c, 'training pearson:', pearson)
            # train_x1, train_x2, train_y, train_feature, train_ml_features = train_data.get_data()
            # c, pre = sess.run([model.mse_loss, model.predictions], feed_dict={
            #     model.embedded_x1:train_x1,
            #     model.embedded_x2:train_x2,
            #     model.labels:train_y,
            #     model.features: train_feature,
            #     model.ml_features:train_ml_features
            #     })
            # pearson = np.corrcoef(np.squeeze(pre), np.squeeze(train_y))[0][1]
            # print("[epoch " + str(e) + "] cost:", c, 'training pearson:', pearson)
            
            test_x1, test_x2, test_y, test_feature = test_data.get_data()
            test, test_features = sess.run([model.predictions,model.output_features], feed_dict={
                model.embedded_x1:test_x1,
                model.embedded_x2:test_x2,
                model.labels:test_y,
                model.features:test_feature,
                })
            test_pearson = np.corrcoef(np.squeeze(test), np.squeeze(test_y))[0][1]
            print('epoch: ', e, 'pearson:', 'test_pearson: ', test_pearson)

            # result_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-test-result.pkl")
            # joblib.dump(test, result_path)
            # train_summary_writer.add_summary(merged, i)
            
            # test_all_features.append(test_features)

            # save_path = saver.save(sess, build_path("./models/", data_type, model_type, model.num_layers), global_step=e)
            # print("model saved as", save_path)
            all_features = np.concatenate(all_features)

            # test_all_features = np.concatenate(test_all_features)
            #train_data.labels
            #print('train.labels: ', train_data.labels.dtype)
            # ada.fit(clf_features, train_data.labels)
            gbrt.fit(all_features, train_data.labels)
            xgb_model.fit(all_features, train_data.labels)
            rfr_model.fit(all_features, train_data.labels)

            # all_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-all_features.pkl")
            # joblib.dump(all_features, all_path)
            # clf_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-clf.pkl")
            # joblib.dump(clf_features, clf_path)
            # test_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-test_features.pkl")
            # joblib.dump(test_all_features, test_path)

            print('============gbrt============')
            ml_pre_gbrt = gbrt.predict(test_features)
            test_pearson = np.corrcoef(np.squeeze(ml_pre_gbrt), np.squeeze(test_y))[0][1]
            print('epoch: ', e, 'pearson:', 'ml_test_pearson: ', test_pearson)

            # print('============ada============')
            # ml_pre_ada = ada.predict(test_features)
            # test_pearson = np.corrcoef(np.squeeze(ml_pre_ada), np.squeeze(test_y))[0][1]
            # print('epoch: ', e, 'pearson:', 'ml_test_pearson: ', test_pearson)

            print('============xgb============')
            ml_pre_xgb = xgb_model.predict(test_features)
            test_pearson = np.corrcoef(np.squeeze(ml_pre_xgb), np.squeeze(test_y))[0][1]
            print('epoch: ', e, 'pearson:', 'ml_test_pearson: ', test_pearson)

            print('============rfr============')
            ml_pre_rfr = rfr_model.predict(test_features)
            test_pearson = np.corrcoef(np.squeeze(ml_pre_rfr), np.squeeze(test_y))[0][1]
            print('epoch: ', e, 'pearson:', 'ml_test_pearson: ', test_pearson)

            ##对三个ml model取均值
            print('===============模型average结果==================')
            average_pred = (ml_pre_gbrt+ml_pre_xgb+ml_pre_rfr)/3.0
            test_pearson = np.corrcoef(np.squeeze(average_pred), np.squeeze(test_y))[0][1]
            print('epoch: ', e, 'pearson:', 'ml_test_pearson: ', test_pearson)
            
        print(test)
        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":

    # Paramters
    # --lr: learning rate
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --batch_size: batch size
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: MSRP or WikiQA data or Med data

    # default parameters
    params = {
        "model": "ABCNN+LSTM",
        "lr": 0.03,
        "epoch": 300,
        "batch_size": 50,
        "model_type": "ABCNN1",
        "data_type": "Med",
        "word2vec": Word2Vec()
    }

    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])


    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    train(lr=float(params["lr"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]),data_type=params["data_type"], word2vec=params["word2vec"], model_type=params["model_type"])
