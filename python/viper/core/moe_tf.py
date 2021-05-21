# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:32:26 2019

@author: Andri
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import sys
import os
from sklearn.tree import DecisionTreeRegressor as DTC

warnings.filterwarnings('ignore')


class MoE_config:

    def __init__(self, no_x1=4, no_out=1, no_experts=4,
                 init_learning_rate=0.5, learning_rate_decay=0.99,
                 max_epochs=100, init_epoch=5):
        self.input_size_x = no_x1
        self.output_size = no_out
        self.experts_no = no_experts
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_epochs = max_epochs
        self.init_epoch = init_epoch

    def build_graph(self):

        tf.reset_default_graph()
        MoE_graph = tf.Graph()

        with MoE_graph.as_default():
            learning_rate = tf.placeholder(tf.float32, None,
                                           name="learning_rate")
            inputs = tf.placeholder(tf.float32, [None, self.input_size_x],
                                    name="inputs")
            targets = tf.placeholder(tf.float32, [None, self.output_size],
                                     name="targets")
            experts = tf.placeholder(tf.float32, [None, self.experts_no],
                                     name="experts")

            weights = tf.Variable(
                tf.random.uniform([self.input_size_x, self.no_experts]))
            biases = tf.Variable(tf.zeros([self.experts_no]))

            with tf.name_scope("output_layer"):
                logits = tf.add(tf.matmul(inputs, weights), biases,
                                name="logits")
                gate = tf.nn.softmax(logits, name="gate")
                prediction = tf.reduce_sum(tf.multiply(gate, experts), axis=1,
                                           name="prediction")
                tf.summary.histogram("weights", weights)
                tf.summary.histogram("biases", biases)

            with tf.name_scope("train"):
                loss = tf.reduce_mean(tf.square(prediction - targets),
                                      name="loss_mse")
                optimizer = tf.train.AdamOptimizer(learning_rate)
                minimize = optimizer.minimize(loss,
                                              name="loss_mse_adam_minimize")
                tf.summary.scalar("loss_mse", loss)

        return MoE_graph

    def fit(MoE_graph, MODEL_DIR, self, x, y, max_depth):

        graph_name = "graph_1"

        def _compute_learning_rates(init_learning_rate, init_epoch):
            learning_rates_to_use = [init_learning_rate * (
                self.learning_rate_decay ** max(float(i + 1 - self.init_epoch),
                                                0.0)
            ) for i in range(self.max_epoch)]
            return learning_rates_to_use

        final_prediction = []
        final_loss = None
        learning_rates_to_use = compute_learning_rates(self.init_learning_rate,
                                                       self.learning_rate_decay,
                                                       self.init_epoch)

        with tf.Session(graph=MoE_graph) as sess:
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('_logs/' + graph_name, sess.graph)
            writer.add_graph(sess.graph)

            graph = tf.get_default_graph()
            tf.global_variables_initializer().run()

            inputs = graph.get_tensor_by_name('inputs:0')
            targets = graph.get_tensor_by_name('targets:0')
            experts = graph.get_tensor_by_name('experts:0')
            learning_rate = graph.get_tensor_by_name('learning_rate:0')
            loss = graph.get_tensor_by_name('train/loss_mse:0')
            minimize = graph.get_operation_by_name(
                'train/loss_mse_adam_minimize')
            gate = graph.get_tensor_by_name('output_layer/gate:0')
            prediction = graph.get_tensor_by_name('output_layer/prediction:0')

            for epoch_step in range(self.max_epoch):
                current_lr = learning_rates_to_use[epoch_step]

                for batch_X, batch_Y in batches(self.batch_size):

                    self.dtc_list = []
                    expert = np.zeros([batch_X.shape[0], self.experts_no])
                    train_data_gate = {inputs: batch_X}
                    gating = sess.run(gate, train_data_gate)

                    for j in range(self.experts_no):
                        self.dtc_list.append([])
                        self.dtc_list[j] = DTC(max_depth=max_depth)
                        self.dtc_list[j].fit(batch_X, batch_Y,
                                             sample_weight=gating[:, j].T)
                        expert[:, j] = self.dtc_list[j].predict(batch_X)

                    train_data_feed = {inputs: batch_X,
                                       targets: batch_Y,
                                       learning_rate: current_lr,
                                       experts: expert}

                    train_loss, _, _summary = sess.run(
                        [loss, minimize, merged_summary], train_data_feed)

                writer.add_summary(_summary, global_step=epoch_step)

            graph_saver_dir = os.path.join(MODEL_DIR, graph_name)
            if not os.path.exists(graph_saver_dir):
                os.mkdir(graph_saver_dir)

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(graph_saver_dir, graph_name),
                       global_step=epoch_step)

    def predict(self, X_test, Y_test):
