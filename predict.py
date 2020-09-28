import os

import numpy as np
import argparse
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_io


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def load_graph(graph_def):
    wide = tf.placeholder(shape=(1, 762), dtype='float', name='wide')
    workclass_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='workclass_inp')
    education_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='education_inp')
    marital_status_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='marital_status_inp')
    occupation_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='occupation_inp')
    relationship_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='relationship_inp')
    race_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='race_inp')
    gender_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='gender_inp')
    native_country_inp = tf.placeholder(shape=(1, 1), dtype='int32', name='native_country_inp')
    age_in = tf.placeholder(shape=(1, 1), dtype='float', name='age_in')
    capital_gain_in = tf.placeholder(shape=(1, 1), dtype='float', name='capital_gain_in')
    capital_loss_in = tf.placeholder(shape=(1, 1), dtype='float', name='capital_loss_in')
    hours_per_week_in = tf.placeholder(shape=(1, 1), dtype='float', name='hours_per_week_in')
    tf.import_graph_def(graph_def, name='',
                        input_map={"wide": wide,
                                   "workclass_inp": workclass_inp, "education_inp": education_inp,
                                   "marital_status_inp": marital_status_inp, "occupation_inp": occupation_inp,
                                   "relationship_inp": relationship_inp, "race_inp": race_inp, "gender_inp": gender_inp,
                                   "native_country_inp": native_country_inp,
                                   "age_in": age_in, "capital_gain_in": capital_gain_in,
                                   "capital_loss_in": capital_loss_in,
                                   "hours_per_week_in": hours_per_week_in})
    return


def load_frozen_model_v1(pb_path, prefix='', print_nodes=False):
    """Load frozen model (.pb file) for testing.
    After restoring the model, operators can be accessed by
    graph.get_tensor_by_name('<prefix>/<op_name>')
    Args:
        pb_path: the path of frozen model.
        prefix: prefix added to the operator name.
        print_nodes: whether to print node names.
    Returns:
        graph: tensorflow graph definition.
    """
    if os.path.exists(pb_path):
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name=prefix
            )
            if print_nodes:
                for op in graph.get_operations():
                    print(op.name)
            return graph
    else:
        print('Model file does not exist', pb_path)
        exit(-1)


def load_input():
    wide = np.loadtxt("data/wide.txt")
    wide = wide.astype(dtype=np.int32).reshape([1, len(wide)])
    deep = np.loadtxt("data/deep.txt")
    workclass_inp = deep[0].astype(dtype=np.int32).reshape([1, 1])
    education_inp = deep[1].astype(dtype=np.int32).reshape([1, 1])
    marital_status_inp = deep[2].astype(dtype=np.int32).reshape([1, 1])
    occupation_inp = deep[3].astype(dtype=np.int32).reshape([1, 1])
    relationship_inp = deep[4].astype(dtype=np.int32).reshape([1, 1])
    race_inp = deep[5].astype(dtype=np.int32).reshape([1, 1])
    gender_inp = deep[6].astype(dtype=np.int32).reshape([1, 1])
    native_country_inp = deep[7].astype(dtype=np.int32).reshape([1, 1])
    age_in = deep[8].astype(dtype=np.float).reshape([1, 1])
    capital_gain_in = deep[9].astype(dtype=np.float).reshape([1, 1])
    capital_loss_in = deep[10].astype(dtype=np.float).reshape([1, 1])
    hours_per_week_in = deep[11].astype(dtype=np.float).reshape([1, 1])
    return wide, workclass_inp, education_inp, marital_status_inp, occupation_inp, relationship_inp, race_inp, gender_inp, native_country_inp, age_in, capital_gain_in, capital_loss_in, hours_per_week_in


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb_file", type=str, default="model_1_13/saved_model.h5")
    args = vars(ap.parse_args())
    pb_file = args["pb_file"]

    # from tensorflow.python.framework import op_def_registry
    # print(tf.__version__)
    # print(op_def_registry.get_registered_ops().get('ResourceGather'))
    # exit()

    K.set_learning_phase(0)
    model = tf.keras.models.load_model(pb_file, compile=False)
    # model.summary()
    inp = model.input
    print(inp)
    output = model.output
    print(output)

    output = model.predict(load_input())  # p: 0.50864685
    print(output)
    print(tf.__version__)

    sess = K.get_session()
    kgraph = sess.graph

    frozen_graph = freeze_session(sess, output_names=[model.output.op.name])
    graph_io.write_graph(frozen_graph, 'model', 'tensor_model.pb', as_text=False)

    load_frozen_model_v1('model/tensor_model.pb', "", print_nodes=True)

    # frozen_op = []
    # for op in kgraph.get_operations():
    #     if ('ResourceGather' == op.type or 'RandomUniform' == op.type):
    #         frozen_op.append(op.name)
    # print(frozen_op)

    graph_def = kgraph.as_graph_def()
    load_graph(graph_def)
    wide_data, workclass_inp_data, education_inp_data, marital_status_inp_data, occupation_inp_data, relationship_inp_data, \
    race_inp_data, gender_inp_data, native_country_inp_data, age_in_data, capital_gain_in_data, \
    capital_loss_in_data, hours_per_week_in_data = load_input()

    predictions = sess.run(sess.graph.get_tensor_by_name('wide_deep/Sigmoid:0'), {
        'hours_per_week_in:0': hours_per_week_in_data,
        'capital_loss_in:0': capital_loss_in_data,
        'capital_gain_in:0': capital_gain_in_data,
        'age_in:0': age_in_data,
        'native_country_inp:0': native_country_inp_data,
        'gender_inp:0': gender_inp_data,
        'race_inp:0': race_inp_data,
        'relationship_inp:0': relationship_inp_data,
        'occupation_inp:0': occupation_inp_data,
        'marital_status_inp:0': marital_status_inp_data,
        'education_inp:0': education_inp_data,
        'workclass_inp:0': workclass_inp_data,
        'wide:0': wide_data})
    print(predictions)