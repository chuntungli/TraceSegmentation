


import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout

pd.options.display.max_colwidth = 200

def get_dataframe_from_jsontrace(trace_path):
    start_time = time.time()
    print(start_time)

    data = []
    with open(trace_path, 'rb') as f:
        for line in f:
            # print(str(json_lines.index(line)) + '-th line of ' + str(len(json_lines)) + ' has been processed!')
            try:
                data.append(json.loads(line))
            except:
                continue
                # following codes will not be executed because of the existence of too many problematic lines in traces
                # some items in original traces are not properly contained by quotation marks
                # so far we identified "-inf", "inf", and "nan"
                if '-inf' in line:
                    # print('-inf')
                    line = line.replace('-inf', '"-inf"')
                    data.append(json.loads(line))
                elif 'inf' in line:
                    # print('inf')
                    line = line.replace('inf', '"inf"')
                    data.append(json.loads(line))
                elif 'nan' in line:
                    # print('nan')
                    line = line.replace('nan', '"nan"')
                    data.append(json.loads(line))


    print(str(np.round(time.time() - start_time,2)) + ' seconds consumed for parsing trace: ' + trace_url)

    # First check: whether traces of the original file equals to the sum of traces of TEN events
    original_trace_df = pd.DataFrame(data=data)

    if check_traces_euqal(original_trace_df):
        print('First check passed: traces of the original file equals to the sum of traces of TEN events!')
    else:
        raise ValueError('First check failed!')

    # Second check: whether traces of threadID 1 segment equals to the sum of traces of TEN events in the same segment
    thread_one_df = original_trace_df[original_trace_df['threadID'] == 1]

    if check_traces_euqal(thread_one_df):
        print('Second check passed: traces of thread one equals to the sum of traces of TEN events!')
    else:
        raise ValueError('First check failed!')

    # the following two lines make the thread_one_df drop lines in which the "event" value is 5
    # (i.e. "method" value is "null"). So far. it seems that the removal operation will not twist the call relations
    thread_one_df = thread_one_df[thread_one_df['event'] != 5]

    # replace items with different formats as those in the same format
    thread_one_df = thread_one_df.replace(np.nan, 'N/A', regex=True)
    thread_one_df = thread_one_df.replace('nan', 'N/A', regex=True)

    event_eight_df = original_trace_df[original_trace_df['event'] == 8]
    event_nine_df = original_trace_df[original_trace_df['event'] == 9]

    return thread_one_df, event_eight_df, event_nine_df


def check_traces_euqal(my_trace_df):
    check_flag = False

    event_zero_df = my_trace_df[my_trace_df['event'] == 0]
    event_one_df = my_trace_df[my_trace_df['event'] == 1]
    event_two_df = my_trace_df[my_trace_df['event'] == 2]
    event_three_df = my_trace_df[my_trace_df['event'] == 3]
    event_four_df = my_trace_df[my_trace_df['event'] == 4]
    event_five_df = my_trace_df[my_trace_df['event'] == 5]
    event_six_df = my_trace_df[my_trace_df['event'] == 6]
    event_seven_df = my_trace_df[my_trace_df['event'] == 7]
    event_eight_df = my_trace_df[my_trace_df['event'] == 8]
    event_nine_df = my_trace_df[my_trace_df['event'] == 9]

    if event_six_df.shape[0] > 0 or event_seven_df.shape[0] > 0:
        raise ValueError('Found that event 6 or event 7 traces without any processing logic!')

    all_events_traces = event_zero_df.shape[0] + event_one_df.shape[0] + event_two_df.shape[0] + event_three_df.shape[0] \
                        + event_four_df.shape[0] + event_five_df.shape[0] + event_six_df.shape[0] + event_seven_df.shape[0] \
                        + event_eight_df.shape[0] + event_nine_df.shape[0]

    if my_trace_df.shape[0] == all_events_traces:
        check_flag = True

    return check_flag

def getMethodName(method_name_df, id):
    return method_name_df.method[method_name_df.retVal == id].to_string().split('    ')[-1]
    # return method_name_df.method[method_name_df.retVal == id].to_string()

if __name__ == '__main__':
    archive_url = 'data'
    app_names = os.listdir(archive_url)
    for app in app_names:
        if app.startswith('.'):
            continue

        trace_names = os.listdir('%s/%s' % (archive_url,app))

        method_names = []
        for trace in trace_names:
            if not trace.endswith('.trace'):
                continue
            trace_url = '%s/%s/%s' % (archive_url, app, trace)
            thread_one_df, event_eight_df, event_nine_df = get_dataframe_from_jsontrace(trace_url)

            start_method_df = thread_one_df[thread_one_df.event == 0]
            start_method_df.index = np.arange(1, len(start_method_df)+1)
            end_method_df = thread_one_df[(thread_one_df.event == 1) | (thread_one_df.event == 2)]
            end_method_df.index = np.arange(1, len(end_method_df)+1)

            # Find unique method names and construct method name list
            methods = list(np.unique(start_method_df.method)) + list(np.unique(start_method_df.pre1))
            try:
                methods.remove('N/A')
            except:
                pass

            for method in methods:
                method_name = getMethodName(event_eight_df, method)
                try:
                    method_names.index(method_name)
                except:
                    method_names.append(method_name)

            # trace = trace.split('_')[1]

            '''
            =============================================================
                                Dynamic Call Tree
            =============================================================
            '''
            G = nx.DiGraph()
            # Generate Tree
            alive_methods = []
            next_end = None
            node_dict = {'retVals':[], 'nameIds':[]}
            i = j = 0
            while j < end_method_df.shape[0]:
                # Read Start Method
                if i < start_method_df.shape[0]:
                    method = start_method_df.method.iloc[i]
                    callee = start_method_df.pre1.iloc[i]
                    i += 1
                    method_name = getMethodName(event_eight_df, method)
                    method_idx = method_names.index(method_name)

                    # Add Node
                    node_idx = len(G.nodes)
                    G.add_node(node_idx)
                    node_dict['retVals'].append(method)
                    node_dict['nameIds'].append(method_idx)
                    alive_methods.append({'node_idx': node_idx, 'retVal': method})

                    # Add Edge
                    if callee == 'N/A':
                        continue
                    found_alive = False
                    for k in np.arange(len(alive_methods)-1, -1, -1):
                        if alive_methods[k]['retVal'] == callee:
                            found_alive = True
                            G.add_edge(alive_methods[k]['node_idx'], node_idx)
                            break
                    if not found_alive:
                        # Add Higher level methods Node
                        callee_name = getMethodName(event_eight_df, callee)
                        callee_idx = method_names.index(callee_name)
                        node_idx = len(G.nodes)
                        G.add_node(node_idx)
                        node_dict['retVals'].append(callee)
                        node_dict['nameIds'].append(callee_idx)
                        G.add_edge(node_idx, node_idx-1)

                # Check Node Alive
                while len(alive_methods) > 0:
                    if next_end == None:
                        next_end = end_method_df.method.iloc[j]
                        j += 1
                    for k in np.arange(len(alive_methods) - 1, -1, -1):
                        if alive_methods[k]['retVal'] == next_end:
                            alive_methods.remove(alive_methods[k])
                            next_end = None
                            break
                    if next_end:
                        break

            labels = {i: node_dict['nameIds'][i] for i in range(0, len(node_dict['nameIds']))}

            # Identify preliminary phases
            label_values = np.array(list(labels.values()))
            components = [set(label_values[list(c)]) for c in nx.weakly_connected_components(G)]
            # Write with pickle
            folder = 'components/%s' % app
            if not os.path.exists(folder):
                os.makedirs(folder)
            pickle.dump(components, open('%s/%s.p' % (folder, trace), "wb"))
            # Read with pickle
            # components = pickle.load(open('components_%s_%s.p' % (app, trace), "rb"))
            #
            # pos = graphviz_layout(G, prog='dot')
            #
            # fig = plt.figure(figsize=(100, 5), dpi=30)
            # # nx.draw(graph, pos, **options)
            # nx.draw_networkx_edges(G, pos, edge_color='dimgray', width=1)
            # nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=200)
            # nx.draw_networkx_labels(G, pos, labels, font_size=8)
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
            # plt.title('%s %s' % (app, trace))
            # plt.tight_layout()
            # fig.savefig('plots/%s/%s_DCT.pdf' % (app, trace), dpi=30)
            # plt.show()
            # plt.close(fig)

            '''
            =============================================================
                                    Call Graph
            =============================================================
            '''
            # G = nx.DiGraph()
            # labels = {}
            #
            # # Add Vertices
            # for method_name in method_names:
            #     method_idx = method_names.index(method_name)
            #     G.add_node(method_idx)
            #     # Calculate Occurence
            #     retVals = list(event_eight_df.retVal[event_eight_df.method == method_name])
            #     occurence = len(start_method_df[start_method_df.method.isin(retVals)])
            #     if occurence == 0:
            #         occurence = len(start_method_df[start_method_df.pre1.isin(retVals)])
            #     labels[method_idx] = '%d:%d' % (method_idx, occurence)
            # # Add Weighted Edges
            # pairs = start_method_df[start_method_df.pre1 != 'N/A'].groupby(['method','pre1']).size().reset_index().rename(columns={0:'weight'})
            # for idx, row in pairs.iterrows():
            #     caller = method_names.index(getMethodName(event_eight_df, row.pre1))
            #     callee = method_names.index(getMethodName(event_eight_df, row.method))
            #     if G.has_edge(caller, callee):
            #         G[caller][callee]['weight'] += row.weight
            #     else:
            #         G.add_edge(caller, callee, weight=row.weight)
            #
            # pos = graphviz_layout(G, prog="dot")
            #
            # fig = plt.figure(figsize=(40,6), dpi=80)
            # # nx.draw(graph, pos, **options)
            # nx.draw_networkx_edges(G, pos, edge_color='dimgray', width=1)
            # nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=250)
            # nx.draw_networkx_labels(G, pos, labels, font_size=5)
            # # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G,'weight'))
            # plt.title('%s %s' % (app, trace))
            # plt.show()
            # fig.savefig('plots/%s/%s_CG.pdf' % (app, trace), dpi=80)
            # plt.close(fig)

        folder = 'method list/%s' % app
        if not os.path.exists(folder):
            os.makedirs(folder)
        method_names_df = pd.DataFrame(method_names)
        method_names_df.to_csv('%s/%s.csv' % (folder, trace), index=True, header=False)

# archive_url = os.getcwd() + '/data'
# category_names = os.listdir(archive_url)
# category = category_names[3]
# app_names = os.listdir(archive_url + '/' + category)
# app = app_names[0]
# trace_names = os.listdir(archive_url + '/' + category + '/' + app)
# trace = trace_names[1]
#
#
# method_name = method_names[61]
# retVal = event_eight_df[event_eight_df.method == method_name].retVal.to_string().split('    ')[-1]
# traces = thread_one_df.iloc[np.where(thread_one_df.method == retVal)]
#
# event_eight_df[event_eight_df.method == '@0x75e15e70']