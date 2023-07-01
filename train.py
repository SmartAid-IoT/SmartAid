from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn.linear_model as lm
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import pickle
import torch
import click

from data.cap_values import cap_values

import warnings
warnings.filterwarnings("ignore")


### PREPROCESS ###

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data,f)
        
def get_closest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def change_empty(log): ## replaces empty string to 0
    log.replace('', '0', inplace=True)
    return log

def group_cap(groups, cap): # groups capability in given groups
    count = 1
    while count != 0:
        count = 0
        if len(groups) == 1:
            break
        for i,j in itertools.combinations(groups, 2):
            if len(set(i).intersection(set(j))) > 0:
                count += 1
                if list(set(i+j)) not in groups:
                    groups.append(list(set(i+j)))
                    groups.remove(i)
                    groups.remove(j)
                    break
    # cap order match
    for i in range(len(groups)):
        new = []
        for c in cap:
            if (c,) in groups[i]:
                new.append(c)
        groups[i] = new
    return groups

def find_groups_by_change(log): # find group of capability in by change in log
    cap_groups = log['prev'].compare(log['next']).agg(lambda x: tuple(set([i[0] for i in list(x.dropna().index)])), axis=1).value_counts().keys()
    groups = []
    group_cap = []
    for cap in log.prev.columns:
        if (cap,) not in cap_groups:
            for g in cap_groups:
                if cap in g:
                    if cap in group_cap:
                        continue
                    groups.append(list(g))
                    group_cap += list(g)
                    continue
    for i in range(len(groups)):
        new = []
        for c in log.prev.columns:
            if c in groups[i]:
                new.append(c)
        groups[i] = new
    return groups

def set_groups(log, groups): # rename capabilities in log to set group
    for i in range(len(groups)):
        for c in range(len(groups[i])):
            if '+' not in groups[i][c]:
                log.rename(columns={groups[i][c]: 'h'+str(i)+'+'+groups[i][c]}, level=1, inplace=True, errors='ignore')
                groups[i][c] = 'h'+str(i)+'+'+groups[i][c]
    return log, groups

def get_capabilities(log, groups): # get available capabilites from log
    capabilites = log.prev.columns.values.tolist()
    for cap in groups:
        for c in cap:
            if c in capabilites:
                capabilites.remove(c)
    capabilites = [(c,) if type(c)!=list else tuple(c) for c in capabilites+groups]
    if tuple([]) in capabilites:
        capabilites.remove(())
    return capabilites

def get_log_with_target_change(c_log, target): # get log with targeted capability change
    def powerset(iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
    target = [i for i in powerset(target)]
    change = c_log['prev'].compare(c_log['next'])
    idx = []
    change = pd.DataFrame(change.agg(lambda x: tuple(set([i[0] for i in x.dropna().index])), axis=1), columns=['c']).groupby('c')
    for i in target:
        try:
            idx += list(change.get_group(i).index)
        except:
            continue
    for i in target:
        try:
            idx += list(change.get_group(i[::-1]).index)
        except:
            continue
    return c_log.loc[idx]

def get_log_with_single_change(log): # get log with only single capability change
    change2caps = log[log['prev'].compare(log['next'], keep_shape=True).agg(lambda x: list(x.dropna()), axis=1).str.len().gt(2)].index
    changeNoCaps = log[log['prev'].compare(log['next'], keep_shape=True).agg(lambda x: list(x.dropna()), axis=1).str.len().lt(1)].index
    return log.drop(np.concatenate([change2caps,changeNoCaps]))

def get_dep(device, log, encoders, capabilites): # find dependency from the given log
    dep = {}
    for cap in log['prev']:
        if cap in encoders.keys():
            if log['prev'][cap].nunique() != len(encoders[cap].classes_):
                dep[(cap,)] = [tuple(encoders[cap].inverse_transform([i])) for i in log['prev'][cap].unique()]
        else:
            if log['prev'][cap].nunique() == 1:
                dep[(cap,)] = [(log['prev'][cap].iloc[0],)]
    h = list(set([cap if len(cap)>1 else 1 for cap in capabilites]))
    if 1 in h:
        h.remove(1)
    h = {cap: [-1]*len(cap) for cap in h}
    # find dependency with hierarchy
    for cap in list(dep.keys()):
        if '+' in cap[0]:
            for i in h.keys():
                for j in range(len(i)):
                    if i[j] == cap[0]:
                        h[i][j] = dep[cap][0]
            dep.pop(cap)
    # set values for other leaf_cap in hierarchy cap
    for cap in list(h.keys()):
        if set(h[cap]) != set([-1]):
            for i in range(len(cap)):
                if h[cap][i] == -1:
                    h[cap][i] = cap_values[device][(cap[i].split('+')[-1],)][0]
        else:
            h.pop(cap) # if no depedency, drop
    dep.update(h)
    return dep

def encode(cmd_cap): # encode string values
    encoders = {}
    for cmd in cmd_cap:
        for cap in cmd_cap[cmd]['log']['prev']:
            if type(cmd_cap[cmd]['log']['prev'][cap].iloc[0]) in [np.bool_, list, bool]:
                    cmd_cap[cmd]['log'][('prev', cap)] = cmd_cap[cmd]['log'][('prev', cap)].apply(str)
            if cmd_cap[cmd]['log']['prev'][cap].iloc[0] != None:
                if cap not in encoders:
                    try:
                        cmd_cap[cmd]['log'][('prev', cap)] = cmd_cap[cmd]['log'][('prev', cap)].astype(int)
                    except:
                        encoder = LabelEncoder()
                        encoder.fit(cmd_cap[cmd]['log'][('prev', cap)])
                        cmd_cap[cmd]['log'][('prev', cap)] = encoder.transform(cmd_cap[cmd]['log'][('prev', cap)])
                        encoders[cap] = encoder
                else:
                    encoder = encoders[cap]
                    for label in np.unique(cmd_cap[cmd]['log'][('prev', cap)]):
                        if label not in encoder.classes_: # unseen label
                            encoder.classes_ = np.append(encoder.classes_, label) 
                    cmd_cap[cmd]['log'][('prev', cap)] = encoder.transform(cmd_cap[cmd]['log'][('prev', cap)])
                    encoders[cap] = encoder
        for cap in cmd_cap[cmd]['log']['next']:
            if type(cmd_cap[cmd]['log']['next'][cap].iloc[0]) in [np.bool_, list, bool]:
                    cmd_cap[cmd]['log'][('next', cap)] = cmd_cap[cmd]['log'][('next', cap)].apply(str)
            if cmd_cap[cmd]['log'][('next', cap)].iloc[0] != None:
                if cap not in encoders:
                    try:
                        cmd_cap[cmd]['log'][('next', cap)] = cmd_cap[cmd]['log'][('next', cap)].astype(int)
                    except:
                        None
                else:
                    encoder = encoders[cap]
                    for label in np.unique(cmd_cap[cmd]['log'][('next', cap)]):
                        if label not in encoder.classes_: 
                            encoder.classes_ = np.append(encoder.classes_, label) 
                    cmd_cap[cmd]['log'][('next', cap)] = encoder.transform(cmd_cap[cmd]['log'][('next', cap)])
                    encoders[cap] = encoder
    return cmd_cap, encoders

def divide_by_cmd_cap(device, log, capabilites): # devide logs of commands by changed capability values

    cmds = log.cmd.command.unique()
    grouped = log.groupby(('cmd','command'))
    cmd_logs = {c:grouped.get_group(c) for c in cmds}

    for cmd in cmds:
        change = cmd_logs[cmd]['prev'].compare(cmd_logs[cmd]['next'])

        if ('+' in ''.join(list(change.columns.levels[0]))):
            for c in change.columns.levels[0]:
                if '+' in c:
                    hi = c.split('+')[0]
                    break
            hi = (sum([[c] if (hi in c) else [] for c in cmd_logs[cmd]['prev'].columns], []))
            change_cap = set([(c,) if c not in hi else tuple(hi) for c in list(change.columns.levels[0])])
            cmd_logs.update({cmd+'>'+str(i): 
                             {'target':(c,) if type(c) == str else c,
                              'log':get_log_with_single_change(cmd_logs[cmd].loc[change[c[0]].dropna().index]) if type(c)==str
                              else get_log_with_target_change(cmd_logs[cmd], c)
                              } for i,c in enumerate(change_cap)})
            cmd_logs.pop(cmd)
        elif len(change.columns) > 2:
            cmd_logs.update({cmd+'>'+str(i):
                             {'target':(c,),
                              'log':get_log_with_single_change(cmd_logs[cmd].loc[change[c].dropna().index])}
                             for i,c in enumerate(change.columns.levels[0])})
            cmd_logs.pop(cmd)
        else:
            cmd_logs.update({cmd: {'target':(change.columns.levels[0][0],), 'log':cmd_logs[cmd]}})
    
    for cmd in list(cmd_logs.keys()):
        if len(cmd_logs[cmd]['log']) == 0:
            cmd_logs.pop(cmd)
            continue
        
    cmd_logs, encoders = encode(cmd_logs)

    for cmd in list(cmd_logs.keys()):
        cmd_logs[cmd]['dep'] = get_dep(device, cmd_logs[cmd]['log'], encoders, capabilites)
        
    return cmd_logs, encoders

def find_groups_by_dep(cmd_cap): # create group of capabilities considering the dependency
    groups = []
    pair = []
    cap = 0
    for i in cmd_cap:
        cap = list(cmd_cap[i]['log'].prev.columns)
        pair.append([list(i) for i in itertools.product((cmd_cap[i]['target'],), list(cmd_cap[i]['dep']))])
    pair = sum(pair, [])
    for p in pair:
        if p[::-1] in pair and p[::-1] not in groups:
            groups.append(p)
    groups = [[i[0]]+i[1] for i in pd.DataFrame(groups, columns=['1', '2']).groupby('1', as_index=False).agg(list).values.tolist()]
    change = 1
    while change == 1:
        change = 0
        if len(groups) == 1:
            break
        for i,j in itertools.combinations(groups, 2):
            if set(i) == set(j):
                groups.remove(i)
                change = 1
                break
            if len(set(i).intersection(set(j))) > 0:
                groups.append(set(i).union(set(j)))
                groups.remove(i)
                groups.remove(j)
                change = 1
                break
    drop = []
    # cap order match
    for i in range(len(groups)):
        new = []
        for c in cap:
            if (c,) in groups[i]:
                new.append(c)
        groups[i] = new
        if len(set(groups[i])) == 1:
            drop.append(groups[i])
    
    for i in drop:
        groups.remove(i)
    
    return groups

def check_cmds(cmd_cap, cmd_logs, capabilites): # check if the commands work as expected in when prerequisite condition is matched
    success_target = []
    success_cmd = []
    for cmd in cmd_cap:
        orginal_cmd = cmd.split('>')[0]
        filtered_log = cmd_logs[orginal_cmd]
        for cap in cmd_cap[cmd]['dep']:
            for i in range(len(cap)):
                groups = filtered_log['prev'].groupby([cap[i]])
                filtered_log = filtered_log.loc[sum([groups.get_group(v).index.values.tolist() if v in groups.groups.keys() else [] for v in cmd_cap[cmd]['dep'][cap][i]], [])]
        if len(filtered_log)==0:
            continue;
        change = list(filtered_log['prev'].compare(filtered_log['next']).dropna(axis=1, how='all').columns.levels[0])
        if list(cmd_cap[cmd]['target']) == change:
            success_cmd.append(cmd)
            success_target.append(tuple(cmd_cap[cmd]['target']))
        else:
            check = 1
            for i in change:
                if i not in list(cmd_cap[cmd]['target']):
                    check = 0
            if check == 1:
                success_cmd.append(cmd)
                success_target.append(tuple(cmd_cap[cmd]['target']))
    success_target = set(success_target)
    fail_cap = []
    for cap in capabilites:
        if cap not in success_target:
            fail_cap.append(cap)
    
    if len(fail_cap) > 0:
        print('Cannot handle:', fail_cap)
        exit
    print()
    print('AVAILABLE COMMANDS:')
    print('\n'.join([str(i+1)+'. '+c for i,c in enumerate(success_cmd)]))
    print()
    return success_cmd

def group_and_encode(device, path, cap_value): # process all grouping and encoding
    log = change_empty(pd.read_pickle(path))
        
    groups = find_groups_by_change(log)
    log, groups = set_groups(log, groups)
    capabilites = get_capabilities(log, groups)
    
    cmd_cap, encoders = divide_by_cmd_cap(device, log, capabilites) 
    groups+= find_groups_by_dep(cmd_cap)
    log, groups = set_groups(log, groups)
    capabilites = get_capabilities(log, groups)
    
    cmd_logs = get_cmd_logs(log)
    cmd_cap, encoders = divide_by_cmd_cap(device, log, capabilites)
    success_cmd = check_cmds(cmd_cap, cmd_logs, capabilites)
    
    cmd_cap = remove_failed_cmd(cmd_cap, success_cmd)
    encoders = rename_encoders(encoders)
    
    return cmd_cap, encoders


### TRAIN MODELS ###

class LogisticRegression(torch.nn.Module): # classification model 
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.num_class = output_dim
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
    def weighted_l1(self):
        return torch.sum(torch.abs(self.linear.weight[:, :-self.num_class])) #+ torch.sum(torch.abs(self.linear.bias))
    def l1(self):
        return torch.sum(torch.abs(self.linear.weight))# + torch.sum(torch.abs(self.linear.bias))
    
class LinearRegression(torch.nn.Module): # regression model
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    def weighted_l1(self):
        return torch.sum(torch.abs(self.linear.weight[:, :-1]))# + torch.sum(torch.abs(self.linear.bias))
    def l1(self):
        return torch.sum(torch.abs(self.linear.weight)) #+ torch.sum(torch.abs(self.linear.bias))
    
def fit_classification(x_train, x_test, y_train, y_test, num_class, argument): # fit classification model

    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_train = torch.nn.functional.one_hot(torch.LongTensor(y_train), num_classes=num_class).squeeze().type(torch.float)
    y_test = torch.nn.functional.one_hot(torch.LongTensor(y_test), num_classes=num_class).squeeze().type(torch.float)

    model = LogisticRegression(x_train.shape[1], num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
 
    epochs = 20*num_class
    Loss = []
    acc = []
    best = 1000000000000
    count = 0
    weight = model.linear.weight
    bias = model.linear.bias
    for epoch in tqdm(range(epochs)):
        for x, y in zip(x_train, y_train):
            optimizer.zero_grad()
            outputs = model(x)
            if argument:
                loss = criterion(outputs, y) + model.weighted_l1()*0.7
            else:
                loss = criterion(outputs, y) #+ model.l1()*0.5
            loss.backward()
            optimizer.step()
        Loss.append(loss.item())
        correct = 0
        for x, y in zip(x_test, y_test):
            outputs = model(x)
            predicted = torch.argmax(outputs.detach())
            correct += (predicted == y).sum()
        accuracy = 100 * (correct.item()) / len(y_test)
        acc.append(accuracy)
        if loss < best:
            best = loss
            weight = model.linear.weight.clone().detach()
            bias = model.linear.bias.clone().detach()
        else:
            count+=1
            if count > 5*num_class:
                break
    return weight, bias, np.mean(acc), model

def fit_regression(x_train, x_test, y_train, y_test, argument): # fit regression model

    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_train = torch.tensor(y_train).type(torch.float)
    y_test = torch.tensor(y_test).type(torch.float)

    model = LinearRegression(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
 
    epochs = 30
    Loss = []
    acc = []
    best = 1000000000
    count = 0
    weight = model.linear.weight
    bias = model.linear.bias
    for epoch in tqdm(range(epochs)):
        for x, y in zip(x_train, y_train):
            optimizer.zero_grad()
            outputs = model(x)
            if argument:
                loss = criterion(outputs, y) + model.weighted_l1()*0.7
            else:
                loss = criterion(outputs, y) 
            loss.backward()
            optimizer.step()
        Loss.append(loss.item())
        correct = 0
        for x, y in zip(x_test, y_test):
            outputs = model(x)
            predicted = np.round(outputs.detach())
            correct += (predicted == y).sum()
        accuracy = 100 * (correct.item()) / len(y_test)
        acc.append(accuracy)
        if loss < best:
            count = 0
            best = loss
            weight = model.linear.weight.clone().detach()
            bias = model.linear.bias.clone().detach()
        else:
            count+=1
            if count > 10:
                break
    return weight, bias, np.mean(acc), model

def arg_exists(log): #check if the command has argument
    if type(log['cmd']['arguments'].iloc[0]) == bool:
            return None
    else:
        return True
    
def product_dep(dep): # create all possible state in given condition
    for d in list(dep.keys()):
        if len(d) > 1:
            name = tuple(n.split('+')[-1] for n in d)
            dep[name] = [i for i in itertools.product(*(i for i in dep[d]))]
            dep.pop(d)
    return dep

def normalize_toOnehot(cmd, x, encoders, target, cap_value, normalize=False): # normalize numerical value and change ccategorical value to one hot vector
    new_x = []
    for cap in list(x.columns):
        if cap == 'arguments':
            try:
                x['arguments'] = x['arguments'].astype(int)
            except:
                None
            if type(x[cap].iloc[0]) != str:
                if normalize:
                    for t in target:
                        t = t.split('+')[-1]
                        if t not in encoders:
                            new_arg = pd.DataFrame(x[cap].apply(lambda e:(e-cap_value[(t,)][0][0])/(cap_value[(t,)][0][-1]-cap_value[(t,)][0][0]))).reset_index(drop=True)
                else:
                    new_arg = pd.DataFrame(x[cap].reset_index(drop=True))
            else:
                if cmd in encoders:
                    encoder = encoders[cmd]
                else:
                    encoder = LabelEncoder()
                    encoder.fit(x['arguments'])
                    encoders[cmd] = encoder
                encode_arg = pd.DataFrame(encoder.transform(x['arguments']), columns=['arg'])
                num = len(encoder.classes_)
                new_arg = pd.DataFrame((encode_arg['arg'].apply(lambda e: np.eye(num)[e]).values.tolist())).reset_index(drop=True)
            new_x.append(new_arg)
        elif cap.split('+')[-1] not in encoders.keys():
            if normalize:
                new_x.append(pd.DataFrame(x[cap].apply(lambda e:(e-cap_value[(cap.split('+')[-1],)][0][0])/(cap_value[(cap.split('+')[-1],)][0][-1]-cap_value[(cap.split('+')[-1],)][0][0]))).reset_index(drop=True))
            else:
                new_x.append(pd.DataFrame(x[cap].reset_index(drop=True)))
        else:
            num = len(encoders[cap.split('+')[-1]].classes_)
            new_x.append(pd.DataFrame(x[cap].apply(lambda e: np.eye(num)[e]).values.tolist()))
    return pd.concat(new_x, axis=1), encoders

def split_x_y(cmd_log, arguments): # split log to input and output

    log = cmd_log['log']
    target = list(cmd_log['target'])
    dependencies = cmd_log['dep']

    x = log[['prev', 'cmd']]
    x.columns = x.columns.droplevel()
    if arguments:
        x = pd.DataFrame(x[target+ ['arguments']])
    else:
        x = pd.DataFrame(x[target])
    x = x.drop(columns=list(dependencies.keys()), errors='ignore')
    
    x_list, y_list = [], []
    for t in target:
        x_list.append(x)
        y = pd.DataFrame(log['next'][t])
        y = y.drop(columns=list(dependencies.keys()), errors='ignore')
        y_list.append(y)
 
    return x_list, y_list

def train(cmd_logs, encoders, cap_value): # train all models

    result = []
    acc_r, acc_c = [], []
    for cmd in cmd_logs.keys():
        cap_list = cmd_logs[cmd]['target']
        print('---', 'TRAIN:', cmd, '---')
        argument = arg_exists(cmd_logs[cmd]['log'])
        x_list, y_list = split_x_y(cmd_logs[cmd], argument)

        types, score, test_len, models = [], [], [], []
        for c, x, y in zip(cap_list, x_list, y_list):

            if len(x)==1:
                continue;

            if y.columns[0].split('+')[-1] not in encoders.keys(): # regression
                type = 'r'
                types.append(type)
                x, encoders = normalize_toOnehot(cmd, x, encoders, cmd_logs[cmd]['target'], cap_value)  
            else:
                type = 'c'
                types.append(type)
                x, encoders = normalize_toOnehot(cmd, x, encoders, cmd_logs[cmd]['target'], cap_value, normalize=True)
            
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

            if type == 'c':
                num_class = len(encoders[y_test.columns[0].split('+')[-1]].classes_)

            x_train = x_train.to_numpy()
            x_test = x_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()

            if not argument and x.shape[1] == 1:
                    x_train = x_train.reshape(-1, 1)
                    x_test = x_test.reshape(-1, 1)
            
            if type == 'c':
                model = lm.LogisticRegression()
                model.coef_, model.intercept_, acc, LR = fit_classification(x_train, x_test, y_train, y_test, num_class, argument)
                model.coef_ = model.coef_.numpy()
                model.intercept_ = model.intercept_.numpy()
                model.classes_ = np.array([i for i in range(num_class)])

                s = model.score(x_test, y_test)
                acc_c.append(s)
            else:
                model = lm.LinearRegression()
                model.coef_, model.intercept_, acc, LR = fit_regression(x_train, x_test, y_train, y_test, argument)
                model.coef_ = model.coef_.numpy()
                model.intercept_ = model.intercept_.numpy()
                predict = [get_closest(cap_value[(c.split('+')[-1],)][0], p) for p in model.predict(x_test)]
                s = accuracy_score(predict, y_test)
                acc_r.append(s)
            
            models.append(model)
            print('Accuracy:', s)
            score.append(s)
            test_len.append(len(y_test))

        result.append(pd.DataFrame({'types': [[i for i in types]],
                                    'target': [tuple(t.split('+')[-1] for t in cmd_logs[cmd]['target'])],
                                    'argument': [argument if argument else False],
                                    'dependencies': [product_dep(cmd_logs[cmd]['dep'])],
                                    'score': [score],
                                    'test_len': [test_len],
                                    'model':[models],
                                    'score_mean': np.mean(score)
        }, index=[cmd]))
        print()

    print('Average Regression Score:', np.mean(acc_r))
    print('Average Classification Score:', np.mean(acc_c))
    print()
    result = pd.concat(result).sort_values(by=['score_mean', 'argument'], ascending=False).drop(columns='score_mean')
    print('=== RESULT ===')
    print(result)
    return result


def get_cmd_logs(log): # get logs grouped by commands
    cmd_groups = log.groupby(('cmd', 'command')).groups
    cmd_logs = {i:log.loc[cmd_groups[i]] for i in cmd_groups}
    return cmd_logs

def remove_failed_cmd(cmd_cap, success_cmd): # remove not working commands
    for c in list(cmd_cap.keys()):
        if c not in success_cmd:
            cmd_cap.pop(c)
    return cmd_cap

def rename_encoders(encoders): # change encoder name to original capabilities's name
    for cap in list(encoders.keys()):
        if '+' in cap:
            encoders.update({cap.split('+')[-1]:encoders[cap]})
            encoders.pop(cap)
    return encoders


@click.command()
@click.option('--device', type=str, default='bulb')
@click.option('--save', type=bool, default=True)
def main(device, save):
    
    if device == 'robotCleaner':
        path = 'data/log/robotCleaner'
    elif device == 'bulb':
        path = 'data/log/bulb'
    elif device == 'TV':
        path = 'data/log/TV'
    elif device == 'airPurifier':
        path = 'data/log/airPurifier'
        
    print('\n=====', device.upper(), '=====')

    cap_value = cap_values[device]
    
    cmd_cap, encoders = group_and_encode(device, path, cap_value)
    
    result = train(cmd_cap, encoders, cap_value)

    if save:
        save_data('models/'+device+'_trained', result)
        save_data('models/'+device+'_encoders', encoders)
    
if __name__ == '__main__':
    main()
