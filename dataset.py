import os
import csv
import time
import datetime
import random
import numpy as np
import copy
import json

class Dataset(object):

    class _Dataset(object):
        def __init__(self, data, label, shuffle, unit_length):
            self.data = data
            self.label = label
            self.shuffle = shuffle
            self.ulength = unit_length
            self.reset()

        def reset(self):
            self.samples = range(self.length())
            if self.shuffle:
                random.shuffle(self.samples)

        def get(self, index=None):
            begin = None
            if index is None:
                if len(self.samples) > 0:
                    begin = self.samples.pop(0)
            else:
                if index < self.length():
                    begin = index
            if begin is None:
                return None, None
            else:
                end   = begin + self.unit_length()
                return self.data[begin:end],  self.label[end+1]

        def length(self):
            return self.data.shape[0] - self.unit_length() - 1

        def unit_length(self):
            return self.ulength

    def __init__(self, header, dataset, stocks, shuffle, unit_length):
        self.columns = columns = [ 10, 11, 12, 13, 6, 8, 9 ]
        header  = map(lambda x: header[x], columns)
        print("Dataset columns is {}.".format(header))
        dataset, means, sigmas, maxes, mins = self._normalization(dataset, columns)
        data  = dataset[:, :, columns]
        label = dataset[:, stocks, -1]
        label = np.greater_equal(label, np.zeros(label.shape))
        label = np.asarray(label, dtype=np.int)
        label = label.reshape([-1])
        train_length = int(len(data) * 0.9)
        self.train = self._Dataset(data[:train_length],
                                   label[:train_length], shuffle, unit_length)
        self.test  = self._Dataset(data[train_length:],
                                   label[train_length:], shuffle, unit_length)
        print("Dataset shape is ({}x{}x{}).".format(self.unit_length(), self.columns_size(), self.channel_size()))
        print("Train dataset length is {}.".format(self.train.length()))
        print("Test dataset length is {}.".format(self.test.length()))

    def unit_length(self):
        return self.train.unit_length()

    def columns_size(self):
        return self.train.data.shape[1]

    def channel_size(self):
        return len(self.columns)

    def reset(self):
        self.train.reset()
        self.test.reset()

    def _normalization(self, dataset, columns): # times, markets, prices
        dataset = np.array(dataset).transpose(2, 0, 1) # prices, times, markets
        means   = []
        sigmas  = []
        maxes   = []
        mins    = []
        for i in columns:
            tensor = dataset[i]
            mean   = np.mean(tensor)
            sigma  = np.std(tensor)
            means.append(mean)
            sigmas.append(sigma)
            maxes.append(np.max(tensor))
            mins.append(np.min(tensor))
            dataset[i] = (tensor - mean) / sigma
        dataset  = dataset.transpose(1, 2, 0) # times, markets, prices
        means    = np.asarray(means, dtype=np.float32)
        sigmas   = np.asarray(sigmas, dtype=np.float32)
        maxes    = np.asarray(maxes, dtype=np.float32)
        mins     = np.asarray(mins, dtype=np.float32)
        return dataset, means, sigmas, maxes, mins

def load(market_dir, stock_dir, shuffle=True, unit_length=20):
    market_list = _csv_list(market_dir)
    stock_list  = _csv_list(stock_dir)
    file_list   = stock_list + market_list

    print("Filelist is {}. length={}".format(np.asarray(file_list), len(file_list)))
    dataset = map(lambda path: _load(path), file_list)
    dataset = map(lambda data: _add_xdays(data), dataset)
    dataset = map(lambda data: _parse_float(data), dataset)
    dataset = map(lambda data: _sort(data), dataset)

    base_date = max(map(lambda (h, d): d[0][1], dataset))
    print("Dataset from {}.".format(base_date))

    dataset = map(lambda data: _reject_too_old(data, base_date), dataset)
    dataset = map(lambda data: _date_padding(data), dataset)
    dataset = map(lambda data: _add_gap(data), dataset)

    header, dataset = _reshape(dataset)
    print("Original dataset columns is {}.".format(header))

    return Dataset(header, dataset, range(len(stock_list)), shuffle, unit_length)

def _csv_list(dir):
    return map(lambda filename: os.path.join(dir, filename), filter(lambda filename: filename.endswith(".csv"), os.listdir(dir)))

def _reshape(datasets):
    header = None
    merged_dataset = []
    for (header, dataset) in datasets:
        merged_dataset.append(dataset)
    merged_dataset = np.asarray(merged_dataset, dtype=object) # markets, times, prices
    merged_dataset = merged_dataset.transpose(1, 0, 2) # times, markets, prices
    return header, merged_dataset

def _add_gap(dataset):
    header, dataset = dataset
    prev = None
    new_dataset = []
    for row in dataset:
        if prev is not None:
            prev_closed = prev[5]
            for i in xrange(2, 6):
                row.append((row[i] - prev_closed) / prev_closed)
            dayOfGap = (row[5] - row[2]) / row[2]
            row.append(dayOfGap)
            new_dataset.append(row)
        prev = row
    header = header + [ 'openGap', 'closeGap', 'higherGap', 'lowerGap', 'dayOfGap' ]
    return header, new_dataset

def _date_padding(dataset):
    header, dataset = dataset
    prev = None
    new_dataset = []
    for row in dataset:
        current = row[1]
        if prev is not None:
            period = (current - (prev + datetime.timedelta(days=1))).days
            for diff in xrange(1, period + 1):
                new_row = copy.copy(row)
                new_row[1] = prev + datetime.timedelta(days=diff)
                new_dataset.append(new_row)
        new_dataset.append(row)
        prev = current
    new_dataset = filter(lambda x: x[1].weekday() < 5, new_dataset)
    return header, new_dataset

def _reject_too_old(dataset, base_date):
    header, dataset = dataset
    dataset = filter(lambda row: row[1] >= base_date, dataset)
    return header, dataset

def _sort(dataset):
  header, dataset = dataset
  dataset.sort(key=lambda x: x[1])
  return header, dataset

def _parse_float(dataset):
    header, dataset = dataset
    for row in dataset:
        for i in xrange(2, len(row)):
            row[i] = float(row[i])
    return header, dataset

def _add_xdays(dataset):
    header, payload = dataset
    dataset = []
    def to_date(x):
        if "/" in x:
            return datetime.datetime.strptime(x, "%Y/%m/%d")
        else:
            return datetime.datetime.strptime(x, "%Y-%m-%d")
    def to_yday(x):
        return to_date(x).timetuple().tm_yday
    def to_wday(x):
        return to_date(x).timetuple().tm_wday
    for row in payload:
      date = row[1]
      row[1] = to_date(date)
      row = row + [ to_yday(date), to_wday(date) ]
      dataset.append(row)
    header = header + [ 'yday', 'wday' ]
    return header, dataset

def _load(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = [ 'path' ] + next(reader)
        header = map(lambda x: x.lower(), header)
        payload = []
        for row in reader:
            payload.append([ path ] + row)
    return header, payload

