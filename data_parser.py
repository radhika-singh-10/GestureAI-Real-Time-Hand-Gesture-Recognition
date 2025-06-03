import os
import csv

from collections import namedtuple

ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'label', 'path'])

class JpegDataset(object):

    def __init__(self, csv_path_input, csv_path_labels, data_root):
        self.classes = self.read_csv_labels(csv_path_labels)
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.csv_data = self.read_csv_input(csv_path_input, data_root)


    def read_csv_input(self, csv_path, data_root):
        csv_data = []
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')  
            for row in csv_reader:
                if len(row) < 2:
                    print(f"Warning: skipping line {csv_reader.line_num} in {csv_path} because it doesn't have enough columns: {row}")
                    continue
                # Assuming the first column is the video id and the second column is the label
                video_id, label = row[:2]  # Use slicing to avoid IndexError
                full_path = os.path.join(data_root, video_id)
                if label in self.classes:
                    csv_data.append(ListDataJpeg(video_id, label, full_path))
        return csv_data


    def read_csv_labels(self, csv_path):
        classes = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                classes.append(row[0])
        return classes

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict
