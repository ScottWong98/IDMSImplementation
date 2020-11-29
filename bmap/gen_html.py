import os
import shutil


def get_json(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(line)
    return lines[0]


def get_template(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(line)
    return lines


def replace_content(content, raw_data, filename):
    data_line = f"var raw_data = '{raw_data}';\n"
    with open(filename, 'w', encoding='utf-8') as f:

        for cnt, line in enumerate(content):
            if cnt == 22:
                f.write("%s" % data_line)
            else:
                f.write("%s" % line)


json_dir_name = r'../data/cluster_poi/'
html_dir_name = r'./cluster_poi/'
template_filename = r'./cluster_poi_template.html'

for filename in os.listdir(json_dir_name):
    uid = filename.split('.')[0]
    out_filename = html_dir_name + uid + '.html'
    shutil.copy(template_filename, out_filename)
    raw_json = get_json(json_dir_name + filename)
    template_content = get_template(template_filename)
    replace_content(template_content, raw_json, out_filename)
