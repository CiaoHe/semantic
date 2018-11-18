import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import matplotlib.pyplot as plt
import collections
import numpy as np
from data_util import *

# return list[line1 line2...]
def read_lines(filename):
	if not os.path.exists(filename):
		print('file cannot find!')
		exit()
	f = open(filename)
	lines = f.readlines()
	f.close()

	return lines

# return string
def read_all(filename):
	if not os.path.exists(filename):
		print('file cannot find!')
		exit()
	f = open(filename)
	data = f.read()
	f.close()
	return data

## part file into 2 parts: data and labels
def parse_data(in_file, data_file, label_file):
	'''
	procession: in_file-->data_file + label_file
	filetype: data_file[sen1,sen2,sen3,sen4]
			  label_file[sym(sen1,sen2),sym(sen3,sen4)]
	'''
	f1 = open(data_file,'a')
	f2 = open(label_file,'a')
	lines = read_lines(in_file)

	for line in lines:
		line = line.lower().strip() # lower capitalization
		sen1, sen2, score = line.split('\t')
		sen1 = sen1.strip()
		sen2 = sen2.strip()
		sen_all = sen1 + '\n' + sen2

		f1.write(sen_all + '\n')
		f2.write(score + '\n')
	f1.close()
	f2.close()

def number_to_string(numb):
    numb_str = {1000000:'million', 1000:'thousand', 100:'hundred', 90:'ninety', 
                80:'eighty', 70:'seventy', 60:'sixty', 50:'fifty', 40:'fourty', 
                30:'thirty', 20:'twenty', 19:'nineteen', 18:'eighteen', 
                17:'seventeen', 16:'sixteen', 15:'fifteen', 14:'fourteen', 
                13:'thirteen', 12:'twelve', 11:'eleven', 10:'ten', 9:'nine', 
                8:'eight', 7:'seven', 6:'six', 5:'five', 4:'four', 3:'three', 
                2:'two', 1:'one'}
    keys = sorted(numb_str.items(),key=lambda ele:ele[0],reverse=True)
    if numb == 0:
        return 'zero'
    res = ''
    for k, v in keys:
        if numb >= k:
            n = int(numb / k)
            need_numb = (numb >= 100)
            numb = numb % k
            if res != '': res + ' '
            if need_numb:
                res += number_to_string(n)
            res += v + ' '
            if need_numb and numb > 0: res += 'and '
    return res

# 判断字符串中是否都是数字
def is_number(s):
	if s.isdigit():
		return True
	return False

# alter the digits into number in a sentence
def alter_number(sen):
	new_sen = ''
	for w in sen.split():
		if is_number(w):
			new_sen += number_to_string(int(w))
		else:
			new_sen += w + ' '
	return new_sen

# alter the digits into number in a file
def alter_numer_file(filein, fileout):
	lines = read_lines(filein)
	f = open(fileout, 'a')
	for line in lines:
		line = line.strip().split('\t')
		sen1 = alter_number(line[0])
		sen2 = alter_number(line[1])
		f.write(sen1 + '\t' + sen2 + '\t' + line[2] + '\n')
	f.close()

def merge_two_line_to_a_line(file1, file2):
	# 把一个文件每个句子对拆分了的文件再重新把句子对合并
	'''
	file1-->file2
	file2type: [sen1+'\t'+sen2,...]
	'''
	f1 = open(file1)
	f2 = open(file2, 'a+')
	f1_lines = f1.readlines()

	for i in range(len(f1_lines)):
		if i % 2 == 0:
			new_line = '' + alter_number(f1_lines[i].strip()) + '\t' + alter_number(f1_lines[i+1].strip())
			f2.write(new_line +'\n')

	f1.close()
	f2.close()

# remove stopwords
def write_data_without_stopwords(in_file, out_file):
	lines = read_lines(in_file)
	f = open(out_file,'a')

	for line in lines:
		sen1 = line.split('\t')[0].lower()
		sen2 = line.split('\t')[1].lower()
		filtered1 = [w for w in sen1.split() if(w not in stopwords.words('english'))]
		filtered2 = [w for w in sen2.split() if(w not in stopwords.words('english'))]
		s1 = ' '.join(filtered1)
		s2 = ' '.join(filtered2)

		f.write(s1.strip() + '\t' + s2.strip() + '\n')
	f.close()

## average length
def count_sen_len(filename):
	lines = read_lines(filename)

	sent_count = {}
	sent_len_sum = 0
	max_len = 0
	for co, line in enumerate(lines):
		sen1 = line.split('\t')[0]
		sen2 = line.split('\t')[1]
		len1 = len(sen1.split())
		len2 = len(sen2.split())
		if len1 > max_len:
			max_len = len1
		if len2 > max_len:
			max_len = len2

		sent_len_sum += (len1 + len2)
		if len1 not in sent_count:
			sent_count[len1] = 1
		else:
			sent_co = sent_count[len1]
			sent_count[len1] = sent_co + 1

		if len2 not in sent_count:
			sent_count[len2] = 1
		else:
			sent_co = sent_count[len2]
			sent_count[len2] = sent_co + 1

	print('average length: ' + str(sent_len_sum / (2 * len(lines))))

	sent= sorted(sent_count.items(),key=lambda item:item[0])
	print('length \t sentence number')
	for k, v in sent:
		print(str(k) + '\t' + str(v))

	length = [x[0] for x in sent]
	# print(length)
	number = [x[1] for x in sent]
	# print(number)
	plt.plot(length, number)#,label=$cos(x^2)$)
	plt.plot(length, number, 'r')
	plt.xlabel('sentence length')
	plt.ylabel('number')
	plt.ylim(0, 150)
	plt.xlim(0, max_len)
	plt.title('tooken')
	# plt.legend()
	plt.show()

if __name__ == '__main__':
	# step 1: 把训练集和测试集中数据和label分开
	# parse_data('data/clinicalSTS.train.txt', 'data/train.txt', 'data/train_label.txt')
	# parse_data('data/clinicalSTS.test.gs.txt', 'data/test.txt', 'data/test_label.txt')
	# print('successful')
	# # step 2: 把数据进行token、词形还原
	# # #这里process使用的是data_util中的process()
	# process('data/train.txt', 'data/processed_train.txt')
	# process('data/test.txt', 'data/processed_test.txt')
	# print('successful')

	# merge_two_line_to_a_line('data/processed_train.txt', 'data/new_train.txt')
	# merge_two_line_to_a_line('data/processed_test.txt', 'data/new_test.txt')
	# print('successful')
	# step 3: 统计一下句子长度分布
	count_sen_len('data/new_train.txt')
	count_sen_len('data/new_test.txt')
	# print('successful')


