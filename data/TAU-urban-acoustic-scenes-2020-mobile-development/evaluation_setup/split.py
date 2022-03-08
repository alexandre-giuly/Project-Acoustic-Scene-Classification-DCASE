import random

#eg: shuffle_split(80, "xyz.csv")  80= 80% train, 20% Test. xyz.csv= source_file

def shuffle_split(train_size, filename):
	data=open(filename, "r")

	train1=open("train.csv", "w")
	test1= open("test.csv", "w")
	read_data= data.readlines()
	random.shuffle(read_data)
	length = (len(read_data)*train_size)/100
	b=[]
	c=[]
	for x,y in enumerate(read_data):
		if x < length:
			b.append(y)
		else:
			c.append(y)

	train1.writelines(b)
	test1.writelines(c)

	train1.close()
	test1.close()
	data.close()

shuffle_split(70, "fold1_evaluate.csv")