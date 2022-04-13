from math import ceil
import os
def sort_a(text,list):
    with open(text) as a:
        for i in a.readlines():
            if '"if":' in i:
                list.append((float(i[14:-1])))

def sum_a(text):
    length = 0
    dict = {}
    with open(text) as b:
        length+= len(b.readlines())
    with open(text) as a:

        t=0

        while True:
            lines=a.readline()
            t+=1
            if "train_dat" in lines:
                name = lines[22:-3]
                if name not in dict:
                    dict[name]=0.0
                else:
                    dict[name]+=float(next(a)[14:-1])



            if t==length:
                print("finish")
                break

    topk=10
    tt=str(os.path.basename(text).split('-')[1])
    with open("save_top/save_top{}_{}_new.json".format(tt,topk), "w") as w:
        k=sorted(dict.items(),key=lambda x:x[1],reverse=True)[:ceil((topk/100)*len(dict))]
        w.write(str(k))
        print(len(k))
    with open("save_top/save_top{}_{}_new_reverse.json".format(tt,topk), "w") as w:
        k=sorted(dict.items(),key=lambda x:x[1],reverse=False)[:ceil((topk/100)*len(dict))]
        w.write(str(k))
        print(len(k))
def sum_train(text):#把一个train里面所有的相加，我这把traindat和testdat写反了
    length = 0
    dict = {}
    with open(text) as b:
        length+= len(b.readlines())
    with open(text) as a:

        t=0

        while True:
            lines=a.readline()
            t+=1
            if "test_dat" in lines:
                name = lines[17:-3]
                dict[name]=0.0
            else:
                if "train_dat" in lines:

                    dict[name]+=(float(next(a)[14:-1]))



                if t==length:
                    print("finish")
                    break
                else:
                    continue
    topk=100
    tt=str(os.path.basename(text).split('-')[1])
    with open("save_top/save_top{}_{}_new.json".format(tt,topk), "w") as w:
        k=sorted(dict.items(),key=lambda x:x[1],reverse=True)[:ceil((topk/100)*len(dict))]
        w.write(str(k))
        print(len(k))
    with open("save_top/save_top{}_{}_new_reverse.json".format(tt,topk), "w") as w:
        k2=sorted(dict.items(),key=lambda x:x[1],reverse=False)[:ceil((topk/100)*len(dict))]
        w.write(str(k2))
        print(len(k2))
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))
# sum_a("result_Tracin/Tracin-317-macnn_2022-04-08-17-52-37.json")
# x=[]
# y=[]
# sort_a("result_Tracin/Tracin-317-macnn_2022-04-08-17-52-37.json",x)
# sort_a("result_Tracin/Tracin-317-macnn.json",y)
# print('Spearman Rho: %f' % spearman(x, y))
sum_train("result_Tracin/Tracin-1425-macnn.json")
# sum_a("result_Tracin/Tracin-317-macnn_2022-04-08-17-52-37.json")
