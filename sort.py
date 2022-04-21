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
            if "train_dat" in lines:
                name = lines[17:-3]

                dict[name]=0.0
            else:
                if "test_dat" in lines:

                    dict[name]+=(float(next(a)[14:-1]))



                if t==length:
                    print("finish")
                    break
                else:
                    continue
    topk=100
    tt=str(os.path.basename(text).split('-')[1])
    with open("save_top_no_model/save_top{}_{}_new.json".format(tt,topk), "w") as w:
        k=sorted(dict.items(),key=lambda x:x[1],reverse=True)[:ceil((topk/100)*len(dict))]
        w.write(str(k))
        print(len(k))
    with open("save_top_no_model/save_top{}_{}_new_reverse.json".format(tt,topk), "w") as w:
        k2=sorted(dict.items(),key=lambda x:x[1],reverse=False)[:ceil((topk/100)*len(dict))]
        w.write(str(k2))
        print(len(k2))

# sum_a("result_Tracin/Tracin-317-macnn_2022-04-08-17-52-37.json")


# sum_train("result_Tracin/Tracin-1425-macnn_2022-04-13-10-44-40.json")
# sum_a("result_Tracin/Tracin-317-macnn_2022-04-08-17-52-37.json")

def sum_checkpoint(text1,text2,text3):  # 把一个train里面所有的相加，我这把traindat和testdat写反了
    length1 = 0
    length2 = 0
    length3 = 0
    dict = {}
    with open(text1) as a1:
        length1 += len(a1.readlines())
    with open(text2) as a2:
        length2 += len(a2.readlines())
    with open(text3) as a3:
        length3 += len(a3.readlines())
    with open(text1) as a:
        t = 0
        while True:
            lines = a.readline()
            t += 1
            if "train_dat" in lines:
                name = lines[17:-3]
                dict[name] = 0.0
            else:
                if "test_dat" in lines:
                    dict[name] += (float(next(a)[14:-1]))
                if t == length1:
                    print("finish")
                    break
                else:
                    continue
    with open(text2) as b:
        t = 0
        while True:
            lines = b.readline()
            t += 1
            if "train_dat" in lines:
                name = lines[17:-3]

            else:
                if "test_dat" in lines:
                    dict[name] += (float(next(b)[14:-1]))
                if t == length2:
                    print("finish")
                    break
                else:
                    continue
    with open(text3) as c:
        t = 0
        while True:
            lines = c.readline()
            t += 1
            if "train_dat" in lines:
                name = lines[17:-3]

            else:
                if "test_dat" in lines:
                    dict[name] += (float(next(c)[14:-1]))
                if t == length3:
                    print("finish")
                    break
                else:
                    continue
    topk = 70
    tt = str(os.path.basename(text1).split('-')[1])
    with open("save_top_no_model/save_top{}_{}_new.json".format(tt, topk), "w") as w:
        k = sorted(dict.items(), key=lambda x: x[1], reverse=True)[:ceil((topk / 100) * len(dict))]
        w.write(str(k))
        print(len(k))
    with open("save_top_no_model/save_top{}_{}_new_reverse.json".format(tt, topk), "w") as w:
        k2 = sorted(dict.items(), key=lambda x: x[1], reverse=False)[:ceil((topk / 100) * len(dict))]
        w.write(str(k2))
        print(len(k2))
sum_checkpoint("result_Tracin/Tracin-1425-macnn.json","result_Tracin/Tracin-1425-macnn_2022-04-20-14-31-42.json","result_Tracin/Tracin-1425-macnn_2022-04-20-17-06-39.json")
