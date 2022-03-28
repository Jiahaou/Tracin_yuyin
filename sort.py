from math import ceil
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

    topk=50
    with open("save_top/save_top{}.json".format(topk), "w") as w:
        k=sorted(dict.items(),key=lambda x:x[1],reverse=True)[:ceil((topk/100)*len(dict))]
        w.write(str(k))
        print(k[199])

sum_a("result_Tracin/Tracin-99macnn_2022-03-24-22-05-54.json")

