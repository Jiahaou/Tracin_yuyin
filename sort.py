

x= {}
def sort_a(text,list):
    t=0
    with open(text) as a:
        for i in a.readlines():
            if '"if":' in i:
                list[t]=((float(i[14:-1])))
                t+=1
sort_a("result_Tracin/Tracin-71macnn.json",x)

print([i[0] for i in x.items() if i[1]==min(x.values())])
print(2030//170,2030%170)
print(min(x.values()))