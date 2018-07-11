import  Image
im = Image.open("images/test.gif")
his = im.histogram()
print(len(his))
values = {}
for i in range(0,256):
    values[i] = his[i]

temp = sorted(values.items(),key=lambda x: x[1],reverse=True)
#print(temp)

#将占比为10的颜色打印出来:
for j, k in temp[:10]:
    print(j, k)
#构造性的无杂质图片:
im2 = Image.new("P",im.size,1024)
print(im2.size[1])