from PIL import Image

#- Import desired image
img = Image.open("img1.png")
#- Create list of pixels. Output format: [(R, G, B, A)]
pixels = list(img.getdata())
def listing(pix):
    #- Turn tuple into list
    list1 = []
    for i in pix:
        templist = []
        for q in i:
            templist.append(q)
        list1.append(templist)   
    return list1
    #- I'm writing this after i made it, i havent the foggiest clue how this works.
#- Results
print(listing(pixels))
