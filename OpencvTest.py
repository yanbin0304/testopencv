# -*- coding=GBK -*-
import cv2 as cv
import Image
import pytesseract


# def recognize_text():
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
#     binl = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
#     open_out = cv.morphologyEx(binl, cv.MORPH_OPEN, kernel)
#     cv.bitwise_not(open_out, open_out)  # ������Ϊ��ɫ
#     cv.imshow("ת��", open_out)
#     textImage = Image.fromarray(open_out)
#     text = pytesseract.image_to_string(textImage)
#     print("This OK:%s" % text)
#
#
# src = cv.imread("images/cmef.jpg")
# cv.imshow("ԭ��", src)
# recognize_text()
# cv.waitKey(0)
# cv.destroyAllWindows()
from PIL import Image
from pytesseract import *
from fnmatch import fnmatch
from queue import Queue
import matplotlib.pyplot as plt
import cv2
import time
import os





def clear_border(img,img_name):
  '''ȥ���߿�
  '''

  filename = './out_img/' + img_name.split('.')[0] + '-clearBorder.jpg'
  h, w = img.shape[:2]
  for y in range(0, w):
    for x in range(0, h):
      # if y ==0 or y == w -1 or y == w - 2:
      if y < 4 or y > w -4:
        img[x, y] = 255
      # if x == 0 or x == h - 1 or x == h - 2:
      if x < 4 or x > h - 4:
        img[x, y] = 255

  cv2.imwrite(filename,img)
  return img


def interference_line(img, img_name):
  '''
  �����߽���
  '''

  filename =  './out_img/' + img_name.split('.')[0] + '-interferenceline.jpg'
  h, w = img.shape[:2]
  # ������opencv������Ƿ���
  # img[1,2] 1:ͼƬ�ĸ߶ȣ�2��ͼƬ�Ŀ��
  for y in range(1, w - 1):
    for x in range(1, h - 1):
      count = 0
      if img[x, y - 1] > 245:
        count = count + 1
      if img[x, y + 1] > 245:
        count = count + 1
      if img[x - 1, y] > 245:
        count = count + 1
      if img[x + 1, y] > 245:
        count = count + 1
      if count > 2:
        img[x, y] = 255
  cv2.imwrite(filename,img)
  return img

def interference_point(img,img_name, x = 0, y = 0):
    """�㽵��
    9�����,�Ե�ǰ��Ϊ���ĵ����ֿ�,�ڵ����
    :param x:
    :param y:
    :return:
    """
    filename =  './out_img/' + img_name.split('.')[0] + '-interferencePoint.jpg'
    # todo �ж�ͼƬ�ĳ��������
    cur_pixel = img[x,y]# ��ǰ���ص��ֵ
    height,width = img.shape[:2]

    for y in range(0, width - 1):
      for x in range(0, height - 1):
        if y == 0:  # ��һ��
            if x == 0:  # ���϶���,4����
                # ���ĵ��Ա�3����
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # ���϶���
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # ���ϷǶ���,6����
                sum = int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        elif y == width - 1:  # ������һ��
            if x == 0:  # ���¶���
                # ���ĵ��Ա�3����
                sum = int(cur_pixel) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x, y - 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # ���¶���
                sum = int(cur_pixel) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y - 1])

                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # ���·Ƕ���,6����
                sum = int(cur_pixel) \
                      + int(img[x - 1, y]) \
                      + int(img[x + 1, y]) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x + 1, y - 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        else:  # y���ڱ߽�
            if x == 0:  # ��߷Ƕ���
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # �ұ߷Ƕ���
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            else:  # �߱�9����������
                sum = int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 4 * 245:
                  img[x, y] = 0
    cv2.imwrite(filename,img)
    return img

def _get_dynamic_binary_image(filedir, img_name):
  '''
  ����Ӧ��ֵ��ֵ��
  '''

  filename =   './out_img/' + img_name.split('.')[0] + '-binary.jpg'
  img_name = filedir + '/' + img_name
  print('.....' + img_name)
  im = cv2.imread(img_name)
  im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

  th1 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
  cv2.imwrite(filename,th1)
  return th1

def _get_static_binary_image(img, threshold = 140):
  '''
  �ֶ���ֵ��
  '''

  img = Image.open(img)
  img = img.convert('L')
  pixdata = img.load()
  w, h = img.size
  for y in range(h):
    for x in range(w):
      if pixdata[x, y] < threshold:
        pixdata[x, y] = 0
      else:
        pixdata[x, y] = 255

  return img


def cfs(im,x_fd,y_fd):
  '''�ö��кͼ��ϼ�¼������������������浥���ݹ��Խ��cfs���ʹ�������
  '''

  # print('**********')

  xaxis=[]
  yaxis=[]
  visited =set()
  q = Queue()
  q.put((x_fd, y_fd))
  visited.add((x_fd, y_fd))
  offsets=[(1, 0), (0, 1), (-1, 0), (0, -1)]#������

  while not q.empty():
      x,y=q.get()

      for xoffset,yoffset in offsets:
          x_neighbor,y_neighbor = x+xoffset,y+yoffset

          if (x_neighbor,y_neighbor) in (visited):
              continue  # �Ѿ����ʹ���

          visited.add((x_neighbor, y_neighbor))

          try:
              if im[x_neighbor, y_neighbor] == 0:
                  xaxis.append(x_neighbor)
                  yaxis.append(y_neighbor)
                  q.put((x_neighbor,y_neighbor))

          except IndexError:
              pass
  # print(xaxis)
  if (len(xaxis) == 0 | len(yaxis) == 0):
    xmax = x_fd + 1
    xmin = x_fd
    ymax = y_fd + 1
    ymin = y_fd

  else:
    xmax = max(xaxis)
    xmin = min(xaxis)
    ymax = max(yaxis)
    ymin = min(yaxis)
    #ymin,ymax=sort(yaxis)

  return ymax,ymin,xmax,xmin

def detectFgPix(im,xmax):
  '''�����������
  '''

  h,w = im.shape[:2]
  for y_fd in range(xmax+1,w):
      for x_fd in range(h):
          if im[x_fd,y_fd] == 0:
              return x_fd,y_fd

def CFS(im):
  '''�и��ַ�λ��
  '''

  zoneL=[]#�����鳤��L�б�
  zoneWB=[]#�������X��[��ʼ���յ�]�б�
  zoneHB=[]#�������Y��[��ʼ���յ�]�б�

  xmax=0#��һ��������ڵ������,�����ǳ�ʼ��
  for i in range(10):

      try:
          x_fd,y_fd = detectFgPix(im,xmax)
          # print(y_fd,x_fd)
          xmax,xmin,ymax,ymin=cfs(im,x_fd,y_fd)
          L = xmax - xmin
          H = ymax - ymin
          zoneL.append(L)
          zoneWB.append([xmin,xmax])
          zoneHB.append([ymin,ymax])

      except TypeError:
          return zoneL,zoneWB,zoneHB

  return zoneL,zoneWB,zoneHB


def cutting_img(im,im_position,img,xoffset = 1,yoffset = 1):
  filename =  './out_img/' + img.split('.')[0]
  # ʶ������ַ�����
  im_number = len(im_position[1])
  # �и��ַ�
  for i in range(im_number):
    im_start_X = im_position[1][i][0] - xoffset
    im_end_X = im_position[1][i][1] + xoffset
    im_start_Y = im_position[2][i][0] - yoffset
    im_end_Y = im_position[2][i][1] + yoffset
    cropped = im[im_start_Y:im_end_Y, im_start_X:im_end_X]
    cv2.imwrite(filename + '-cutting-' + str(i) + '.jpg',cropped)



def main():
  filedir = './images'

  for file in os.listdir(filedir):
    if fnmatch(file, '*.jpg'):
      img_name = file

      # ����Ӧ��ֵ��ֵ��
      im = _get_dynamic_binary_image(filedir, img_name)

      # ȥ���߿�
      im = clear_border(im,img_name)

      # ��ͼƬ���и����߽���
      im = interference_line(im,img_name)

      # ��ͼƬ���е㽵��
      im = interference_point(im,img_name)

      # �и��λ��
      im_position = CFS(im)

      maxL = max(im_position[0])
      minL = min(im_position[0])

      # �����ճ���ַ������һ���ַ��ĳ��ȹ�������Ϊ��ճ���ַ��������м�����и�
      if(maxL > minL + minL * 0.7):
        maxL_index = im_position[0].index(maxL)
        minL_index = im_position[0].index(minL)
        # �����ַ��Ŀ��
        im_position[0][maxL_index] = maxL // 2
        im_position[0].insert(maxL_index + 1, maxL // 2)
        # �����ַ�X��[��ʼ���յ�]λ��
        im_position[1][maxL_index][1] = im_position[1][maxL_index][0] + maxL // 2
        im_position[1].insert(maxL_index + 1, [im_position[1][maxL_index][1] + 1, im_position[1][maxL_index][1] + 1 + maxL // 2])
        # �����ַ���Y��[��ʼ���յ�]λ��
        im_position[2].insert(maxL_index + 1, im_position[2][maxL_index])

      # �и��ַ���Ҫ���еúþ͵����ò�����ͨ�� 1 or 2 �Ϳ���
      cutting_img(im,im_position,img_name,1,1)

      # ʶ����֤��
      cutting_img_num = 0
      for file in os.listdir('./out_img'):
        str_img = ''
        if fnmatch(file, '%s-cutting-*.jpg' % img_name.split('.')[0]):
          cutting_img_num += 1
      for i in range(cutting_img_num):
        try:
          file = './out_img/%s-cutting-%s.jpg' % (img_name.split('.')[0], i)
          # ʶ����֤��
          str_img = str_img + image_to_string(Image.open(file),lang = 'eng', config='-psm 10') #�����ַ���10��һ���ı���7
        except Exception as err:
          pass
      print('��ͼ��%s' % cutting_img_num)
      print('ʶ��Ϊ��%s' % str_img)

if __name__ == '__main__':
  main()