import pyb
import sensor, image, time, math
import os, tf
from machine import UART
######此为老车右sd卡main函数
pi=3.1415926
NET_SIZE = 80
BORDER_WIDTH = (NET_SIZE + 9) // 10
CROP_SIZE = NET_SIZE + BORDER_WIDTH * 2
pic_x=160#标准的中心坐标
pic_y=120
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA) # we run out of memory if the resolution is much bigger...
sensor.set_brightness(800)
sensor.skip_frames(time = 20)
sensor.set_auto_gain(False)  # must turn this off to prevent image washout...
sensor.set_auto_whitebal(False,(0,0,0))  # must turn this off to prevent image washout...
sensor.set_auto_exposure(False)
uart = UART(1, baudrate=115200)     # 初始化串口 波特率设置为115200 TX是B12 RX是B13
do_classify=False
do_line = False
rect_flag = False
model_select=False
#255是蓝色b阈值，看场上情况调整
rect_select=12550#检查是否为12550
clock = time.clock()
net_path1 = "kapian.tflite"
labels1 = [line.rstrip() for line in open("/sd/lables_kapian.txt")]
net_path2 = "zimu.tflite"
labels2 = [line.rstrip() for line in open("/sd/lables_zimu.txt")]
net1 = tf.load(net_path1, load_to_fb=True)
net2 = tf.load(net_path2, load_to_fb=True)
#20是黑胶宽度，0,1,2调试选项
threshold=200#检查是否为200
result_index=-1
count=0
def check_a_b(a,b,k):

    if(a-b<=k and a-b>=-k):
        return 1
    else:
        return 0

#计算两点角度值
def get_point_point_angle(p0,p1):
    x1,y1=p0
    x2,y2=p1
    if(y2!=y1):
        theta=math.atan((x2-x1)/(y2-y1))
        theta=theta/pi*180
        return int(theta)
    return 0

#矩形四点获取距离
def get_distance(p0,p1,p2,p3):
    x1,y1=p0
    x2,y2=p1
    x3,y3=p2
    x4,y4=p3

    dis1=(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)#得到距离
    dis2=(x2-x3)*(x2-x3)+(y2-y3)*(y2-y3)
    dis3=(x3-x4)*(x3-x4)+(y3-y4)*(y3-y4)
    dis4=(x4-x1)*(x4-x1)+(y4-y1)*(y4-y1)

    dis1=math.sqrt(dis1)
    dis2=math.sqrt(dis2)
    dis3=math.sqrt(dis3)
    dis4=math.sqrt(dis4)

    return int(dis1),int(dis2),int(dis3),int(dis4)

#两点长度计算
def get_point_point_distance(p0, p1):
    line_s_x, line_s_y=p0
    line_e_x, line_e_y=p1
    line_long = math.sqrt(pow(line_s_x - line_e_x, 2) + pow(line_s_y - line_e_y, 2))
    return line_long

#点到直线距离公式
def get_point_line_distance(point, p0, p1):
    point_x = point[0]
    point_y = point[1]
    line_s_x, line_s_y=p0
    line_e_x, line_e_y=p1
    #若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    #若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
    elif line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    else:
        #斜率
        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        #截距
        b = line_s_y - k * line_s_x
        #带入公式得到距离dis
        dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
        return dis
# 这个示例演示如何加载tflite模型并运行
# 这个示例演示如何加载tflite模型并运行
# 这个示例演示如何加载tflite模型并运行
while(True):
    clock.tick()
    msg=uart.read()
    #print(msg)
    rect_flag=False
    do_classify = b"\xf4" in msg or do_classify
    if(b"\xf5" in msg):
        model_select=not model_select
    if(b"\xf5" in msg):
        rect_select=(int)(257-rect_select/10000)*10000+rect_select%10000;
    if(b"\xd3" in msg):
        rect_select = (int)(1 - rect_select % 10) + (int)(rect_select / 10) * 10;
    img = sensor.snapshot()
    for r in img.find_rects(threshold = threshold,quality = rect_select):             # 在图像中搜索矩形
        rect_flag=True
        img_x=(int)((r.rect()[0]+r.rect()[2]/2)/2)              # 图像中心的x值
        img_y=(int)((r.rect()[1]+r.rect()[3]/2)/2)            # 图像中心的y值
        p0, p1, p2, p3 = r.corners()
        dis1, dis2, dis3, dis4 = get_distance(p0, p1, p2, p3)
        blob_angle = get_point_point_angle(p1, p2)
        x_error = img_x - pic_x
        y_error = img_y - pic_y
        #print("img_x:" + str(r.rect()[0]) + ",img_y:" + str(r.rect()[3])+","+str(blob_angle))
        if (x_error < 0):
            x_flag = 0  # 正负标志位
            x_error = -x_error
        else:
            x_flag = 1

        if (y_error < 0):
            y_flag = 0  # 正负标志位
            y_error = -y_error
        else:
            y_flag = 1
        uart_array1 = bytearray([0XE1,0xE2,x_error,y_error,x_flag,y_flag,blob_angle,0XE3])
        uart.write(uart_array1)
        if do_classify:
            corners = r.corners()
            img.rotation_corr(x_translation=CROP_SIZE, y_translation=CROP_SIZE, corners=corners)
            if model_select:
                obj=net2.classify(img,roi=(BORDER_WIDTH, BORDER_WIDTH, NET_SIZE, NET_SIZE))[0]
                res=obj.output()
                m=max(res)
                result_index=labels2[res.index(m)]
                uart_array2 = bytearray([0XBF,0XBF])#没有识别出任何东西
                if result_index== '1':#1
                    uart_array2 = bytearray([0XBF,0xB0])
                if result_index == '2':#2
                    uart_array2 = bytearray([0XBF,0xB1])
                if result_index == '3':#3
                    uart_array2 = bytearray([0XBF,0xB2])

                if result_index == 'A':#A
                    uart_array2 = bytearray([0XBF,0xC0])
                if result_index == 'B':#B
                    uart_array2 = bytearray([0XBF,0xC1])
                if result_index == 'C':#C
                    uart_array2 = bytearray([0XBF,0xC2])

                if result_index == 'D':#D
                    uart_array2 = bytearray([0XBF,0xC3])
                if result_index == 'E':#E
                    uart_array2 = bytearray([0XBF,0xC4])
                if result_index == 'F':#F
                    uart_array2 = bytearray([0XBF,0xC5])

                if result_index == 'G':#G
                    uart_array2 = bytearray([0XBF,0xC6])
                if result_index == 'H':#H
                    uart_array2 = bytearray([0XBF,0xC7])
                if result_index == 'I':#I
                    uart_array2 = bytearray([0XBF,0xC8])

                if result_index == 'J':#J
                    uart_array2 = bytearray([0XBF,0xC9])
                if result_index == 'K':#K
                    uart_array2 = bytearray([0XBF,0xCA])
                if result_index == 'L':#L
                    uart_array2 = bytearray([0XBF,0xCB])

                if result_index == 'M':#M
                    uart_array2 = bytearray([0XBF,0xCC])
                if result_index == 'N':#N
                    uart_array2 = bytearray([0XBF,0xCD])
                if result_index == 'O':#O
                    uart_array2 = bytearray([0XBF,0xCE])
                #print("1")
                #print("%s: %f" % (result_index, m))
                uart.write(uart_array2)
                do_classify=False
            else:
                obj=net1.classify(img,roi=(BORDER_WIDTH, BORDER_WIDTH, NET_SIZE, NET_SIZE))[0]
                res=obj.output()
                m=max(res)
                result_index=labels1[res.index(m)]
                uart_array2 = bytearray([0XBF,0XBF])#没有识别出任何东西
                #中0
                if result_index== 'firearms':#枪
                    uart_array2 = bytearray([0XBF,0xA0])
                if result_index == 'explosive':#炸弹
                    uart_array2 = bytearray([0XBF,0xA1])
                if result_index == 'dagger':#匕首
                    uart_array2 = bytearray([0XBF,0xA2])
                #上1
                if result_index == 'spontoon':#警棍
                    uart_array2 = bytearray([0XBF,0xA3])
                if result_index == 'fire_axe':#消防斧
                    uart_array2 = bytearray([0XBF,0xA4])
                if result_index == 'first_aid_kit':#急救包
                    uart_array2 = bytearray([0XBF,0xA5])
                #下2
                if result_index == 'flashlight':#手电筒
                    uart_array2 = bytearray([0XBF,0xA6])
                if result_index == 'intercom':#bb机
                    uart_array2 = bytearray([0XBF,0xA7])
                if result_index == 'bulletproof_vest':#防弹衣
                    uart_array2 = bytearray([0XBF,0xA8])
                #左3
                if result_index == 'telescope':#望远镜
                    uart_array2 = bytearray([0XBF,0xA9])
                if result_index == 'helmet':#头盔
                    uart_array2 = bytearray([0XBF,0xAA])
                if result_index == 'fire_engine':#救火车
                    uart_array2 = bytearray([0XBF,0xAB])
                #右4
                if result_index == 'ambulance':#救护车
                    uart_array2 = bytearray([0XBF,0xAC])
                if result_index == 'armoured_car':#装甲车
                    uart_array2 = bytearray([0XBF,0xAD])
                if result_index == 'motorcycle':#摩托车
                    uart_array2 = bytearray([0XBF,0xAE])
                #print("2")
                #print("%s: %f" % (result_index, m))
                uart.write(uart_array2)
                do_classify=False
        img.draw_rectangle(r.rect(), color = (255, 255, 255))
