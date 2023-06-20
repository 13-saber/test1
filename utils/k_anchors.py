import glob
import xml.etree.ElementTree as ET
import numpy as np
import bisect
import math

import matplotlib.pyplot as plt
import os, cv2
import seaborn as sns

current_palette = list(sns.xkcd_rgb.values())

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']


#计算IOU值
def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou


def iou_kpp(box, clusters):
    x = np.minimum(clusters[0], box[0])
    y = np.minimum(clusters[1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[0] * clusters[1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])




'''
def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    #print(d)
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers
'''

def  init_centroids(boxes,n_anchors):
    #使用k-means++ 初始化，尽量使用使质心之间的距离较大
    centroids=[]
    boxes_num=len(boxes)
    # distance_list = [0 for _ in range(boxes_num)]


    #先随机从所有bbox中选一个作为初始的质心
    centroids_index= np.random.choice(boxes_num)
    centroids.append(boxes[centroids_index])
    # centroids.append(np.random.choice(boxes))
    #然后依次迭代，挑选剩下的质心，直到满足规定的数量
    for _ in range(n_anchors-1):
        #所有bbox到当前已有质心的最短距离总和
        sum_distance=0
        #记录每个bbox到当前已有质心的最短距离
        distance_list=[]

        for box in boxes:
            min_distance=1
            #对每个bbox计算其到当前已存在的质心的距离，记录最短的距离
            for centroid_i,centroid in enumerate(centroids):
                distance =(1-iou_kpp(box,centroid))
                if distance<min_distance:
                    min_distance=distance
            sum_distance+=min_distance
            distance_list.append(min_distance)
        
        #各bbox的最短距离累加，形成概率区间,概率区间通过累加各gt box到最近质心的距离来构建
        # p=[]
        # for dis in distance_list:
        #     p.append(sum(p)+dis/sum_distance)
        # #选取新的质心
        # thresh=np.random.random()
        # new_centroid_idex=bisect.bisect(p,thresh)
        # print(new_centroid_idex)
        # new_cettroid=boxes[new_centroid_idex]
        # centroids.append(new_cettroid)
        print(sum_distance)
        sum_distance *= np.random.random()
        print(sum_distance)
        for i, di in enumerate(distance_list): # 轮盘法选出下一个聚类中心；
            sum_distance -= di
            if sum_distance > 0:
                continue
            centroids.append(boxes[i])
            break
    return centroids

def kmeans(bbox,k):
    # 取出一共有多少框
    row = bbox.shape[0]

    #每个框各个点的位置
    distance = np.empty((row,k))

    #最后聚类的位置
    last_clu = np.zeros((row,))

    np.random.seed()

    #随机选取5个当聚类中心 --k-means 算法
    cluster = bbox[np.random.choice(row,k,replace=False)]

    #k-means ++ 初始化聚类中心
    # cluster=init_centroids(bbox,k)
    # cluster=np.array(cluster)
    # print(cluster)

    while True:
        for i in range(row):
            # 计算每一行距离五个点的iou情况
            distance[i]=1-cas_iou(bbox[i],cluster)
        # 取出最小点
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break

        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(bbox[near == j],axis=0)
            
        last_clu = near

    return cluster ,near, distance

def load_data(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
         # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


def calculate_anchors(dataset_anno_path = r'./VOCdevkit/VOC2007/Annotations', anchors_num = 9, SIZE = 300):
    # 计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成ssd_anchors.txt
    # SIZE = 300
    # anchors_num = 9
    # # 载入数据集，可以使用VOC的xml
    # path = r'./VOCdevkit/VOC2007/Annotations'
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data(dataset_anno_path)
    # 使用k聚类算法
    out,_,_= kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)

    anchor_ratio = np.around(out[:, 0] / out[:, 1], decimals=2)
    anchor_ratio = list(anchor_ratio)
    print('Final anchor_ratio: ', anchor_ratio)
    print('Sorted anchor ratio: ', sorted(anchor_ratio))


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object':[]}
 
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text
                img['filename'] = path_to_image
                ## make sure that the image exists:
                if not os.path.exists(path_to_image):
                    assert False, "file does not exist!\n{}".format(path_to_image)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        
                        obj['name'] = attr.text
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']]  = 1
                        
 
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
 
        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels

def plot_cluster_result(plt,clusters,nearest_clusters,WithinClusterSumDist,wh,k):
    for icluster in np.unique(nearest_clusters):
        pick = nearest_clusters==icluster
        c = current_palette[icluster]
        plt.rc('font', size=8) 
        plt.plot(wh[pick,0],wh[pick,1],"p",
                 color=c,
                 alpha=0.5,label="cluster = {}, N = {:6.0f}".format(icluster,np.sum(pick)))
        plt.text(clusters[icluster,0],
                 clusters[icluster,1],
                 "c{}".format(icluster),
                 fontsize=20,color="red")
        plt.title("Clusters=%d" %k)
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist)) 


if __name__ == '__main__':
    #数据集xml注释路径
    dataset_anno_path = r'./VOCdevkit/VOC2007/Annotations'
    

    train_image_folder = "./VOCdevkit/VOC2007/JPEGImages/"
    train_annot_folder = "./VOCdevkit/VOC2007/Annotations/"

    train_image, seen_train_labels = parse_annotation(train_annot_folder,train_image_folder, labels=LABELS)
    print("N train = {}".format(len(train_image)))

    y_pos = np.arange(len(seen_train_labels))
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(1,1,1)
    ax.barh(y_pos,list(seen_train_labels.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(seen_train_labels.keys()))
    ax.set_title("The total number of objects = {} in {} images".format(
        np.sum(list(seen_train_labels.values())),len(train_image)
    ))
    plt.savefig("object.png")
    # plt.show()
    plt.cla()

    wh = []
    for anno in train_image:
        aw = float(anno['width'])  # width
        ah = float(anno['height']) # height 
        for obj in anno["object"]:
            w = (obj["xmax"] - obj["xmin"])/aw # 归一化 范围为[0,GRID_W)
            h = (obj["ymax"] - obj["ymin"])/ah # 归一化 范围为[0,GRID_H)
            temp = [w,h]
            wh.append(temp)
    wh = np.array(wh)
    print("clustering feature data is ready. shape = (N object, width and height) =  {}".format(wh.shape))

    #查看归一化后分布图
    plt.figure(figsize=(10,10))
    plt.scatter(wh[:,0],wh[:,1],alpha=0.3)
    plt.title("Clusters",fontsize=20)
    plt.xlabel("normalized width",fontsize=20)
    plt.ylabel("normalized height",fontsize=20)
    plt.savefig("Clusters.png")
    # plt.show()
    plt.cla()


    #生成的anchors数量
    anchors_num = 2
    #输入的图片尺寸
    SIZE = 300

    result={}
    clusters,nearest_clusters,distances=kmeans(wh,anchors_num)
    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
    result = {"clusters":             clusters,
              "nearest_clusters":     nearest_clusters,
              "distances":            distances,
              "WithinClusterMeanDist": WithinClusterMeanDist}
    print("{:2.0f} clusters: mean IoU = {:5.4f}".format(anchors_num,1-result["WithinClusterMeanDist"]))

    figsize = (15,15)
    count =1 
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,count)
    plot_cluster_result(plt,clusters,nearest_clusters,1 - WithinClusterMeanDist,wh,anchors_num)
    plt.savefig(str(anchors_num)+"_results.png")
    # plt.show()
    plt.cla()

    out=clusters
    out = out[np.argsort(out[:,0])]
    anchor_ratio = np.around(out[:, 0] / out[:, 1], decimals=2)
    anchor_ratio = list(anchor_ratio)
    print('Final anchor_ratio: ', anchor_ratio)
    print('Sorted anchor ratio: ', sorted(anchor_ratio))







    # #生成的anchors数量
    # anchors_num = 6
    # #输入的图片尺寸
    # SIZE = 300
    # calculate_anchors(dataset_anno_path, anchors_num, SIZE)
