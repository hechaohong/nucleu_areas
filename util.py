import sqlite3, cv2, msgpack
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as sco
import matplotlib.pyplot as plt
plt.ion()

# http://sip3.delinp.cn:7180/#/embryo
COLORS = {
    1: (0,255,0),   # 原核
    2: (255,255,0), # 卵裂球
    3: (255,0,0),   # 碎片
    4: (0,255,255)  # 边界
}

def load_annotates():
    sconn = sqlite3.connect('PC10T4LL_44771.6090665393')
    cursor = sconn.cursor().execute('select * from annotates')
    rows = cursor.fetchall()
    sconn.close()
    ans = [msgpack.loads(an) for path,an in rows]
    ans.sort(key=lambda an:an['time'])   
    return ans

def show_annotate(an):
    path,labels,segs = 'D2022.07.30_S02010_I3263_P/WELL02/'+an['path'], an['labels'],an['segs']
    image = cv2.imread(path)
    # 绘制标注
    for label,seg in zip(labels,segs):
        color = COLORS[label]
        seg = (np.array(seg)*2).reshape([1,-1,2])
        image = cv2.polylines(image, seg, True, color, 2 )
    plt.imshow(image)
    
    q = cv2.Laplacian(image, cv2.CV_64F).var()
    print('time:%s 图片质量: %f' % (an['time'], q)  )
        
def get_moment(seg):
    # idx 表示这条曲线在标注中的位置
    M = cv2.moments( (np.array(seg)*2).astype(np.int32).reshape((1,-1,2)) );
    return [int(M['m00']), int(round(M["m10"]/M['m00'])) ,int(round(M["m01"]/M['m00'])) ]

def moments_pn(ans):
    # 对单细胞时期(并且只识别到1个原核和2个原核的标注数据)计算面积和重心坐标，返回一个列表
    MS=[]
    for an in ans:
        if an['labels'].count(2)>1:
            break
        idxs = np.where( np.array( an['labels'] ) == 1 )[0]
        if len(idxs) in [1,2]:
            pns = [ get_moment(seg)  for idx,seg in enumerate(an['segs']) if idx in idxs ]         # area = M['m00'], cx=int(M["m10"] / M["m00"]),cy=int(M["m01"] / M["m00"])
            sorted(pns, key=lambda l:l[0], reverse=True )
            MS.append( {'time': float(an['path'].split('_')[1][1:])  , 'pns': pns } )
    return MS

def pos_pn2( moments ):
    """ 
    机器识别有出错的可能，可能多识别，也可能漏识别
    从原核数据中估计 双原核开始的位置,返回双原核图片的序号    
    """
    tns = np.array([ [v['time'],len(v['pns'])]  for v in moments ])
    # 从第一个1pn开始
    idx0=0
    idx = 0
    L = len(tns)
    # 连续3张2pn
    for idx in range(L-3):
        if tns[idx,1]>1 and tns[idx:idx+3,1].sum()==6: break
    if idx<L/2: 
        return idx +idx0

    # 连续4张中有3张2pn
    for idx in range(L-4): # 至少要留5张2pn的图片来做拟合
        if tns[idx,1]>1 and tns[idx:idx+4,1].sum()>=7: break
    if idx<L/2: 
        return idx +idx0

    # 连续5张中有3张2pn
    for idx in range(L-5): # 至少要留5张2pn的图片来做拟合
        if tns[idx,1]>1 and tns[idx:idx+5,1].sum()>=8: break
    if idx<L/2: 
        return idx +idx0

    # 放宽到 连续2张2pn
    for idx in range(L-2):
        if tns[idx,1]>1 and tns[idx:idx+2,1].sum()==4: break
    if idx<(L*0.8): 
        return idx +idx0
    # 只要找到一张就认为是
    for idx in range(L):
        if tns[idx,1]>1:
            return idx +idx0
    # 没有得到双原核的图片位置        
    return -1

def find_center(points):
    # 查找一系列点的中心点
    def func(x):
        # 到中心点的距离之后
        d = np.array(points)-x
        return np.sum(d*d)
    x0 =  np.array(points).mean(axis=0).tolist() 
    return sco.fmin_cg(func,x0,disp=False)

def tidy_moments( moments ):
    # 用位置跟踪的方式整理原核数据列表
    v1 = np.diff(np.array(moments[0]['pns'])[:,1:3], axis=0).flatten()      # 前一张图片两个原核重心点组成的向量
    for i  in range(1,len(moments)):
        v2 = np.diff(np.array(moments[i]['pns'])[:,1:3], axis=0).flatten()  # 后一张图片两个原核重心点组成的向量
        if np.dot(v1,v2)<0:  # 两向量点积小于0表示 两原核在列表中的位置需要交换一下位置 
            moments[i]['pns'].reverse()
            v1 = 0-v2
        else:
            v1 = v2
            
def func_growth(x, b1, b2):
    # 定义面积的拟合函数
    return  b1*np.power(b2,x)
