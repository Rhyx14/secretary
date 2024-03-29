import numpy as np

def get_ap(rec,prec,use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def eval(preds,target,classes: list | tuple,threshold=0.5,use_07_metric=False,):
    """_summary_

    Args:
        preds : prediction results
            {'cat':[
                [image_id,confidence,x1,y1,x2,y2],
                [image_id,confidence,x1,y1,x2,y2],
                ...],
            'dog':[
                [image_id,confidence,x1,y1,x2,y2],
                [image_id,confidence,x1,y1,x2,y2],
                ...],
            ...}

        target : {(image_id,class):[[],]}

            EXAMPLES:
                preds = {'cat':[
                        ['image01',0.9,20,20,40,40],
                        ['image01',0.8,20,20,50,50],
                        ['image02',0.8,30,30,50,50]],
                    'dog':[
                        ['image01',0.78,60,60,90,90]
                    ]}
                target = {
                    ('image01','cat'):[[20,20,41,41]],
                    ('image01','dog'):[[60,60,91,91]],
                    ('image02','cat'):[[30,30,51,51]]}

        classes : _description_
        threshold (float, optional): _description_. Defaults to 0.5.
        use_07_metric (bool, optional): _description_. Defaults to False.
    """

    ap_per_classes={}
    aps=[]
    for i,_classname in enumerate(classes):
        pred = preds[_classname] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = 0
            print('---class {} ap {}---'.format(_classname,ap))
            aps += [ap]
            continue
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == _classname:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,_classname) in target:
                BBGT = target[(image_id,_classname)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,_classname)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = get_ap(rec, prec, use_07_metric)
        print('---class {} AP {}---'.format(_classname,ap))
        aps += [ap]

    mAP=np.mean([v for k,v in ap_per_classes.items()])
    print('---mAP {}---'.format(np.mean(aps)))
    return ap_per_classes,mAP