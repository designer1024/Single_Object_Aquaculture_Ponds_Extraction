import ee
from ee.batch import Export, Task

def collection_CP(area,startDate,endDate,flag):
    def maskEdges(s2_img):
        return s2_img.updateMask(
            s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()))
    
    def maskClouds(img):
        MAX_CLOUD_PROBABILITY=40
        clouds = ee.Image(img.get('cloud_mask')).select('probability')
        isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
        return img.updateMask(isNotCloud)

    s2=ee.ImageCollection("COPERNICUS/S2_SR")
    s2_cloud = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    CLOUDY_PIXEL_PERCENTAGE=50
    s2Sr =s2.filterDate(startDate, endDate).filterBounds(area).sort('system:time_start',flag)\
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PIXEL_PERCENTAGE))
    s2Sr=s2Sr.map(maskEdges)
    s2Clouds = s2_cloud.filterDate(startDate, endDate).filterBounds(area).sort('system:time_start',flag)
    s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
    primary=s2Sr,
    secondary=s2Clouds,
    condition=ee.Filter.equals(leftField='system:index', rightField= 'system:index')
    )
    s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask).map(maskClouds)
    return s2CloudMasked

def ndwiCollection(collection):
    def ndwi(img,i,j):
        nir = img.select(i)
        green = img.select(j)
        ndwiImg = green.subtract(nir).divide(green.add(nir))#.add(0.00000001)
        return ndwiImg.select(0).rename('ndwi')
    def _ndwi(image):
        return ndwi(image,7,3)
    ndwis=collection.map(_ndwi)
    return ndwis

def z_score_img(collection,scale):
    img_mean=collection.mean()
    img_stdDev=collection.reduce(ee.Reducer.stdDev())
    img_z=img_mean.add(img_stdDev.multiply(ee.Number(scale)))
    return img_z

def mask_adv(mask1, mask2):
    mask2=mask2.mask()
    mask3 = mask1.subtract(mask2) 
    mask3 = mask3.eq(1).selfMask() 
    return mask3

def remove_small(mask, maxSize):
    mask_small = mask.connectedComponents(
        connectedness=ee.Kernel.plus(1)
        , maxSize=maxSize
    ).select('labels')
    mask2=mask_adv(mask, mask_small)
    return mask2

def remove_hole(feature):
    g = feature.geometry()
    coor = g.coordinates()
    newcoor = coor.slice(0, 1)
    newp = ee.Algorithms.GeometryConstructors.Polygon(newcoor)
    newf = feature.setGeometry(newp)
    return newf

def get_canny(ndwi,i,scale_canny):
    if i==0:
        ndwi_t=ndwi
    else:
        ndwi_t=ndwi.focal_min(kernel=ee.Kernel.square(1))
    canny = ee.Algorithms.CannyEdgeDetector(
    image=ndwi_t,
    threshold= 0.2
    ).multiply(255)
    canny_mask=canny.updateMask(canny).reproject(crs='EPSG:4326',scale=scale_canny)
    return ndwi_t,canny_mask

def segment(water_mask, canny_mask, scale_canny, proj, area_geometry, flag):
    water_mask_seg = mask_adv(water_mask, canny_mask)
    vectors = water_mask_seg.reduceToVectors(
        reducer=ee.Reducer.countEvery()
        ,geometry=area_geometry, scale=scale_canny, geometryType='polygon', eightConnected=False
        , crs=proj        
        , maxPixels=10000000
    )
    return vectors

def segment_by_tile(water_mask, canny_mask, scale_canny, proj, area_geometry, text):
    geom_reproject = ee.Algorithms.ProjectionTransform(
        feature=area_geometry,
        proj=proj.atScale(10000),
        maxError=1
    )
    grid = geom_reproject.geometry().coveringGrid(proj, 10000)
    gridlist = grid.toList(grid.size()).getInfo()
    seglist = []
    for i, item in enumerate(gridlist):
        item = ee.Feature(item)
        img = water_mask.clip(item)
        f2 = segment(img, canny_mask, scale_canny,
                    proj, item.geometry(), str(text))
        seglist.append(f2)
    r = ee.FeatureCollection(seglist[0])
    for i in range(1, len(seglist)):
        t = ee.FeatureCollection(seglist[i])
        r = r.merge(t)
    return r

def filter_small(water_seg,scale_canny,flag):
    import math
    area_small = 300
    count_small = int(area_small / (scale_canny ** 2))
    def fun(f):
        count=ee.Number(f.get('count'))
        d_area=ee.Algorithms.If(count.gte(count_small),1,0)
        f=f.set('count',count)
        f=f.set('d_area',d_area)
        return f
    water_seg=water_seg.map(fun)
    return water_seg

def filter_area(water_seg,scale_canny,flag):
    import math
    area_small = 300
    area_big = 520000
    count_small = int(area_small / (scale_canny ** 2))
    count_big = math.ceil(area_big / (scale_canny ** 2))
    def fun(f):
        count=ee.Number(f.get('count'))
        d_area=ee.Algorithms.If(count.gte(count_small) and count.lte(count_big),1,0)
        f=f.set('count',count)
        f=f.set('d_area',d_area)
        return f
    water_seg=water_seg.map(fun)
    return water_seg

def filter_big(water_seg,scale_canny,flag):
    import math
    area_big = 520000
    count_big = math.ceil(area_big / (scale_canny ** 2))
    def fun(f):
        count=ee.Number(f.get('count'))
        d_area=ee.Algorithms.If(count.lte(count_big),1,0)
        f=f.set('count',count)
        f=f.set('d_area',d_area)
        return f
    water_seg2 = water_seg.filterMetadata('count', 'not_greater_than', count_big)
    return water_seg2

def filter_lsi(f):
    area=f.area(1)
    perimeter=f.perimeter(1)
    lsi=perimeter.divide(ee.Number(4).multiply(area.sqrt()))
    difference=ee.Algorithms.If(lsi.lte(2.5),1,0)
    f=f.set('lsi',lsi)
    f=f.set('d_lsi',difference)
    return f

def filter_perimeter_convex(f):
    perimeter=f.perimeter(1)
    c=f.convexHull(1)
    perimeter_convex=c.perimeter(1)
    difference=ee.Algorithms.If(perimeter.lte(perimeter_convex.multiply(1.5)),1,0)
    f=f.set('perimeter',perimeter)
    f=f.set('p_convex',perimeter_convex)
    f=f.set('dp_convex',difference)
    return f
  
def sdd(water_object,i,scale_canny):
    water_object=filter_small(water_object,scale_canny,'b')
    pond_not=water_object.filterMetadata('d_area','equals',0)
    water_object=water_object.filterMetadata('d_area','equals',1)
    t1=water_object.map(remove_hole)
    t2=t1.map(filter_lsi)
    t3=t2.filterMetadata('d_lsi','equals',1)
    t4=t3.map(filter_perimeter_convex)
    t5=t4.filterMetadata('dp_convex','equals',1)
    pond_object=t5
    def fun(f):
        return f.set('i',i)
    pond_object=pond_object.map(fun)
    return pond_object,pond_not

def get_near_num(fc):
    distance=100
    spatialFilter=ee.Filter.withinDistance(**{
    'distance':distance
    ,'leftField':'.geo'
    ,'rightField':'.geo'
    ,'maxError':1
    })
    saveAllJoin=ee.Join.saveAll(**{
    'matchesKey':'result'
    ,'measureKey':'dis'
    ,'ordering':'dis'
    ,'ascending':True
    })
    fc2=saveAllJoin.apply(fc,fc,spatialFilter)
    def fun(f):
        list1=ee.List(f.get('result'))
        num=list1.size()
        f=f.set('near_num',num)
        return f
    fc3=fc2.map(fun)
    def set_null(f):
        f=f.set('result',None)
        return f
    fc4=fc3.map(set_null)
    return fc4

def get_crop(fc,scale_canny):
    lulc=ee.Image(ee.ImageCollection("ESA/WorldCover/v100").first())
    cropland=lulc.where(lulc.eq(40),1).where(lulc.neq(40),0)
    obj=cropland.reduceRegions(collection=fc,reducer=ee.Reducer.sum(),scale=scale_canny)
    def fun(f):
        crop_sum=ee.Number(f.get('sum'))
        count=ee.Number(f.get('count'))
        crop_ratio=crop_sum.multiply(ee.Number(1.0)).divide(count)
        f=f.set('crop',crop_ratio)
        return f
    obj=obj.map(fun)
    return obj

def export_table_toAsset(collection,taskname,assetId):
    task= Export.table.toAsset(
        collection=collection
        ,description=taskname
        ,assetId=assetId
    )
    task.start()

def get_buffer(obj):
    def fun(f):
        i=ee.Number(f.get('i'))
        return f.buffer(ee.Number(2.5).multiply(i.add(ee.Number(1))))
    obj_buffer=obj.map(fun)
    return obj_buffer